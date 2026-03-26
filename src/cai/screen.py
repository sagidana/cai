"""
screen.py — full-screen TUI via raw ANSI escape codes.

Layout (rows top to bottom, 1-indexed):
  1 … (rows - 2)  : conversation view (content buffer, redrawn on resize)
  (rows - 1)      : prompt / input line
  rows            : status bar

Uses the alternate screen buffer (\033[?1049h) so the original terminal
contents are fully restored when the TUI exits.
"""

import os
import re
import select
import shutil
import signal
import sys
import termios
import tty


# ---------------------------------------------------------------------------
# Module-level ANSI helpers
# ---------------------------------------------------------------------------

# Matches CSI sequences and simple two-byte ESC sequences.
_ANSI_RE = re.compile(
    r'\033'
    r'(?:'
    r'\[[0-9;?]*[mABCDEFGHJKLMPRSTXZ@`abcdefhilnpqrstux~]'   # CSI
    r'|[@-Z\\-_]'                                               # Fe / Fs
    r')'
)


def _ansi_strip(text: str) -> str:
    """Remove ANSI escape sequences (for visual-width calculations)."""
    return _ANSI_RE.sub('', text)


def _wrap_ansi(text: str, width: int) -> list[str]:
    """
    Wrap *text* (which may contain ANSI codes) to *width* visual columns.

    ANSI sequences are preserved verbatim; they contribute zero visual width.
    When a line is broken mid-style the active SGR is closed with \\033[m at
    the break point and re-opened at the start of the continuation line.

    Returns a list of display lines (no trailing \\n).
    """
    if width <= 0:
        return text.split('\n') if text else ['']

    lines: list[str] = []

    for logical in text.split('\n'):
        current: list[str] = []
        col = 0
        active_style = ''
        i = 0
        n = len(logical)

        while i < n:
            ch = logical[i]

            if ch == '\033' and i + 1 < n:
                j = i + 1
                if logical[j] == '[':
                    # CSI sequence: ESC [ … <final-byte>
                    j += 1
                    while j < n and logical[j] not in (
                        'm', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                        'J', 'K', 'L', 'M', 'P', 'R', 'S', 'T', 'X',
                        'Z', '@', '`', 'a', 'b', 'c', 'd', 'e', 'f',
                        'h', 'i', 'l', 'n', 'p', 'q', 'r', 's', 't',
                        'u', 'x', '~',
                    ):
                        j += 1
                    if j < n:
                        j += 1  # include final byte
                    seq = logical[i:j]
                    # Track the most recent SGR sequence for style continuations
                    if seq.endswith('m'):
                        active_style = '' if seq in ('\033[m', '\033[0m') else seq
                    current.append(seq)
                    i = j
                else:
                    # Two-byte ESC sequence (Fe, Fs…)
                    current.append(logical[i:i + 2])
                    i += 2
            else:
                if col >= width:
                    # Hard-wrap: close style, emit line, start continuation
                    if active_style:
                        current.append('\033[m')
                    lines.append(''.join(current))
                    current = [active_style] if active_style else []
                    col = 0
                current.append(ch)
                col += 1
                i += 1

        if active_style and current:
            current.append('\033[m')
        lines.append(''.join(current))

    return lines


# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------

class _SubmitException(Exception):
    def __init__(self, value: str):
        self.value = value


class _CommandException(Exception):
    def __init__(self, value: str):
        self.value = value


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

class Screen:
    """Full-screen TUI for --interactive mode.  Sole owner of the terminal."""

    _PROMPT_PREFIX = '> '
    _CONT_PREFIX   = '  '   # continuation-line prefix (same visual width)

    # ANSI styles
    _USER_STYLE  = '\033[1m'      # bold      — user messages
    _LLM_STYLE   = '\033[36m'     # cyan      — LLM responses (exposed for callers)
    _META_STYLE  = '\033[2;37m'   # dim gray  — tool calls / metadata
    _ERROR_STYLE = '\033[1;31m'   # bold red  — errors
    _RESET       = '\033[m'

    # Status bar: bright azure text on dark gray background
    _STATUS_STYLE = '\033[38;5;45;48;5;238m'

    # ------------------------------------------------------------------ init

    def __init__(self):
        ts = shutil.get_terminal_size()
        self._rows: int = ts.lines
        self._cols: int = ts.columns

        # Conversation content buffer (source of truth)
        self._segments: list[str] = []
        self._display_lines: list[str] = []
        self._scroll_offset: int = 0    # lines hidden from bottom (0 = tail)
        self._follow_tail: bool = True

        # Input state
        self._input_buf: list[str] = []
        self._cursor_pos: int = 0
        self._history: list[str] = []
        self._history_idx: int = -1
        self._in_prompt: bool = False
        self._current_prompt_msg: str = '> '

        # Status bar text
        self._status_text: str = ''

        # Vim-style command mode
        self._cmd_mode: bool = False
        self._cmd_buf: list[str] = []
        self._cmd_history: list[str] = []
        self._cmd_history_idx: int = -1
        self._saved_status: str = ''
        self._cmd_completions: list[str] = []

        # Guard against double-close
        self._closed: bool = False

        # Open /dev/tty so keyboard input works even when stdin is piped
        self._tty_file = open('/dev/tty', 'rb+', buffering=0)
        self._tty_fd = self._tty_file.fileno()

        # Enter alternate screen, clear it, hide cursor
        sys.stdout.write('\033[?1049h\033[2J\033[?25l')
        sys.stdout.flush()

        signal.signal(signal.SIGWINCH, self._on_resize)

        # Draw initial (empty) status bar
        self._redraw_status()
        sys.stdout.flush()

    # ------------------------------------------------------------------ layout

    def _main_rows(self) -> int:
        """Rows available for the conversation view (rows 1 … rows-2)."""
        return max(1, self._rows - 2)

    def _prompt_row(self) -> int:
        return self._rows - 1

    def _status_row(self) -> int:
        return self._rows

    # ------------------------------------------------------------------ content buffer

    def _rebuild_display_lines(self) -> None:
        """Re-wrap all segments into display lines for the current column width."""
        raw = ''.join(self._segments)
        # cols-1 to avoid terminal auto-wrap at the last column
        self._display_lines = _wrap_ansi(raw, max(1, self._cols - 1))
        max_off = max(0, len(self._display_lines) - self._main_rows())
        if self._scroll_offset > max_off:
            self._scroll_offset = max_off

    # ------------------------------------------------------------------ rendering

    def _redraw_main_view(self) -> None:
        """Redraw the conversation area."""
        h = self._main_rows()
        total = len(self._display_lines)

        if self._follow_tail:
            start = max(0, total - h)
            end = start + h
        else:
            end = max(0, total - self._scroll_offset)
            start = max(0, end - h)

        visible = self._display_lines[start:end]

        buf: list[str] = []
        for i, line in enumerate(visible):
            row = i + 1   # 1-indexed
            buf.append(f'\033[{row};1H\033[m\033[K{line}')
        # Clear any unused rows below the content
        for i in range(len(visible), h):
            buf.append(f'\033[{i + 1};1H\033[K')

        sys.stdout.write(''.join(buf))

    def _redraw_prompt_line(self, msg: str = '> ') -> None:
        """Redraw the input row and position the cursor correctly."""
        row = self._prompt_row()
        sys.stdout.write(f'\033[{row};1H\033[m\033[K')

        buf_str = ''.join(self._input_buf)
        lines = buf_str.split('\n')

        chars_before = ''.join(self._input_buf[: self._cursor_pos])
        line_idx = chars_before.count('\n')

        current_line = lines[line_idx] if line_idx < len(lines) else ''
        prefix = msg if line_idx == 0 else self._CONT_PREFIX
        sys.stdout.write(f'{prefix}{current_line}')

        last_nl = chars_before.rfind('\n')
        cursor_in_line = len(chars_before) - last_nl - 1
        col = len(prefix) + cursor_in_line + 1   # 1-indexed
        sys.stdout.write(f'\033[{row};{col}H')

    def _clear_prompt_row(self) -> None:
        row = self._prompt_row()
        sys.stdout.write(f'\033[{row};1H\033[m\033[K')

    def _redraw_status(self) -> None:
        """Redraw the status bar (cursor lands at status row after this)."""
        sr = self._status_row()
        text = self._status_text[: self._cols - 1]
        sys.stdout.write(
            f'\033[{sr};1H\033[m'
            f'{self._STATUS_STYLE}{text}\033[K\033[m'
        )

    def _redraw_all(self) -> None:
        """Full repaint of every region."""
        sys.stdout.write('\033[2J')
        self._redraw_main_view()
        if self._in_prompt:
            self._redraw_prompt_line(self._current_prompt_msg)
        self._redraw_status()

    # ------------------------------------------------------------------ public API

    def write(self, text: str) -> None:
        """Append *text* to the conversation view and redraw."""
        if not text:
            return
        self._segments.append(text)
        self._rebuild_display_lines()
        self._redraw_main_view()
        if self._in_prompt:
            self._redraw_prompt_line(self._current_prompt_msg)
        if not self._cmd_mode:
            self._redraw_status()
        sys.stdout.flush()

    def set_status(self, text: str) -> None:
        """Update the status bar in place."""
        self._status_text = text
        if not self._cmd_mode:
            self._redraw_status()
            if self._in_prompt:
                self._redraw_prompt_line(self._current_prompt_msg)
            sys.stdout.flush()

    def show_prompt_placeholder(self, msg: str = '> ') -> None:
        """Draw the prompt prefix without entering input mode."""
        self._clear_prompt_row()
        row = self._prompt_row()
        sys.stdout.write(f'\033[{row};1H{msg}')
        sys.stdout.flush()

    def prompt(self, msg: str = '> ') -> str:
        """
        Collect user input at the prompt row.  Returns the entered string.

        Blocking.  Raises KeyboardInterrupt on Ctrl-C, EOFError on Ctrl-D
        with an empty buffer.
        """
        self._in_prompt = True
        self._current_prompt_msg = msg
        self._input_buf = []
        self._cursor_pos = 0
        self._history_idx = -1

        old = termios.tcgetattr(self._tty_fd)
        result = None
        caught_exc = None
        try:
            tty.setraw(self._tty_fd)
            termios.tcflush(self._tty_fd, termios.TCIFLUSH)
            sys.stdout.write('\033[?25h\033[5 q')   # show blinking bar cursor
            self._redraw_prompt_line(msg)
            sys.stdout.flush()

            while True:
                key = self._read_key()
                try:
                    self._handle_key(key, msg)
                except _SubmitException as e:
                    result = e.value
                    break
                except _CommandException as e:
                    result = f':{e.value}'
                    break
                sys.stdout.flush()

        except (KeyboardInterrupt, EOFError) as exc:
            caught_exc = exc
        finally:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, old)
            self._in_prompt = False
            sys.stdout.write('\033[?25l\033[0 q')   # hide cursor, reset shape
            sys.stdout.flush()
            if caught_exc is not None:
                self._clear_prompt_row()
                self._redraw_status()
                sys.stdout.flush()
                raise caught_exc

        return result  # type: ignore[return-value]

    def close(self) -> None:
        """Exit alternate screen and restore the terminal."""
        if self._closed:
            return
        self._closed = True
        # Show cursor, reset shape/attrs, exit alternate screen
        sys.stdout.write('\033[?25h\033[0 q\033[m\033[?1049l')
        sys.stdout.flush()
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        try:
            self._tty_file.close()
        except Exception:
            pass

    def set_cmd_completions(self, cmds: list[str]) -> None:
        """Set the list of command names available for tab completion."""
        self._cmd_completions = list(cmds)

    # ------------------------------------------------------------------ input internals

    def _read_key(self) -> str:
        """Read one logical keypress from /dev/tty (handles escape sequences)."""
        ch = os.read(self._tty_fd, 1).decode('utf-8', errors='replace')
        if ch == '\033':
            ready, _, _ = select.select([self._tty_fd], [], [], 0.05)
            if ready:
                rest = os.read(self._tty_fd, 16).decode('utf-8', errors='replace')
                return ch + rest
        return ch

    def _handle_key(self, key: str, msg: str) -> None:
        """Dispatch one keypress.  Raises _SubmitException on Enter."""

        # ---- Delegate to command mode ----
        if self._cmd_mode:
            self._handle_cmd_key(key)
            return

        # ---- Scroll: Page Up / Page Down (full page) ----
        if key == '\033[5~':   # Page Up
            self._follow_tail = False
            full = max(1, self._main_rows())
            max_off = max(0, len(self._display_lines) - self._main_rows())
            self._scroll_offset = min(self._scroll_offset + full, max_off)
            self._redraw_main_view()
            self._redraw_prompt_line(msg)
            return
        if key == '\033[6~':   # Page Down
            full = max(1, self._main_rows())
            self._scroll_offset = max(0, self._scroll_offset - full)
            if self._scroll_offset == 0:
                self._follow_tail = True
            self._redraw_main_view()
            self._redraw_prompt_line(msg)
            return

        # ---- Scroll: Ctrl-U / Ctrl-D (half page) ----
        if key == '\x15':   # Ctrl-U — scroll half page up
            self._follow_tail = False
            half = max(1, self._main_rows() // 2)
            max_off = max(0, len(self._display_lines) - self._main_rows())
            self._scroll_offset = min(self._scroll_offset + half, max_off)
            self._redraw_main_view()
            self._redraw_prompt_line(msg)
            return
        if key == '\x04':   # Ctrl-D — scroll half page down
            half = max(1, self._main_rows() // 2)
            self._scroll_offset = max(0, self._scroll_offset - half)
            if self._scroll_offset == 0:
                self._follow_tail = True
            self._redraw_main_view()
            self._redraw_prompt_line(msg)
            return

        # ---- Submit (or line continuation with \) ----
        if key in ('\r', '\n'):
            if self._cursor_pos > 0 and self._input_buf[self._cursor_pos - 1] == '\\':
                self._input_buf[self._cursor_pos - 1] = '\n'
                self._redraw_prompt_line(msg)
                return
            result = ''.join(self._input_buf)
            if result.strip():
                self._history.insert(0, result)
            # Echo submitted input into the conversation buffer
            for i, line in enumerate(result.split('\n')):
                prefix = msg if i == 0 else self._CONT_PREFIX
                self._segments.append(
                    f'{self._USER_STYLE}{prefix}{line}{self._RESET}\n'
                )
            self._segments.append('\n')
            self._rebuild_display_lines()
            self._in_prompt = False
            self._clear_prompt_row()
            self._redraw_main_view()
            self._redraw_status()
            sys.stdout.flush()
            raise _SubmitException(result)

        # ---- Newline in buffer (Alt-Enter) ----
        if key in ('\033\r', '\033\n'):
            self._input_buf.insert(self._cursor_pos, '\n')
            self._cursor_pos += 1
            self._redraw_prompt_line(msg)
            return

        # ---- Backspace ----
        if key == '\x7f':
            if self._cursor_pos > 0:
                del self._input_buf[self._cursor_pos - 1]
                self._cursor_pos -= 1
                self._redraw_prompt_line(msg)
            return

        # ---- Forward delete ----
        if key == '\033[3~':
            if self._cursor_pos < len(self._input_buf):
                del self._input_buf[self._cursor_pos]
                self._redraw_prompt_line(msg)
            return

        # ---- Ctrl-C ----
        if key == '\x03':
            raise KeyboardInterrupt

        # ---- Arrow keys ----
        if key == '\033[A':   # up — history
            self._history_navigate(1, msg)
            return
        if key == '\033[B':   # down — history
            self._history_navigate(-1, msg)
            return
        if key == '\033[C':   # right
            if self._cursor_pos < len(self._input_buf):
                self._cursor_pos += 1
                self._redraw_prompt_line(msg)
            return
        if key == '\033[D':   # left
            if self._cursor_pos > 0:
                self._cursor_pos -= 1
                self._redraw_prompt_line(msg)
            return

        # ---- Home / End (keyboard or Ctrl-A / Ctrl-E) ----
        if key in ('\033[H', '\x01'):
            self._cursor_pos = 0
            self._redraw_prompt_line(msg)
            return
        if key in ('\033[F', '\x05'):
            self._cursor_pos = len(self._input_buf)
            self._redraw_prompt_line(msg)
            return

        # ---- Ctrl-K (kill to end) ----
        if key == '\x0b':
            self._input_buf = self._input_buf[: self._cursor_pos]
            self._redraw_prompt_line(msg)
            return

        # ---- Enter command mode on ':' when buffer is empty ----
        if key == ':' and not self._input_buf:
            self._cmd_mode = True
            self._cmd_buf = []
            self._cmd_history_idx = -1
            self._saved_status = self._status_text
            self._update_cmd_status()
            return

        # ---- Printable character ----
        if len(key) == 1 and ord(key) >= 32:
            self._input_buf.insert(self._cursor_pos, key)
            self._cursor_pos += 1
            self._redraw_prompt_line(msg)

    def _history_navigate(self, direction: int, msg: str) -> None:
        """direction: +1 = older, -1 = newer."""
        new_idx = self._history_idx + direction
        if direction > 0 and new_idx < len(self._history):
            self._history_idx = new_idx
            self._input_buf = list(self._history[self._history_idx])
            self._cursor_pos = len(self._input_buf)
            self._redraw_prompt_line(msg)
        elif direction < 0 and self._history_idx > 0:
            self._history_idx -= 1
            self._input_buf = list(self._history[self._history_idx])
            self._cursor_pos = len(self._input_buf)
            self._redraw_prompt_line(msg)
        elif direction < 0 and self._history_idx == 0:
            self._history_idx = -1
            self._input_buf = []
            self._cursor_pos = 0
            self._redraw_prompt_line(msg)

    # ------------------------------------------------------------------ command mode

    def _update_cmd_status(self) -> None:
        """Render the command buffer into the status row and place cursor there."""
        cmd_text = ''.join(self._cmd_buf)
        text = f':{cmd_text}'[: self._cols - 1]
        sr = self._status_row()
        col = len(text) + 1   # cursor column after the command text (1-indexed)
        sys.stdout.write(
            f'\033[{sr};1H\033[m\033[K'
            f'\033[7m{text}\033[m'
            f'\033[{sr};{col}H'
        )
        sys.stdout.flush()

    def _exit_cmd_mode(self) -> None:
        """Shared teardown when leaving command mode (ESC / backspace-on-empty)."""
        self._cmd_mode = False
        self._status_text = self._saved_status
        self._redraw_status()
        if self._in_prompt:
            self._redraw_prompt_line(self._current_prompt_msg)
        sys.stdout.flush()

    def _handle_cmd_key(self, key: str) -> None:
        """Handle a keypress while in vim-style command mode."""
        if key in ('\r', '\n'):
            cmd = ''.join(self._cmd_buf).strip()
            if cmd:
                self._cmd_history.insert(0, cmd)
            self._cmd_mode = False
            self._status_text = self._saved_status
            self._redraw_status()
            if self._in_prompt:
                self._redraw_prompt_line(self._current_prompt_msg)
            sys.stdout.flush()
            raise _CommandException(cmd)

        if key == '\x7f':   # backspace
            if self._cmd_buf:
                self._cmd_buf.pop()
                self._update_cmd_status()
            else:
                self._exit_cmd_mode()
            return

        if key == '\033':   # ESC — cancel
            self._exit_cmd_mode()
            return

        if key == '\x03':   # Ctrl-C — cancel and propagate
            self._exit_cmd_mode()
            raise KeyboardInterrupt

        if key == '\033[A':   # up — older history
            self._cmd_history_navigate(1)
            return
        if key == '\033[B':   # down — newer history
            self._cmd_history_navigate(-1)
            return

        if key == '\t':
            self._cmd_tab_complete()
            return

        if len(key) == 1 and ord(key) >= 32:
            self._cmd_buf.append(key)
            self._update_cmd_status()

    def _cmd_history_navigate(self, direction: int) -> None:
        """direction: +1 = older, -1 = newer."""
        new_idx = self._cmd_history_idx + direction
        if direction > 0 and new_idx < len(self._cmd_history):
            self._cmd_history_idx = new_idx
            self._cmd_buf = list(self._cmd_history[self._cmd_history_idx])
        elif direction < 0 and self._cmd_history_idx > 0:
            self._cmd_history_idx -= 1
            self._cmd_buf = list(self._cmd_history[self._cmd_history_idx])
        elif direction < 0 and self._cmd_history_idx == 0:
            self._cmd_history_idx = -1
            self._cmd_buf = []
        self._update_cmd_status()

    def _cmd_tab_complete(self) -> None:
        """Complete to the longest unambiguous prefix among available commands."""
        current = ''.join(self._cmd_buf)
        matches = [c for c in self._cmd_completions if c.startswith(current)]
        if len(matches) == 1:
            self._cmd_buf = list(matches[0])
        elif len(matches) > 1:
            common = matches[0]
            for m in matches[1:]:
                i = 0
                while i < len(common) and i < len(m) and common[i] == m[i]:
                    i += 1
                common = common[:i]
            if len(common) > len(current):
                self._cmd_buf = list(common)
        self._update_cmd_status()

    # ------------------------------------------------------------------ resize

    def _on_resize(self, signum, frame) -> None:
        ts = shutil.get_terminal_size()
        self._rows, self._cols = ts.lines, ts.columns
        self._rebuild_display_lines()
        sys.stdout.write('\033[2J')
        self._redraw_main_view()
        if self._in_prompt:
            self._redraw_prompt_line(self._current_prompt_msg)
        self._redraw_status()
        sys.stdout.flush()

    # ------------------------------------------------------------------ tools overlay

    def prompt_tools_overlay(self, tool_names: list[str], enabled: set) -> set:
        """
        Interactive toggle list of tools.  Returns the updated enabled set.

        Navigate with up/down or j/k, toggle with Space/Enter, close with ESC.
        """
        if not tool_names:
            return set(enabled)

        enabled = set(enabled)
        selected_idx = 0

        old = termios.tcgetattr(self._tty_fd)
        try:
            tty.setraw(self._tty_fd)
            self._draw_tools_overlay(tool_names, enabled, selected_idx)

            while True:
                key = self._read_key()
                if key in ('\033', '\x03'):
                    break
                elif key in ('\033[A', 'k'):
                    selected_idx = max(0, selected_idx - 1)
                    self._draw_tools_overlay(tool_names, enabled, selected_idx)
                elif key in ('\033[B', 'j'):
                    selected_idx = min(len(tool_names) - 1, selected_idx + 1)
                    self._draw_tools_overlay(tool_names, enabled, selected_idx)
                elif key in (' ', '\r', '\n'):
                    name = tool_names[selected_idx]
                    if name in enabled:
                        enabled.discard(name)
                    else:
                        enabled.add(name)
                    self._draw_tools_overlay(tool_names, enabled, selected_idx)
        finally:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, old)
            # Restore conversation view (we are still in the alternate screen)
            sys.stdout.write('\033[2J')
            self._redraw_main_view()
            self._redraw_status()
            sys.stdout.flush()

        return enabled

    def _draw_tools_overlay(self, tool_names: list[str], enabled: set, selected_idx: int) -> None:
        """Render the full-screen tools toggle overlay."""
        rows, cols = self._rows, self._cols

        sys.stdout.write('\033[2J')

        # Title bar
        title = ' Tools  (j/k navigate   Space/Enter toggle   ESC close) '
        sys.stdout.write(f'\033[1;1H\033[7m{title[:cols].ljust(cols)}\033[m')

        list_start_row = 3
        max_visible = max(1, rows - list_start_row - 1)

        # Scroll to keep the selected item visible
        scroll_off = 0
        if selected_idx >= max_visible:
            scroll_off = selected_idx - max_visible + 1

        visible = tool_names[scroll_off: scroll_off + max_visible]
        for i, name in enumerate(visible):
            actual_idx = i + scroll_off
            row = list_start_row + i
            check = '[x]' if name in enabled else '[ ]'
            line = f'  {check} {name}'
            sys.stdout.write(f'\033[{row};1H')
            if actual_idx == selected_idx:
                sys.stdout.write(f'\033[7m{line[:cols]}\033[m')
            else:
                sys.stdout.write(line[:cols])

        # Footer: enabled count
        enabled_count = sum(1 for n in tool_names if n in enabled)
        footer = f' {enabled_count}/{len(tool_names)} tools enabled '
        sys.stdout.write(f'\033[{rows};1H\033[7m{footer[:cols].ljust(cols)}\033[m')

        sys.stdout.flush()
