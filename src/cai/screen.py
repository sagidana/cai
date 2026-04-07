"""
screen.py — inline TUI via raw ANSI escape codes.

Content streams directly into the normal terminal buffer so the user can
scroll back through conversation history with the terminal's own scrollback.

The only "managed" region is a two-line footer (status + prompt) that is
drawn just before input is collected and erased on Enter.  A saved-cursor
anchor (\033[s / \033[u) is used to redraw the footer in place without
knowing absolute row numbers.

Layout while prompting:
  [anchor line — blank, saved with \033[s]
  status bar   — model name / context info
  prompt line  — "> " + user input
"""

import os
import re
import select
import shutil
import signal
import sys
import termios
import threading
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
    active_style = ''

    for logical in text.split('\n'):
        current: list[str] = [active_style] if active_style else []
        col = 0
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



# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

class Screen:
    """Inline TUI for --interactive mode.  Content flows into terminal scrollback."""

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

        # Input state
        self._input_buf: list[str] = []
        self._cursor_pos: int = 0
        self._history: list[str] = []
        self._history_idx: int = -1
        self._in_prompt: bool = False
        self._current_prompt_msg: str = '> '

        # Status bar text
        self._status_text: str = ''

        # Number of rows pre-reserved below the footer anchor (updated as prompt grows)
        self._footer_rows_reserved: int = 0

        # Command completions (used for tab-completing /cmd inputs)
        self._cmd_completions: list[str] = []

        # Guard against double-close
        self._closed: bool = False

        # Threading: render lock + input listener for interrupt during streaming
        self._render_lock = threading.RLock()
        self._interrupt_event = threading.Event()
        self._listener_active = False
        self._listener_thread: threading.Thread | None = None

        # Open /dev/tty so keyboard input works even when stdin is piped
        self._tty_file = open('/dev/tty', 'rb+', buffering=0)
        self._tty_fd = self._tty_file.fileno()
        # Save "cooked" terminal attrs for vim handoff
        self._cooked_attrs = termios.tcgetattr(self._tty_fd)

        # Hide cursor initially (shown when prompt is active)
        sys.stdout.write('\033[?25l')
        sys.stdout.flush()

        signal.signal(signal.SIGWINCH, self._on_resize)

    # ------------------------------------------------------------------ footer rendering

    def _redraw_footer(self, msg: str = '> ') -> None:
        """
        Restore to the saved footer anchor, clear to end of screen, and
        redraw the status line + prompt with the current input buffer.

        Must be called only after \033[s has been written (done at the top of
        prompt() and whenever the footer is redrawn from scratch).
        """
        # Ensure enough rows are reserved below the anchor for the current footer.
        # status (1) + one row per prompt line.  When the prompt grows (backslash
        # continuation, vim paste), extend the reservation and re-save the anchor.
        buf_str = ''.join(self._input_buf)
        n_prompt_lines = buf_str.count('\n') + 1
        total_needed = 1 + n_prompt_lines   # status row + prompt rows
        if total_needed > self._footer_rows_reserved:
            # Go to current anchor, clear below, write enough newlines to push the
            # terminal (scroll if necessary), then move back up and re-anchor.
            sys.stdout.write('\033[u\r\033[J')
            sys.stdout.write('\n' * total_needed + f'\033[{total_needed}A')
            sys.stdout.write('\033[s')   # new anchor — now guaranteed space below
            self._footer_rows_reserved = total_needed

        # Return to anchor (col 1 of the blank anchor line), clear to end
        sys.stdout.write('\033[u\r\033[J')

        # Status line (one line below anchor)
        text = self._status_text[: self._cols - 1]
        sys.stdout.write(f'\n{self._STATUS_STYLE}{text}\033[K{self._RESET}')

        # Prompt line(s) — handle multi-line input (Alt-Enter / backslash)
        lines = buf_str.split('\n')
        chars_before = ''.join(self._input_buf[: self._cursor_pos])
        line_idx = chars_before.count('\n')   # which logical line the cursor is on

        for i, line in enumerate(lines):
            prefix = msg if i == 0 else self._CONT_PREFIX
            # Use \r\n, not bare \n: in raw mode \n is a pure line-feed that
            # does NOT reset the column, so without \r the prompt would start
            # at whatever column the status text ended at.
            sys.stdout.write(f'\r\n{prefix}{line}')

        # Move cursor to the correct line and column
        # After printing all lines the cursor is on the last line.
        # Move up if the edit cursor is on an earlier line.
        lines_below_cursor = len(lines) - 1 - line_idx
        if lines_below_cursor > 0:
            sys.stdout.write(f'\033[{lines_below_cursor}A')

        last_nl = chars_before.rfind('\n')
        cursor_in_line = len(chars_before) - last_nl - 1
        prefix = msg if line_idx == 0 else self._CONT_PREFIX
        col = len(prefix) + cursor_in_line   # 0-indexed offset from col 1
        if col > 0:
            sys.stdout.write(f'\r\033[{col}C')
        else:
            sys.stdout.write('\r')

    # ------------------------------------------------------------------ public API

    def write(self, text: str) -> None:
        """Write *text* directly to stdout (flows into terminal scrollback)."""
        if not text:
            return
        with self._render_lock:
            # Normalize newlines to \r\n.  The input-listener thread calls
            # tty.setraw() during LLM streaming, so a bare \n is a pure
            # line-feed that does NOT reset the column — producing progressively
            # indented output.  Normalising ensures every line starts at col 1
            # in both raw and cooked mode (the extra \r is a no-op in cooked).
            normalized = text.replace('\r\n', '\n').replace('\n', '\r\n')
            sys.stdout.write(normalized)
            sys.stdout.flush()

    def set_status(self, text: str) -> None:
        """Update the status bar text.  Redraws the footer if currently prompting."""
        self._status_text = text
        if self._in_prompt:
            with self._render_lock:
                self._redraw_footer(self._current_prompt_msg)
                sys.stdout.flush()

    def show_prompt_placeholder(self, msg: str = '> ') -> None:
        """No-op in inline mode — streaming output acts as its own indicator."""
        pass

    def prompt(self, msg: str = '> ') -> str:
        """
        Collect user input at the prompt.  Returns the entered string.

        Draws a two-line footer (status + prompt) just above the cursor,
        anchored with \033[s so it can be redrawn in-place on every keypress.
        On Enter the footer is erased and the submitted text is written
        permanently into the terminal buffer before returning.

        Blocking.  Raises KeyboardInterrupt on Ctrl-C, EOFError on Ctrl-D
        with an empty buffer.
        """
        self._in_prompt = True
        self._current_prompt_msg = msg
        self._input_buf = []
        self._cursor_pos = 0
        self._history_idx = -1

        # Pre-reserve FOOTER_ROWS lines below the cursor before saving the anchor.
        # Without this, _redraw_footer's two \n characters would scroll the terminal
        # when the cursor is at or near the bottom row, invalidating the saved position
        # and causing a spurious scroll on every subsequent keypress.
        FOOTER_ROWS = 2   # status line + one prompt line (minimum)
        self._footer_rows_reserved = FOOTER_ROWS
        sys.stdout.write(f'\n' * FOOTER_ROWS + f'\033[{FOOTER_ROWS}A')
        # Save the footer anchor.  _redraw_footer uses \033[u to return here.
        sys.stdout.write('\033[s')
        sys.stdout.flush()

        old = termios.tcgetattr(self._tty_fd)
        result = None
        caught_exc = None
        try:
            tty.setraw(self._tty_fd)
            termios.tcflush(self._tty_fd, termios.TCIFLUSH)
            sys.stdout.write('\033[?25h\033[5 q')   # show blinking bar cursor
            self._redraw_footer(msg)
            sys.stdout.flush()

            while True:
                key = self._read_key()
                try:
                    self._handle_key(key, msg)
                except _SubmitException as e:
                    result = e.value
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
                # Clear the footer on interrupt/EOF
                sys.stdout.write('\033[u\r\033[J')
                sys.stdout.flush()
                raise caught_exc

        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------ streaming listener

    def start_input_listener(self) -> None:
        """Start a daemon thread that reads interrupt keys during LLM streaming."""
        self._interrupt_event.clear()
        self._listener_active = True
        self._listener_thread = threading.Thread(
            target=self._input_listener_loop, daemon=True, name='cai-input-listener'
        )
        self._listener_thread.start()

    def stop_input_listener(self) -> None:
        """Stop the input listener thread and wait for it to exit."""
        self._listener_active = False
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=0.5)
            self._listener_thread = None

    def _input_listener_loop(self) -> None:
        """Daemon loop: read raw keys and handle interrupt during streaming."""
        old = termios.tcgetattr(self._tty_fd)
        try:
            tty.setraw(self._tty_fd)
            while self._listener_active:
                rlist, _, _ = select.select([self._tty_fd], [], [], 0.05)
                if not rlist:
                    continue
                key = self._read_key()
                self._handle_listener_key(key)
        finally:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, old)

    def _handle_listener_key(self, key: str) -> None:
        """Handle a keypress received during streaming (Ctrl-C interrupt only)."""
        if key == '\x03':    # Ctrl-C → signal interrupt
            self._interrupt_event.set()

    def close(self) -> None:
        """Restore the terminal to a clean state."""
        if self._closed:
            return
        self._closed = True
        # Show cursor, reset shape/attrs, move to a fresh line
        sys.stdout.write('\033[?25h\033[0 q\033[m\n')
        sys.stdout.flush()
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        try:
            self._tty_file.close()
        except Exception:
            pass

    def set_cmd_completions(self, cmds: list[str]) -> None:
        """Set the list of command names available for tab completion."""
        self._cmd_completions = list(cmds)

    # ------------------------------------------------------------------ vim integration

    def _open_in_vim(self, prompt_msg: str) -> None:
        """Open the current prompt buffer in vim; load the result back on exit."""
        import subprocess
        import tempfile
        content = ''.join(self._input_buf)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            tmp = f.name
        try:
            # Restore cooked terminal so vim has normal terminal control
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._cooked_attrs)
            sys.stdout.write('\033[?25h')   # show cursor for vim
            sys.stdout.flush()
            subprocess.run(['nvim', tmp])
            # Re-enter raw mode for the prompt loop
            tty.setraw(self._tty_fd)
            with open(tmp, 'r') as f:
                new_content = f.read().rstrip('\n')
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        self._input_buf = list(new_content)
        self._cursor_pos = len(self._input_buf)
        # Redraw footer with updated input
        self._redraw_footer(prompt_msg)
        sys.stdout.flush()

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

        # ---- Submit (or line continuation with \) ----
        if key in ('\r', '\n'):
            if self._cursor_pos > 0 and self._input_buf[self._cursor_pos - 1] == '\\':
                self._input_buf[self._cursor_pos - 1] = '\n'
                self._redraw_footer(msg)
                return
            result = ''.join(self._input_buf)
            if result.strip():
                self._history.insert(0, result)
            # Erase the footer and write the user's turn as permanent terminal output.
            # Use \r\n (not bare \n): this handler runs inside tty.setraw() so a
            # bare \n is a pure line-feed that does not reset the column.
            sys.stdout.write('\033[u\r\033[J')
            for i, line in enumerate(result.split('\n')):
                prefix = msg if i == 0 else self._CONT_PREFIX
                sys.stdout.write(f'{self._USER_STYLE}{prefix}{line}{self._RESET}\r\n')
            sys.stdout.write('\r\n')
            sys.stdout.flush()
            self._in_prompt = False
            raise _SubmitException(result)

        # ---- Newline in buffer (Alt-Enter) ----
        if key in ('\033\r', '\033\n'):
            self._input_buf.insert(self._cursor_pos, '\n')
            self._cursor_pos += 1
            self._redraw_footer(msg)
            return

        # ---- Backspace ----
        if key == '\x7f':
            if self._cursor_pos > 0:
                del self._input_buf[self._cursor_pos - 1]
                self._cursor_pos -= 1
                self._redraw_footer(msg)
            return

        # ---- Ctrl-W / Ctrl-Backspace — delete word before cursor ----
        if key in ('\x17', '\x08', '\033\x7f'):
            # skip trailing whitespace, then delete a word
            while self._cursor_pos > 0 and self._input_buf[self._cursor_pos - 1] in ' \t':
                del self._input_buf[self._cursor_pos - 1]
                self._cursor_pos -= 1
            while self._cursor_pos > 0 and self._input_buf[self._cursor_pos - 1] not in ' \t\n':
                del self._input_buf[self._cursor_pos - 1]
                self._cursor_pos -= 1
            self._redraw_footer(msg)
            return

        # ---- Forward delete ----
        if key == '\033[3~':
            if self._cursor_pos < len(self._input_buf):
                del self._input_buf[self._cursor_pos]
                self._redraw_footer(msg)
            return

        # ---- Ctrl-C ----
        if key == '\x03':
            if self._input_buf:
                self._input_buf.clear()
                self._cursor_pos = 0
                self._redraw_footer(msg)
                return
            raise KeyboardInterrupt

        # ---- Ctrl-V — open prompt in vim ----
        if key == '\x16':
            self._open_in_vim(msg)
            return

        # ---- Arrow keys ----
        if key == '\033[A':   # up
            all_lines = ''.join(self._input_buf).split('\n')
            before = ''.join(self._input_buf[:self._cursor_pos])
            lines_before = before.split('\n')
            cur_line = len(lines_before) - 1
            if cur_line == 0:
                self._history_navigate(1, msg)
            else:
                cur_col = len(lines_before[-1])
                target_col = min(cur_col, len(all_lines[cur_line - 1]))
                new_pos = sum(len(all_lines[i]) + 1 for i in range(cur_line - 1)) + target_col
                self._cursor_pos = new_pos
                self._redraw_footer(msg)
            return
        if key == '\033[B':   # down
            all_lines = ''.join(self._input_buf).split('\n')
            before = ''.join(self._input_buf[:self._cursor_pos])
            lines_before = before.split('\n')
            cur_line = len(lines_before) - 1
            if cur_line == len(all_lines) - 1:
                self._history_navigate(-1, msg)
            else:
                cur_col = len(lines_before[-1])
                target_col = min(cur_col, len(all_lines[cur_line + 1]))
                new_pos = sum(len(all_lines[i]) + 1 for i in range(cur_line + 1)) + target_col
                self._cursor_pos = new_pos
                self._redraw_footer(msg)
            return
        if key == '\033[C':   # right
            if self._cursor_pos < len(self._input_buf):
                self._cursor_pos += 1
                self._redraw_footer(msg)
            return
        if key == '\033[D':   # left
            if self._cursor_pos > 0:
                self._cursor_pos -= 1
                self._redraw_footer(msg)
            return

        # ---- Home / End ----
        # Home: beginning of current line; Ctrl-A: absolute beginning
        if key in ('\033[H', '\033[1~', '\033OH'):
            before = ''.join(self._input_buf[:self._cursor_pos])
            last_nl = before.rfind('\n')
            self._cursor_pos = last_nl + 1
            self._redraw_footer(msg)
            return
        if key == '\x01':   # Ctrl-A — absolute beginning
            self._cursor_pos = 0
            self._redraw_footer(msg)
            return
        # End: end of current line; Ctrl-E: absolute end
        if key in ('\033[F', '\033[4~', '\033OF'):
            rest = ''.join(self._input_buf[self._cursor_pos:])
            next_nl = rest.find('\n')
            if next_nl == -1:
                self._cursor_pos = len(self._input_buf)
            else:
                self._cursor_pos += next_nl
            self._redraw_footer(msg)
            return
        if key == '\x05':   # Ctrl-E — absolute end
            self._cursor_pos = len(self._input_buf)
            self._redraw_footer(msg)
            return

        # ---- Ctrl-K (kill to end) ----
        if key == '\x0b':
            self._input_buf = self._input_buf[: self._cursor_pos]
            self._redraw_footer(msg)
            return

        # ---- Tab — complete /command when input starts with / ----
        if key == '\t':
            self._tab_complete(msg)
            return

        # ---- Printable character ----
        if len(key) == 1 and ord(key) >= 32:
            self._input_buf.insert(self._cursor_pos, key)
            self._cursor_pos += 1
            self._redraw_footer(msg)

    def _history_navigate(self, direction: int, msg: str) -> None:
        """direction: +1 = older, -1 = newer."""
        new_idx = self._history_idx + direction
        if direction > 0 and new_idx < len(self._history):
            self._history_idx = new_idx
            self._input_buf = list(self._history[self._history_idx])
            self._cursor_pos = len(self._input_buf)
            self._redraw_footer(msg)
        elif direction < 0 and self._history_idx > 0:
            self._history_idx -= 1
            self._input_buf = list(self._history[self._history_idx])
            self._cursor_pos = len(self._input_buf)
            self._redraw_footer(msg)
        elif direction < 0 and self._history_idx == 0:
            self._history_idx = -1
            self._input_buf = []
            self._cursor_pos = 0
            self._redraw_footer(msg)

    # ------------------------------------------------------------------ tab completion

    def _tab_complete(self, msg: str) -> None:
        """Tab-complete /command when the input buffer starts with '/'."""
        buf_str = ''.join(self._input_buf)
        if not buf_str.startswith('/'):
            return
        current = buf_str[1:]   # text after the leading /
        matches = [c for c in self._cmd_completions if c.startswith(current)]
        if len(matches) == 1:
            completed = matches[0]
            self._input_buf = list(f'/{completed}')
            self._cursor_pos = len(self._input_buf)
            self._redraw_footer(msg)
        elif len(matches) > 1:
            common = matches[0]
            for m in matches[1:]:
                i = 0
                while i < len(common) and i < len(m) and common[i] == m[i]:
                    i += 1
                common = common[:i]
            if len(common) > len(current):
                self._input_buf = list(f'/{common}')
                self._cursor_pos = len(self._input_buf)
                self._redraw_footer(msg)

    # ------------------------------------------------------------------ resize

    def _on_resize(self, signum, frame) -> None:
        ts = shutil.get_terminal_size()
        self._rows, self._cols = ts.lines, ts.columns
        if self._in_prompt:
            self._redraw_footer(self._current_prompt_msg)
            sys.stdout.flush()

    # ------------------------------------------------------------------ tools overlay

    def prompt_tools_overlay(self, tool_names: list[str], enabled: set) -> set:
        """
        Interactive tools toggle overlay — centered floating box.

        Uses the alternate screen buffer for a clean canvas while the overlay
        is active; restores the main screen (with conversation history) on exit.

        Navigation : j / k / arrows
        Toggle     : Space
        Search fwd : /pattern  then Enter to confirm, n / N to cycle
        Search bwd : ?pattern  then Enter to confirm, N / n to cycle
        Close      : ESC or Enter (normal mode)
        """
        if not tool_names:
            return set(enabled)

        enabled = set(enabled)
        selected_idx = 0

        # Search state
        search_mode      = False
        search_direction = 1          # +1 = forward (/), -1 = backward (?)
        search_buf: list[str] = []
        search_pattern   = ''
        search_matches: list[int] = []
        search_match_idx = -1
        pre_search_idx   = 0          # cursor position before entering search

        _resize_pending = [False]
        _prev_lines: dict[int, str] = {}   # diff cache for flicker-free redraw
        _first_draw  = [True]              # True → full background redraw needed

        def _on_overlay_resize(signum, frame):
            ts = shutil.get_terminal_size()
            self._rows, self._cols = ts.lines, ts.columns
            _resize_pending[0] = True
            _first_draw[0]     = True      # geometry changed → full redraw

        def _find_matches(pattern: str) -> list[int]:
            if not pattern:
                return []
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error:
                rx = re.compile(re.escape(pattern), re.IGNORECASE)
            return [i for i, nm in enumerate(tool_names) if rx.search(nm)]

        def _nearest_fwd(matches: list[int], from_idx: int) -> int:
            for i, m in enumerate(matches):
                if m >= from_idx:
                    return i
            return 0

        def _nearest_bwd(matches: list[int], from_idx: int) -> int:
            for i in range(len(matches) - 1, -1, -1):
                if matches[i] <= from_idx:
                    return i
            return len(matches) - 1

        def _sync_cursor_to_search() -> None:
            nonlocal selected_idx, search_match_idx
            if search_matches:
                if search_direction == 1:
                    search_match_idx = _nearest_fwd(search_matches, pre_search_idx)
                else:
                    search_match_idx = _nearest_bwd(search_matches, pre_search_idx)
                selected_idx = search_matches[search_match_idx]
            else:
                search_match_idx = -1
                selected_idx = pre_search_idx

        def _redraw() -> None:
            self._draw_tools_overlay(
                tool_names, enabled, selected_idx,
                search_pattern, search_matches, search_match_idx,
                search_mode, search_buf, search_direction,
                _prev_lines, _first_draw[0],
            )
            _first_draw[0] = False

        old_attrs    = termios.tcgetattr(self._tty_fd)
        orig_handler = signal.getsignal(signal.SIGWINCH)

        # Enter alternate screen for a clean overlay canvas
        sys.stdout.write('\033[?1049h\033[2J')
        sys.stdout.flush()

        try:
            signal.signal(signal.SIGWINCH, _on_overlay_resize)
            tty.setraw(self._tty_fd)
            _redraw()

            prev_key = ''
            while True:
                if _resize_pending[0]:
                    _resize_pending[0] = False
                    _redraw()

                key = self._read_key()

                # ── Search input mode ────────────────────────────────────
                if search_mode:
                    if key in ('\r', '\n'):
                        search_mode = False          # confirm, cursor stays
                    elif key == '\033':
                        search_mode    = False       # cancel — restore cursor
                        selected_idx   = pre_search_idx
                        search_pattern = ''
                        search_buf     = []
                        search_matches = []
                        search_match_idx = -1
                    elif key == '\x7f':              # backspace
                        if search_buf:
                            search_buf.pop()
                            search_pattern = ''.join(search_buf)
                            search_matches = _find_matches(search_pattern)
                            _sync_cursor_to_search()
                        else:
                            search_mode    = False   # empty buf → cancel
                            selected_idx   = pre_search_idx
                            search_pattern = ''
                            search_matches = []
                            search_match_idx = -1
                    elif len(key) == 1 and ord(key) >= 32:
                        search_buf.append(key)
                        search_pattern = ''.join(search_buf)
                        search_matches = _find_matches(search_pattern)
                        _sync_cursor_to_search()
                    _redraw()
                    continue

                # ── Normal navigation mode ───────────────────────────────
                if key in ('\033', '\r', '\n', '\x03'):
                    break
                elif key in ('\033[A', 'k'):
                    selected_idx = max(0, selected_idx - 1)
                elif key in ('\033[B', 'j'):
                    selected_idx = min(len(tool_names) - 1, selected_idx + 1)
                elif key == ' ':
                    nm = tool_names[selected_idx]
                    if nm in enabled:
                        enabled.discard(nm)
                    else:
                        enabled.add(nm)
                elif key == '/':
                    search_mode      = True
                    search_direction = 1
                    search_buf       = []
                    search_pattern   = ''
                    search_matches   = []
                    search_match_idx = -1
                    pre_search_idx   = selected_idx
                elif key == '?':
                    search_mode      = True
                    search_direction = -1
                    search_buf       = []
                    search_pattern   = ''
                    search_matches   = []
                    search_match_idx = -1
                    pre_search_idx   = selected_idx
                elif key == 'n' and search_matches:
                    if search_direction == 1:
                        search_match_idx = (search_match_idx + 1) % len(search_matches)
                    else:
                        search_match_idx = (search_match_idx - 1) % len(search_matches)
                    selected_idx = search_matches[search_match_idx]
                elif key == 'N' and search_matches:
                    if search_direction == 1:
                        search_match_idx = (search_match_idx - 1) % len(search_matches)
                    else:
                        search_match_idx = (search_match_idx + 1) % len(search_matches)
                    selected_idx = search_matches[search_match_idx]
                elif key == 'G':   # jump to last tool
                    selected_idx = len(tool_names) - 1
                elif key == 'g' and prev_key == 'g':   # gg → jump to first tool
                    selected_idx = 0
                else:
                    # Ctrl-U / Ctrl-D: move cursor half a visible page
                    _overhead = 4
                    _vis = max(1, min(len(tool_names), max(5, int(self._rows * 0.85)) - _overhead))
                    _half = max(1, _vis // 2)
                    if key == '\x15':   # Ctrl-U — half page up
                        selected_idx = max(0, selected_idx - _half)
                    elif key == '\x04':   # Ctrl-D — half page down
                        selected_idx = min(len(tool_names) - 1, selected_idx + _half)

                prev_key = key
                _redraw()

        finally:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, old_attrs)
            signal.signal(signal.SIGWINCH, orig_handler)
            # Exit alternate screen (restores conversation history) and hide cursor
            sys.stdout.write('\033[?1049l\033[?25l')
            sys.stdout.flush()

        return enabled

    def _draw_tools_overlay(
        self,
        tool_names: list[str],
        enabled: set,
        selected_idx: int,
        search_pattern: str,
        search_matches: list[int],
        search_match_idx: int,
        search_mode: bool,
        search_buf: list[str],
        search_direction: int,
        prev_lines: dict,
        first_draw: bool,
    ) -> None:
        """Render the centered floating tools overlay."""
        rows, cols = self._rows, self._cols
        n = len(tool_names)

        # ── Box dimensions ────────────────────────────────────────────────
        max_name_len = max((len(nm) for nm in tool_names), default=10)
        # "  [x] <name>  " → prefix 6 chars + name + 2 padding
        content_w   = max_name_len + 8
        max_inner_w = max(20, int(cols * 0.85) - 2)
        inner_w     = max(20, min(content_w, max_inner_w))
        box_w       = inner_w + 2   # side borders

        # Layout rows: ┌─┐ [tools…] ├─┤ search/status └─┘  (no title row)
        overhead  = 4
        max_box_h = max(overhead + 1, int(rows * 0.85))
        visible_n = max(1, min(n, max_box_h - overhead))
        box_h     = visible_n + overhead

        # Center the box
        start_r = max(1, (rows - box_h) // 2 + 1)
        start_c = max(1, (cols - box_w) // 2 + 1)

        # ── Scroll window ─────────────────────────────────────────────────
        scroll = 0
        if selected_idx >= visible_n:
            scroll = selected_idx - visible_n + 1
        scroll = max(0, min(scroll, n - visible_n))

        # ── Box chars ─────────────────────────────────────────────────────
        H  = '─'; TL, TR = '┌', '┐'; BL, BR = '└', '┘'
        VL = '│'; ML, MR = '├', '┤'
        h_line = H * inner_w

        dir_char = '/' if search_direction == 1 else '?'

        # ── Build new line content (row_offset → text) ────────────────────
        new_lines: dict[int, tuple[int, str]] = {}   # row_off → (screen_row, text)

        def put(row_off: int, text: str) -> None:
            r = start_r + row_off
            if 1 <= r <= rows:
                new_lines[row_off] = (r, text)

        # ── Draw box ──────────────────────────────────────────────────────
        put(0, f'{TL}{h_line}{TR}')

        for i in range(visible_n):
            ai = i + scroll
            if ai >= n:
                put(1 + i, f'{VL}{" " * inner_w}{VL}')
                continue

            nm    = tool_names[ai]
            check = '[x]' if nm in enabled else '[ ]'
            max_nm_len = inner_w - 7
            display    = nm[:max_nm_len] if len(nm) > max_nm_len else nm
            raw_line   = f'  {check} {display}'
            cell       = raw_line[:inner_w].ljust(inner_w)

            is_sel   = (ai == selected_idx)
            is_match = bool(search_matches) and (ai in search_matches)

            if is_sel and is_match:
                styled = f'\033[7;33m{cell}\033[m'
            elif is_sel:
                styled = f'\033[7m{cell}\033[m'
            elif is_match:
                styled = f'\033[33m{cell}\033[m'
            else:
                styled = cell

            put(1 + i, f'{VL}{styled}{VL}')

        put(1 + visible_n, f'{ML}{h_line}{MR}')

        # ── Search / status bar ───────────────────────────────────────────
        enabled_count = sum(1 for nm in tool_names if nm in enabled)
        hints = '  j/k /:search ESC/↵:close'
        if search_mode:
            search_text = ''.join(search_buf)
            if search_matches:
                m_info = f' [{search_match_idx + 1}/{len(search_matches)}]'
            elif search_text:
                m_info = ' [no match]'
            else:
                m_info = ''
            raw_status  = f' {dir_char}{search_text}{m_info}'
            status_cell = raw_status[:inner_w].ljust(inner_w)
            put(1 + visible_n + 1, f'{VL}\033[7m{status_cell}\033[m{VL}')
        else:
            count_str = f' {enabled_count}/{n} enabled'
            if search_pattern:
                m_label = f' [{search_match_idx + 1}/{len(search_matches)}]' if search_matches else ''
                count_str += f'   {dir_char}{search_pattern}{m_label}'
            if len(count_str) + len(hints) <= inner_w:
                count_str += hints
            status_cell = count_str[:inner_w].ljust(inner_w)
            put(1 + visible_n + 1, f'{VL}{status_cell}{VL}')

        put(1 + visible_n + 2, f'{BL}{h_line}{BR}')

        # ── Emit only changed rows (flicker-free diff redraw) ─────────────
        out: list[str] = []

        if first_draw:
            # Full clear on first open or after resize (already in alternate screen)
            sys.stdout.write('\033[2J')
            for row_off, (r, text) in new_lines.items():
                out.append(f'\033[{r};{start_c}H{text}')
        else:
            for row_off, (r, text) in new_lines.items():
                if prev_lines.get(row_off) != text:
                    out.append(f'\033[{r};{start_c}H{text}')

        # ── Cursor visibility ─────────────────────────────────────────────
        if search_mode:
            search_text = ''.join(search_buf)
            # position: border(1) + space(1) + dir_char(1) + len(search_text)
            cursor_col = start_c + 1 + 1 + 1 + len(search_text)
            cursor_row = start_r + 1 + visible_n + 1
            out.append(f'\033[?25h\033[{cursor_row};{cursor_col}H')
        else:
            out.append('\033[?25l')

        # ── Update diff cache ─────────────────────────────────────────────
        prev_lines.clear()
        for row_off, (r, text) in new_lines.items():
            prev_lines[row_off] = text

        sys.stdout.write(''.join(out))
        sys.stdout.flush()
