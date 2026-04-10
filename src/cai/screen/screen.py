"""Screen — thin orchestrator for the inline TUI.

Content streams into the terminal's normal scrollback buffer so the user can
scroll back through conversation history.  The only managed region is a
two-line footer (status + prompt) redrawn via a saved-cursor anchor.
"""

import select
import shutil
import signal
import sys
import termios
import threading
import tty

from .ansi import (
    SGR_RESET, SGR_BOLD, SGR_CYAN, SGR_DIM_GRAY, SGR_BOLD_RED,
    SGR_AZURE_ON_DGRAY,
    CUR_SHOW, CUR_HIDE, CUR_SAVE, CUR_RESTORE,
    CURSOR_BAR, CURSOR_RESET,
    ERASE_TO_END,
)
from .state import _SubmitException
from .footer import redraw_footer, handle_resize
from .input import (
    read_key, handle_listener_key,
    get_overlay_matches, tab_complete, history_navigate,
    delete_word_before, open_in_vim,
)
from .overlays.tools import prompt_tools_overlay as _tools_overlay
from .overlays.context import prompt_context_overlay as _ctx_overlay


class Screen:
    """Inline TUI for --interactive mode.  Content flows into terminal scrollback."""

    _PROMPT_PREFIX = '> '
    _CONT_PREFIX   = '  '

    # Style constants — class-level so cli.py can access as Screen._LLM_STYLE
    _USER_STYLE   = SGR_BOLD
    _LLM_STYLE    = SGR_CYAN
    _META_STYLE   = SGR_DIM_GRAY
    _ERROR_STYLE  = SGR_BOLD_RED
    _RESET        = SGR_RESET
    _STATUS_STYLE = SGR_AZURE_ON_DGRAY

    def __init__(self):
        ts = shutil.get_terminal_size()
        self._rows: int = ts.lines
        self._cols: int = ts.columns

        self._input_buf: list[str] = []
        self._cursor_pos: int = 0
        self._history: list[str] = []
        self._history_idx: int = -1
        self._in_prompt: bool = False
        self._current_prompt_msg: str = '> '

        self._status_text: str = ''
        self._footer_rows_reserved: int = 0
        self._scroll_debt: int = 0

        self._cmd_completions: list[str] = []
        self._cmd_overlay_idx: int = -1

        self._closed: bool = False
        self._resize_pending: bool = False

        self._render_lock = threading.RLock()
        self._interrupt_event = threading.Event()
        self._listener_active = False
        self._listener_thread: 'threading.Thread | None' = None

        self._tty_file = open('/dev/tty', 'rb+', buffering=0)
        self._tty_fd = self._tty_file.fileno()
        self._cooked_attrs = termios.tcgetattr(self._tty_fd)

        sys.stdout.write(CUR_HIDE)
        sys.stdout.flush()

        signal.signal(signal.SIGWINCH, self._on_resize)

    # ── Public API ────────────────────────────────────────────────────────────

    def write(self, text: str) -> None:
        """Write *text* to stdout (flows into terminal scrollback)."""
        if not text:
            return
        with self._render_lock:
            # Normalize \n → \r\n: in raw mode a bare \n does not reset the column.
            normalized = text.replace('\r\n', '\n').replace('\n', '\r\n')
            sys.stdout.write(normalized)
            sys.stdout.flush()

    def set_status(self, text: str) -> None:
        """Update status bar text; triggers redraw on next prompt-loop tick."""
        self._status_text = text
        if self._in_prompt:
            self._resize_pending = True

    def show_prompt_placeholder(self, msg: str = '> ') -> None:
        """No-op in inline mode — streaming output acts as its own indicator."""

    def set_cmd_completions(self, cmds: list[str]) -> None:
        """Set the command names available for tab completion."""
        self._cmd_completions = list(cmds)

    def close(self) -> None:
        """Restore the terminal to a clean state."""
        if self._closed:
            return
        self._closed = True
        sys.stdout.write(f'{CUR_SHOW}{CURSOR_RESET}{SGR_RESET}\n')
        sys.stdout.flush()
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        try:
            self._tty_file.close()
        except Exception:
            pass

    def prompt(self, msg: str = '> ') -> str:
        """Collect user input. Blocking. Raises KeyboardInterrupt / EOFError."""
        self._in_prompt = True
        self._current_prompt_msg = msg
        self._input_buf = []
        self._cursor_pos = 0
        self._history_idx = -1

        FOOTER_ROWS = 2
        self._footer_rows_reserved = FOOTER_ROWS
        self._scroll_debt = 0
        sys.stdout.write(f'\n' * FOOTER_ROWS + f'\033[{FOOTER_ROWS}A')
        sys.stdout.write(CUR_SAVE)
        sys.stdout.flush()

        old = termios.tcgetattr(self._tty_fd)
        result     = None
        caught_exc = None
        try:
            tty.setraw(self._tty_fd)
            termios.tcflush(self._tty_fd, termios.TCIFLUSH)
            sys.stdout.write(f'{CUR_SHOW}{CURSOR_BAR}')
            self._redraw_footer(msg)
            sys.stdout.flush()

            while True:
                if self._resize_pending:
                    self._resize_pending = False
                    self._handle_resize_in_prompt(msg)

                rlist, _, _ = select.select([self._tty_fd], [], [], 0.05)
                if not rlist:
                    continue

                key = read_key(self._tty_fd)
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
            sys.stdout.write(f'{CUR_HIDE}{CURSOR_RESET}')
            sys.stdout.flush()
            if caught_exc is not None:
                sys.stdout.write(f'{CUR_RESTORE}\r{ERASE_TO_END}')
                sys.stdout.flush()
                raise caught_exc

        return result  # type: ignore[return-value]

    # ── Overlay entry points ──────────────────────────────────────────────────

    def prompt_tools_overlay(self, tool_names: list[str], enabled: set) -> set:
        """Interactive tools toggle overlay. See overlays/tools.py for docs."""
        return _tools_overlay(self, tool_names, enabled)

    def prompt_context_overlay(
        self, messages: list, context_size: int = 0, prompt_tokens: int = 0
    ) -> tuple:
        """Interactive context viewer/editor. See overlays/context.py for docs."""
        return _ctx_overlay(self, messages, context_size, prompt_tokens)

    # ── Streaming listener ────────────────────────────────────────────────────

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
        old = termios.tcgetattr(self._tty_fd)
        try:
            tty.setraw(self._tty_fd)
            while self._listener_active:
                rlist, _, _ = select.select([self._tty_fd], [], [], 0.05)
                if not rlist:
                    continue
                key = read_key(self._tty_fd)
                handle_listener_key(key, self._interrupt_event)
        finally:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, old)

    # ── Footer ────────────────────────────────────────────────────────────────

    def _redraw_footer(self, msg: str = '> ') -> None:
        self._footer_rows_reserved, self._scroll_debt = redraw_footer(
            input_buf=self._input_buf,
            cursor_pos=self._cursor_pos,
            rows=self._rows,
            cols=self._cols,
            status_text=self._status_text,
            cmd_overlay_idx=self._cmd_overlay_idx,
            overlay_matches=get_overlay_matches(
                ''.join(self._input_buf), self._cmd_completions
            ),
            footer_rows_reserved=self._footer_rows_reserved,
            scroll_debt=self._scroll_debt,
            prompt_prefix=msg,
            cont_prefix=self._CONT_PREFIX,
            status_style=self._STATUS_STYLE,
        )

    def _handle_resize_in_prompt(self, msg: str) -> None:
        self._footer_rows_reserved, self._scroll_debt = handle_resize(
            input_buf=self._input_buf,
            cursor_pos=self._cursor_pos,
            rows=self._rows,
            cols=self._cols,
            status_text=self._status_text,
            cmd_overlay_idx=self._cmd_overlay_idx,
            overlay_matches=get_overlay_matches(
                ''.join(self._input_buf), self._cmd_completions
            ),
            scroll_debt=self._scroll_debt,
            prompt_prefix=msg,
            cont_prefix=self._CONT_PREFIX,
            status_style=self._STATUS_STYLE,
        )
        sys.stdout.flush()

    # ── Key dispatch ──────────────────────────────────────────────────────────

    def _handle_key(self, key: str, msg: str) -> None:
        """Dispatch one keypress. Raises _SubmitException on Enter."""

        if key in ('\r', '\n'):
            self._handle_submit(msg)
            return

        if key in ('\033\r', '\033\n'):
            self._input_buf.insert(self._cursor_pos, '\n')
            self._cursor_pos += 1
            self._redraw_footer(msg)
            return

        if key == '\x7f':   # Backspace
            if self._cursor_pos > 0:
                del self._input_buf[self._cursor_pos - 1]
                self._cursor_pos -= 1
                self._cmd_overlay_idx = -1
                self._redraw_footer(msg)
            return

        if key in ('\x17', '\x08', '\033\x7f'):   # Ctrl-W / Ctrl-Backspace
            self._input_buf, self._cursor_pos = delete_word_before(
                self._input_buf, self._cursor_pos
            )
            self._cmd_overlay_idx = -1
            self._redraw_footer(msg)
            return

        if key == '\033[3~':   # Forward delete
            if self._cursor_pos < len(self._input_buf):
                del self._input_buf[self._cursor_pos]
                self._redraw_footer(msg)
            return

        if key == '\x03':   # Ctrl-C
            if self._input_buf:
                self._input_buf.clear()
                self._cursor_pos = 0
                self._cmd_overlay_idx = -1
                self._redraw_footer(msg)
                return
            raise KeyboardInterrupt

        if key == '\x16':   # Ctrl-V — open in vim
            new_buf = open_in_vim(self._tty_fd, self._cooked_attrs, self._input_buf)
            self._input_buf  = new_buf
            self._cursor_pos = len(new_buf)
            self._redraw_footer(msg)
            sys.stdout.flush()
            return

        if key == '\033':   # Escape — deselect overlay item
            if self._cmd_overlay_idx >= 0:
                self._cmd_overlay_idx = -1
                self._redraw_footer(msg)
            return

        if key == '\033[A':   # Arrow up
            self._handle_arrow_up(msg)
            return

        if key == '\033[B':   # Arrow down
            self._handle_arrow_down(msg)
            return

        if key == '\033[C':   # Arrow right
            if self._cursor_pos < len(self._input_buf):
                self._cursor_pos += 1
                self._redraw_footer(msg)
            return

        if key == '\033[D':   # Arrow left
            if self._cursor_pos > 0:
                self._cursor_pos -= 1
                self._redraw_footer(msg)
            return

        if key in ('\033[H', '\033[1~', '\033OH'):   # Home
            before = ''.join(self._input_buf[:self._cursor_pos])
            self._cursor_pos = before.rfind('\n') + 1
            self._redraw_footer(msg)
            return

        if key == '\x01':   # Ctrl-A — absolute beginning
            self._cursor_pos = 0
            self._redraw_footer(msg)
            return

        if key in ('\033[F', '\033[4~', '\033OF'):   # End
            rest = ''.join(self._input_buf[self._cursor_pos:])
            next_nl = rest.find('\n')
            self._cursor_pos = (len(self._input_buf) if next_nl == -1
                                else self._cursor_pos + next_nl)
            self._redraw_footer(msg)
            return

        if key == '\x05':   # Ctrl-E — absolute end
            self._cursor_pos = len(self._input_buf)
            self._redraw_footer(msg)
            return

        if key == '\x0b':   # Ctrl-K — kill to end
            self._input_buf = self._input_buf[:self._cursor_pos]
            self._redraw_footer(msg)
            return

        if key == '\t':   # Tab
            self._tab_complete(msg)
            return

        if len(key) == 1 and ord(key) >= 32:
            self._input_buf.insert(self._cursor_pos, key)
            self._cursor_pos += 1
            self._cmd_overlay_idx = -1
            self._redraw_footer(msg)

    def _handle_submit(self, msg: str) -> None:
        # Line continuation with backslash
        if self._cursor_pos > 0 and self._input_buf[self._cursor_pos - 1] == '\\':
            self._input_buf[self._cursor_pos - 1] = '\n'
            self._redraw_footer(msg)
            return
        # If an overlay item is selected, complete to it before submitting
        matches = get_overlay_matches(''.join(self._input_buf), self._cmd_completions)
        if 0 <= self._cmd_overlay_idx < len(matches):
            self._input_buf   = list(f'/{matches[self._cmd_overlay_idx]}')
            self._cursor_pos  = len(self._input_buf)
            self._cmd_overlay_idx = -1
        result = ''.join(self._input_buf)
        if result.strip():
            self._history.insert(0, result)
        # Erase footer; write permanent user-turn output
        sys.stdout.write(f'{CUR_RESTORE}\r{ERASE_TO_END}')
        for i, line in enumerate(result.split('\n')):
            prefix = msg if i == 0 else self._CONT_PREFIX
            sys.stdout.write(f'{self._USER_STYLE}{prefix}{line}{self._RESET}\r\n')
        sys.stdout.write('\r\n')
        sys.stdout.flush()
        self._in_prompt = False
        raise _SubmitException(result)

    def _handle_arrow_up(self, msg: str) -> None:
        matches = get_overlay_matches(''.join(self._input_buf), self._cmd_completions)
        if matches:
            self._cmd_overlay_idx = (self._cmd_overlay_idx - 1) % len(matches)
            self._redraw_footer(msg)
            return
        all_lines    = ''.join(self._input_buf).split('\n')
        before       = ''.join(self._input_buf[:self._cursor_pos])
        lines_before = before.split('\n')
        cur_line     = len(lines_before) - 1
        if cur_line == 0:
            idx, buf, pos = history_navigate(
                1, self._history, self._history_idx, self._input_buf, self._cursor_pos
            )
            self._history_idx = idx
            self._input_buf   = buf
            self._cursor_pos  = pos
        else:
            cur_col    = len(lines_before[-1])
            target_col = min(cur_col, len(all_lines[cur_line - 1]))
            self._cursor_pos = (
                sum(len(all_lines[i]) + 1 for i in range(cur_line - 1)) + target_col
            )
        self._redraw_footer(msg)

    def _handle_arrow_down(self, msg: str) -> None:
        matches = get_overlay_matches(''.join(self._input_buf), self._cmd_completions)
        if matches:
            self._cmd_overlay_idx = (self._cmd_overlay_idx + 1) % len(matches)
            self._redraw_footer(msg)
            return
        all_lines    = ''.join(self._input_buf).split('\n')
        before       = ''.join(self._input_buf[:self._cursor_pos])
        lines_before = before.split('\n')
        cur_line     = len(lines_before) - 1
        if cur_line == len(all_lines) - 1:
            idx, buf, pos = history_navigate(
                -1, self._history, self._history_idx, self._input_buf, self._cursor_pos
            )
            self._history_idx = idx
            self._input_buf   = buf
            self._cursor_pos  = pos
        else:
            cur_col    = len(lines_before[-1])
            target_col = min(cur_col, len(all_lines[cur_line + 1]))
            self._cursor_pos = (
                sum(len(all_lines[i]) + 1 for i in range(cur_line + 1)) + target_col
            )
        self._redraw_footer(msg)

    def _tab_complete(self, msg: str) -> None:
        buf_str = ''.join(self._input_buf)
        new_buf, new_idx = tab_complete(buf_str, self._cmd_completions, self._cmd_overlay_idx)
        if new_buf is not None:
            self._input_buf       = list(new_buf)
            self._cursor_pos      = len(self._input_buf)
            self._cmd_overlay_idx = new_idx
            self._redraw_footer(msg)

    # ── Resize ────────────────────────────────────────────────────────────────

    def _on_resize(self, signum, frame) -> None:
        # Set flag only — never do I/O in a signal handler.
        ts = shutil.get_terminal_size()
        self._rows, self._cols = ts.lines, ts.columns
        self._resize_pending = True
