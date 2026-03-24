"""
screen.py — minimal terminal TUI via raw ANSI escape codes.

Layout (rows top to bottom, 1-indexed):
  1 … (rows - input_rows - 1)  : scrollable output region
  (rows - input_rows)           : status bar (reverse-video)
  (rows - input_rows + 1) … rows: input line(s)

The ANSI scrolling region (CSI Ps;Ps r) confines scrolling to the output
region so that the status bar and input area never move.
"""

import os
import select
import shutil
import signal
import sys
import termios
import tty


class _SubmitException(Exception):
    """Raised internally by _handle_key() when the user presses Enter."""
    def __init__(self, value: str):
        self.value = value


class Screen:
    """Sole owner of terminal drawing for --interactive mode."""

    _PROMPT_PREFIX = "> "
    _CONT_PREFIX   = "  "   # continuation-line prefix (same width as prompt)

    def __init__(self):
        self._rows, self._cols = shutil.get_terminal_size()
        self._input_buf: list[str] = []
        self._cursor_pos: int = 0
        self._history: list[str] = []
        self._history_idx: int = -1
        self._input_rows: int = 1
        self._in_prompt: bool = False
        self._status_text: str = ""

        # Open /dev/tty so keyboard input works even when stdin is piped
        self._tty_file = open("/dev/tty", "rb+", buffering=0)
        self._tty_fd = self._tty_file.fileno()

        signal.signal(signal.SIGWINCH, self._on_resize)

        self._apply_layout()

    # ------------------------------------------------------------------ layout

    def _scroll_bottom(self) -> int:
        """Last row of the scrollable output region (1-indexed)."""
        return max(1, self._rows - self._input_rows - 1)

    def _status_row(self) -> int:
        return self._rows - self._input_rows

    def _input_start_row(self) -> int:
        return self._rows - self._input_rows + 1

    def _apply_layout(self):
        """Set scrolling region, redraw status bar, park cursor at scroll bottom."""
        sb = self._scroll_bottom()
        out = sys.stdout
        out.write(f"\033[1;{sb}r")   # set scrolling region
        self._redraw_status_raw()
        if not self._in_prompt:
            out.write(f"\033[{sb};1H")
        out.flush()

    def _redraw_status_raw(self):
        """Redraw the status bar without flushing (caller must flush)."""
        sr = self._status_row()
        text = self._status_text[: self._cols]
        sys.stdout.write(
            f"\033[s"                     # save cursor
            f"\033[{sr};1H\033[K"         # move + clear line
            f"\033[7m{text}\033[m"        # reverse-video text
            f"\033[u"                     # restore cursor
        )

    def _on_resize(self, signum, frame):
        self._rows, self._cols = shutil.get_terminal_size()
        self._apply_layout()
        if self._in_prompt:
            self._redraw_input_lines()
            sys.stdout.flush()

    # ------------------------------------------------------------------ public API

    def write(self, text: str):
        """Write text into the scrollable output area."""
        if not text:
            return
        if self._in_prompt:
            # During prompt, save/restore cursor so the input line stays intact
            sb = self._scroll_bottom()
            sys.stdout.write(
                f"\033[s"
                f"\033[{sb};1H"
                f"{text}"
                f"\033[u"
            )
        else:
            sys.stdout.write(text)
        sys.stdout.flush()

    def set_status(self, text: str):
        """Update the status bar in place."""
        self._status_text = text
        self._redraw_status_raw()
        sys.stdout.flush()

    def prompt(self, msg: str = "> ") -> str:
        """
        Draw the input line, collect user input (with editing), return the string.

        Blocking.  Raises KeyboardInterrupt on Ctrl-C, EOFError on Ctrl-D with
        an empty buffer.
        """
        self._in_prompt = True
        self._input_buf = []
        self._cursor_pos = 0
        self._input_rows = 1
        self._history_idx = -1

        old = termios.tcgetattr(self._tty_fd)
        result = None
        caught_exc = None
        try:
            tty.setraw(self._tty_fd)
            self._redraw_input_lines(msg)
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
            if caught_exc is not None:
                self._clear_input_area()
                self._input_rows = 1
                self._apply_layout()
                raise caught_exc  # re-raise after terminal restored

        return result  # type: ignore[return-value]

    def close(self):
        """Restore terminal to its normal state."""
        sys.stdout.write("\033[r")     # reset scrolling region
        sys.stdout.write("\033[?25h")  # show cursor
        sys.stdout.flush()
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        try:
            self._tty_file.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ input internals

    def _handle_key(self, key: str, msg: str):
        """Dispatch one keypress. Raises _SubmitException on Enter."""

        # ---- Submit ----
        if key in ("\r", "\n"):
            result = "".join(self._input_buf)
            if result.strip():
                self._history.insert(0, result)
            self._clear_input_area()
            self._input_rows = 1
            self._in_prompt = False
            self._apply_layout()
            # Echo submitted text into the output area
            sb = self._scroll_bottom()
            sys.stdout.write(f"\033[{sb};1H{msg}{result}\n")
            sys.stdout.flush()
            raise _SubmitException(result)

        # ---- Newline in buffer (Ctrl-Enter / Alt-Enter) ----
        if key in ("\033[13;5u", "\033\r", "\033\n"):
            self._input_buf.insert(self._cursor_pos, "\n")
            self._cursor_pos += 1
            self._update_input_rows(msg)
            return

        # ---- Backspace ----
        if key == "\x7f":
            if self._cursor_pos > 0:
                del self._input_buf[self._cursor_pos - 1]
                self._cursor_pos -= 1
                self._update_input_rows(msg)
            return

        # ---- Forward delete ----
        if key == "\033[3~":
            if self._cursor_pos < len(self._input_buf):
                del self._input_buf[self._cursor_pos]
                self._update_input_rows(msg)
            return

        # ---- Ctrl-C ----
        if key == "\x03":
            raise KeyboardInterrupt

        # ---- Ctrl-D ----
        if key == "\x04":
            if not self._input_buf:
                raise EOFError
            # With content, forward-delete
            if self._cursor_pos < len(self._input_buf):
                del self._input_buf[self._cursor_pos]
                self._update_input_rows(msg)
            return

        # ---- Arrow keys ----
        if key == "\033[A":   # up — history
            self._history_navigate(1, msg)
            return
        if key == "\033[B":   # down — history
            self._history_navigate(-1, msg)
            return
        if key == "\033[C":   # right
            if self._cursor_pos < len(self._input_buf):
                self._cursor_pos += 1
                self._redraw_input_lines(msg)
            return
        if key == "\033[D":   # left
            if self._cursor_pos > 0:
                self._cursor_pos -= 1
                self._redraw_input_lines(msg)
            return

        # ---- Home / End (keyboard or Ctrl-A/E) ----
        if key in ("\033[H", "\x01"):   # Home or Ctrl-A
            self._cursor_pos = 0
            self._redraw_input_lines(msg)
            return
        if key in ("\033[F", "\x05"):   # End or Ctrl-E
            self._cursor_pos = len(self._input_buf)
            self._redraw_input_lines(msg)
            return

        # ---- Ctrl-U (kill line) ----
        if key == "\x15":
            self._input_buf = self._input_buf[self._cursor_pos:]
            self._cursor_pos = 0
            self._update_input_rows(msg)
            return

        # ---- Ctrl-K (kill to end) ----
        if key == "\x0b":
            self._input_buf = self._input_buf[: self._cursor_pos]
            self._update_input_rows(msg)
            return

        # ---- Printable character ----
        if len(key) == 1 and ord(key) >= 32:
            self._input_buf.insert(self._cursor_pos, key)
            self._cursor_pos += 1
            self._redraw_input_lines(msg)

    def _history_navigate(self, direction: int, msg: str):
        """direction: +1 = older, -1 = newer."""
        new_idx = self._history_idx + direction
        if direction > 0 and new_idx < len(self._history):
            self._history_idx = new_idx
            self._input_buf = list(self._history[self._history_idx])
            self._cursor_pos = len(self._input_buf)
            self._update_input_rows(msg)
        elif direction < 0 and self._history_idx > 0:
            self._history_idx -= 1
            self._input_buf = list(self._history[self._history_idx])
            self._cursor_pos = len(self._input_buf)
            self._update_input_rows(msg)
        elif direction < 0 and self._history_idx == 0:
            self._history_idx = -1
            self._input_buf = []
            self._cursor_pos = 0
            self._update_input_rows(msg)

    def _update_input_rows(self, msg: str):
        """Recalculate input_rows, reapply layout if changed, then redraw."""
        buf_str = "".join(self._input_buf)
        new_rows = max(1, buf_str.count("\n") + 1)
        if new_rows != self._input_rows:
            self._input_rows = new_rows
            self._apply_layout()
        self._redraw_input_lines(msg)

    def _redraw_input_lines(self, msg: str = "> "):
        """Redraw all input lines and reposition the cursor."""
        buf_str = "".join(self._input_buf)
        lines = buf_str.split("\n")
        prefix_len = len(self._PROMPT_PREFIX)  # both prefixes are same length

        for i, line in enumerate(lines):
            row = self._input_start_row() + i
            prefix = self._PROMPT_PREFIX if i == 0 else self._CONT_PREFIX
            sys.stdout.write(f"\033[{row};1H\033[K{prefix}{line}")

        # Reposition cursor within the input area
        buf_before = "".join(self._input_buf[: self._cursor_pos])
        cur_lines = buf_before.split("\n")
        cur_row_offset = len(cur_lines) - 1
        cur_col = len(cur_lines[-1]) + prefix_len + 1   # 1-indexed
        cur_row = self._input_start_row() + cur_row_offset
        sys.stdout.write(f"\033[{cur_row};{cur_col}H")

    def _clear_input_area(self):
        for i in range(self._input_rows):
            row = self._input_start_row() + i
            sys.stdout.write(f"\033[{row};1H\033[K")
        sys.stdout.flush()

    def _read_key(self) -> str:
        """Read one logical keypress from /dev/tty. Returns string (may be multi-char)."""
        ch = os.read(self._tty_fd, 1).decode("utf-8", errors="replace")
        if ch == "\033":
            # Attempt to read the rest of the escape sequence within 50 ms
            ready, _, _ = select.select([self._tty_fd], [], [], 0.05)
            if ready:
                rest = os.read(self._tty_fd, 16).decode("utf-8", errors="replace")
                return ch + rest
        return ch
