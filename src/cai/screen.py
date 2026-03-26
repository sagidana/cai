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


class _CommandException(Exception):
    """Raised internally by _handle_cmd_key() when a vim-style command is submitted."""
    def __init__(self, value: str):
        self.value = value


class Screen:
    """Sole owner of terminal drawing for --interactive mode."""

    _PROMPT_PREFIX = "> "
    _CONT_PREFIX   = "  "   # continuation-line prefix (same width as prompt)

    # ANSI styles
    _USER_STYLE  = "\033[1m"      # bold  — user messages
    _LLM_STYLE   = "\033[36m"     # cyan  — LLM responses
    _META_STYLE  = "\033[2;37m"   # dim gray — tool calls / metadata
    _ERROR_STYLE = "\033[1;31m"   # bold red — errors
    _RESET       = "\033[m"

    def __init__(self):
        ts = shutil.get_terminal_size()
        self._rows, self._cols = ts.lines, ts.columns
        self._input_buf: list[str] = []
        self._cursor_pos: int = 0
        self._history: list[str] = []
        self._history_idx: int = -1
        self._input_rows: int = 1
        self._in_prompt: bool = False
        self._status_text: str = ""

        # Vim-style command mode state
        self._cmd_mode: bool = False
        self._cmd_buf: list[str] = []
        self._cmd_history: list[str] = []
        self._cmd_history_idx: int = -1
        self._saved_status: str = ""
        self._cmd_completions: list[str] = []

        # Open /dev/tty so keyboard input works even when stdin is piped
        self._tty_file = open("/dev/tty", "rb+", buffering=0)
        self._tty_fd = self._tty_file.fileno()

        signal.signal(signal.SIGWINCH, self._on_resize)

        self._apply_layout()
        sys.stdout.write("\033[?25l")   # hide cursor by default
        sys.stdout.flush()

    # ------------------------------------------------------------------ layout

    def _scroll_bottom(self) -> int:
        """Last row of the scrollable output region (1-indexed)."""
        return max(1, self._rows - self._input_rows - 1)

    def _status_row(self) -> int:
        """Status bar is always pinned to the very last row."""
        return self._rows

    def _input_start_row(self) -> int:
        """Input area sits immediately above the status bar."""
        return self._rows - self._input_rows

    def _apply_layout(self):
        """Set scrolling region, redraw status bar, park cursor at scroll bottom."""
        sb = self._scroll_bottom()
        out = sys.stdout
        out.write(f"\033[1;{sb}r")   # set scrolling region
        self._redraw_status_raw()
        if not self._in_prompt:
            out.write(f"\033[{sb};1H")
        out.flush()

    # Style for the status bar: bright azure on dark gray, no reverse-video
    _STATUS_STYLE = "\033[38;5;45;48;5;238m"

    def _redraw_status_raw(self):
        """Redraw the status bar without flushing (caller must flush)."""
        sr = self._status_row()
        text = self._status_text[: self._cols].ljust(self._cols)
        sys.stdout.write(
            f"\033[s"                                         # save cursor
            f"\033[{sr};1H\033[m"                            # move, reset attrs
            f"{self._STATUS_STYLE}{text}\033[m"              # styled text (full-width)
            f"\033[u"                                         # restore cursor
        )

    def _on_resize(self, signum, frame):
        old_sb = self._scroll_bottom()

        ts = shutil.get_terminal_size()
        self._rows, self._cols = ts.lines, ts.columns
        if self._in_prompt:
            self._input_rows = self._calc_input_rows()

        # Move to the first row below the old scroll area (where old input/
        # status rows lived) and erase everything from there to the screen
        # bottom.  This removes stale input/status content without touching
        # the scrollable conversation above.  We do NOT reset the scroll
        # region first (\033[r) because that clears the screen on some
        # terminals (Windows Terminal / ConPTY).
        sys.stdout.write(f"\033[{old_sb + 1};1H\033[J")

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

    def show_prompt_placeholder(self, msg: str = "> "):
        """Draw the prompt prefix in the input row without entering input mode."""
        row = self._input_start_row()
        sys.stdout.write(
            f"\033[s"
            f"\033[{row};1H\033[m\033[K{msg}"
            f"\033[u"
        )
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
            termios.tcflush(self._tty_fd, termios.TCIFLUSH)  # discard keys typed during LLM response
            sys.stdout.write("\033[?25h\033[5 q")  # show blinking bar cursor
            self._redraw_input_lines(msg)
            sys.stdout.flush()

            while True:
                key = self._read_key()
                try:
                    self._handle_key(key, msg)
                except _SubmitException as e:
                    result = e.value
                    break
                except _CommandException as e:
                    result = f":{e.value}"
                    break
                sys.stdout.flush()

        except (KeyboardInterrupt, EOFError) as exc:
            caught_exc = exc
        finally:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, old)
            self._in_prompt = False
            sys.stdout.write("\033[?25l\033[0 q")  # hide cursor, reset shape
            sys.stdout.flush()
            if caught_exc is not None:
                self._clear_input_area()
                self._input_rows = 1
                self._apply_layout()
                raise caught_exc  # re-raise after terminal restored

        return result  # type: ignore[return-value]

    def close(self):
        """Restore terminal to its normal state and clear the screen."""
        sys.stdout.write("\033[m")     # reset all attributes (color, bold, etc.)
        sys.stdout.write("\033[0 q")   # reset cursor shape to terminal default
        sys.stdout.write("\033[r")     # reset scrolling region
        # Clear status line and move cursor to bottom-left
        sr = self._status_row()
        sys.stdout.write(f"\033[{sr};1H\033[m\033[K")  # move to status row, clear it
        sys.stdout.write(f"\033[{sr};1H")               # park cursor at bottom-left
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

        # ---- Delegate to command mode handler ----
        if self._cmd_mode:
            self._handle_cmd_key(key)
            return

        # ---- Submit (or newline if line ends with \) ----
        if key in ("\r", "\n"):
            # If the character immediately before the cursor is \, replace it
            # with a newline instead of submitting (shell-style line continuation).
            if self._cursor_pos > 0 and self._input_buf[self._cursor_pos - 1] == "\\":
                self._input_buf[self._cursor_pos - 1] = "\n"
                self._update_input_rows(msg)
                return
            result = "".join(self._input_buf)
            if result.strip():
                self._history.insert(0, result)
            self._clear_input_area()
            self._input_rows = 1
            self._in_prompt = False
            self._apply_layout()
            # Echo submitted text into the output area.
            # Write each logical line separately so every \n causes a proper
            # scroll within the output region instead of spilling into the
            # input/status rows.
            sb = self._scroll_bottom()
            for i, line in enumerate(result.split("\n")):
                prefix = msg if i == 0 else self._CONT_PREFIX
                sys.stdout.write(
                    f"\033[{sb};1H{self._USER_STYLE}{prefix}{line}{self._RESET}\n"
                )
            # blank line between user message and LLM response
            sys.stdout.write(f"\033[{sb};1H\n")
            sys.stdout.flush()
            raise _SubmitException(result)

        # ---- Newline in buffer (Alt-Enter) ----
        if key in ("\033\r", "\033\n"):
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

        # ---- Enter command mode on ':' when buffer is empty ----
        if key == ":" and not self._input_buf:
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

    def _calc_input_rows(self) -> int:
        """Count the visual terminal rows needed for the current input buffer."""
        buf_str = "".join(self._input_buf)
        logical_lines = buf_str.split("\n")
        total = 0
        for i, line in enumerate(logical_lines):
            prefix = self._PROMPT_PREFIX if i == 0 else self._CONT_PREFIX
            combined = len(prefix) + len(line)
            total += max(1, (combined + self._cols - 1) // self._cols) if combined else 1
        return max(1, total)

    def _update_input_rows(self, msg: str):
        """Recalculate input_rows (accounting for visual wrapping), reapply layout if changed."""
        new_rows = self._calc_input_rows()
        if new_rows != self._input_rows:
            delta = new_rows - self._input_rows
            if delta > 0:
                # Scroll the output region up by delta lines so the content above
                # the input area is preserved rather than overwritten.
                sb = self._scroll_bottom()   # bottom of current (old) scroll region
                sys.stdout.write(f"\033[{sb};1H")
                sys.stdout.write("\n" * delta)
            elif delta < 0:
                # Reverse-scroll the output region to recover lines that were
                # pushed into the scrollback when the input area grew.
                # \033M (Reverse Index) at the top margin scrolls the region
                # down by one line, pulling scrollback content back in.
                sys.stdout.write("\033[1;1H")        # move to top of current scroll region
                sys.stdout.write("\033M" * (-delta)) # recover scrollback lines
                # Clear the rows being released back to the output region so
                # stale input content doesn't remain visible there.
                old_input_start = self._input_start_row()
                new_input_start = self._rows - new_rows
                for row in range(old_input_start, new_input_start):
                    sys.stdout.write(f"\033[{row};1H\033[m\033[K")
            self._input_rows = new_rows
            self._apply_layout()
        self._redraw_input_lines(msg)

    def _redraw_input_lines(self, msg: str = "> "):
        """Redraw all input lines and reposition the cursor."""
        # Clear all rows in the input area before redrawing to avoid stale content
        # from previous larger inputs or visual-wrap overflow.
        # Reset attributes first so no inherited color bleeds into the prompt.
        for i in range(self._input_rows):
            sys.stdout.write(f"\033[{self._input_start_row() + i};1H\033[m\033[K")

        buf_str = "".join(self._input_buf)
        lines = buf_str.split("\n")

        # Draw each logical line; the terminal handles visual wrapping automatically.
        draw_row = self._input_start_row()
        for i, line in enumerate(lines):
            prefix = self._PROMPT_PREFIX if i == 0 else self._CONT_PREFIX
            combined = len(prefix) + len(line)
            sys.stdout.write(f"\033[{draw_row};1H{prefix}{line}")
            draw_row += max(1, (combined + self._cols - 1) // self._cols) if combined else 1

        # Reposition cursor accounting for visual wrapping.
        buf_before = "".join(self._input_buf[: self._cursor_pos])
        cur_lines = buf_before.split("\n")
        cur_row = self._input_start_row()
        for i, line in enumerate(cur_lines[:-1]):
            prefix = self._PROMPT_PREFIX if i == 0 else self._CONT_PREFIX
            combined = len(prefix) + len(line)
            cur_row += max(1, (combined + self._cols - 1) // self._cols) if combined else 1
        # Cursor position within the last logical line
        last_idx = len(cur_lines) - 1
        last_line = cur_lines[-1]
        prefix = self._PROMPT_PREFIX if last_idx == 0 else self._CONT_PREFIX
        n = len(prefix) + len(last_line)   # chars written on this visual segment
        cur_row += n // self._cols
        cur_col = n % self._cols + 1       # 1-indexed
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

    # ------------------------------------------------------------------ command mode

    def set_cmd_completions(self, cmds: list[str]):
        """Set the list of available command names for tab completion."""
        self._cmd_completions = list(cmds)

    def _update_cmd_status(self):
        """Render current command buffer into the status bar."""
        cmd_text = "".join(self._cmd_buf)
        text = f":{cmd_text}"[: self._cols].ljust(self._cols)
        sr = self._status_row()
        sys.stdout.write(
            f"\033[s"
            f"\033[{sr};1H\033[m\033[K"
            f"\033[7m{text}\033[m"
            f"\033[u"
        )
        sys.stdout.flush()

    def _handle_cmd_key(self, key: str):
        """Handle keypress while in vim-style command mode."""
        if key in ("\r", "\n"):
            cmd = "".join(self._cmd_buf).strip()
            if cmd:
                self._cmd_history.insert(0, cmd)
            self._cmd_mode = False
            self._status_text = self._saved_status
            self._redraw_status_raw()
            sys.stdout.flush()
            raise _CommandException(cmd)

        if key == "\x7f":  # backspace
            if self._cmd_buf:
                self._cmd_buf.pop()
                self._update_cmd_status()
            else:
                # empty buffer — exit command mode
                self._cmd_mode = False
                self._status_text = self._saved_status
                self._redraw_status_raw()
                sys.stdout.flush()
            return

        if key == "\033":  # plain ESC — cancel command mode
            self._cmd_mode = False
            self._status_text = self._saved_status
            self._redraw_status_raw()
            sys.stdout.flush()
            return

        if key == "\x03":  # Ctrl-C — cancel and propagate
            self._cmd_mode = False
            self._status_text = self._saved_status
            self._redraw_status_raw()
            sys.stdout.flush()
            raise KeyboardInterrupt

        if key == "\033[A":  # up — older command history
            self._cmd_history_navigate(1)
            return
        if key == "\033[B":  # down — newer command history
            self._cmd_history_navigate(-1)
            return

        if key == "\t":  # tab completion
            self._cmd_tab_complete()
            return

        if len(key) == 1 and ord(key) >= 32:
            self._cmd_buf.append(key)
            self._update_cmd_status()

    def _cmd_history_navigate(self, direction: int):
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

    def _cmd_tab_complete(self):
        """Complete the current command buffer to the longest unambiguous prefix."""
        current = "".join(self._cmd_buf)
        matches = [c for c in self._cmd_completions if c.startswith(current)]
        if len(matches) == 1:
            self._cmd_buf = list(matches[0])
        elif len(matches) > 1:
            # longest common prefix among matches
            common = matches[0]
            for m in matches[1:]:
                i = 0
                while i < len(common) and i < len(m) and common[i] == m[i]:
                    i += 1
                common = common[:i]
            if len(common) > len(current):
                self._cmd_buf = list(common)
        self._update_cmd_status()

    # ------------------------------------------------------------------ tools overlay

    def prompt_tools_overlay(self, tool_names: list[str], enabled: set) -> set:
        """Show an interactive toggle list of tools. Returns updated enabled set.

        Uses the alternate screen buffer so the conversation is preserved.
        Navigate with up/down, toggle with Space or Enter, close with ESC.
        """
        if not tool_names:
            return set(enabled)

        enabled = set(enabled)
        selected_idx = 0

        # Switch to alternate screen buffer — preserves the main screen content.
        sys.stdout.write("\033[?1049h")
        sys.stdout.flush()

        old = termios.tcgetattr(self._tty_fd)
        try:
            tty.setraw(self._tty_fd)
            self._draw_tools_overlay(tool_names, enabled, selected_idx)

            while True:
                key = self._read_key()
                if key == "\033":  # ESC — close
                    break
                elif key == "\x03":  # Ctrl-C — close
                    break
                elif key in ("\033[A", "k"):  # up
                    selected_idx = max(0, selected_idx - 1)
                    self._draw_tools_overlay(tool_names, enabled, selected_idx)
                elif key in ("\033[B", "j"):  # down
                    selected_idx = min(len(tool_names) - 1, selected_idx + 1)
                    self._draw_tools_overlay(tool_names, enabled, selected_idx)
                elif key in (" ", "\r", "\n"):  # toggle
                    name = tool_names[selected_idx]
                    if name in enabled:
                        enabled.discard(name)
                    else:
                        enabled.add(name)
                    self._draw_tools_overlay(tool_names, enabled, selected_idx)
        finally:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, old)
            # Return to the main screen buffer, restoring the conversation.
            sys.stdout.write("\033[?1049l")
            sys.stdout.flush()

        return enabled

    def _draw_tools_overlay(self, tool_names: list[str], enabled: set, selected_idx: int):
        """Render the full-screen tools toggle overlay."""
        rows, cols = self._rows, self._cols

        sys.stdout.write("\033[2J")  # clear screen

        # Title bar
        title = " Tools  (j/k navigate   Space/Enter toggle   ESC close) "
        sys.stdout.write(f"\033[1;1H\033[7m{title[:cols].ljust(cols)}\033[m")

        list_start_row = 3
        max_visible = max(1, rows - list_start_row - 1)

        # Scroll offset to keep selected item visible
        scroll_offset = 0
        if selected_idx >= max_visible:
            scroll_offset = selected_idx - max_visible + 1

        visible = tool_names[scroll_offset: scroll_offset + max_visible]

        for i, name in enumerate(visible):
            actual_idx = i + scroll_offset
            row = list_start_row + i
            check = "[x]" if name in enabled else "[ ]"
            line = f"  {check} {name}"
            sys.stdout.write(f"\033[{row};1H")
            if actual_idx == selected_idx:
                sys.stdout.write(f"\033[7m{line[:cols]}\033[m")
            else:
                sys.stdout.write(line[:cols])

        # Footer: enabled count
        enabled_count = sum(1 for n in tool_names if n in enabled)
        footer = f" {enabled_count}/{len(tool_names)} tools enabled "
        sys.stdout.write(f"\033[{rows};1H\033[7m{footer[:cols].ljust(cols)}\033[m")

        sys.stdout.flush()
