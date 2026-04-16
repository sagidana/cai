"""Mode state machine and per-mode key handlers for the TUI."""

from .ansi import (
    KEY_BACKSPACE, KEY_ESC, KEY_ENTER, KEY_ALT_ENTER,
    KEY_CTRL_W, KEY_CTRL_BACKSPACE, KEY_ALT_BACKSPACE, KEY_DEL,
    KEY_CTRL_C, KEY_CTRL_D, KEY_CTRL_V, KEY_CTRL_A, KEY_CTRL_E,
    KEY_CTRL_K, KEY_CTRL_U, KEY_TAB,
    KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT, KEY_HOME, KEY_END,
    clipboard_copy,
)
from .state import Mode, TUIState, _SubmitException, _CommandException
from .input import (
    get_overlay_matches, tab_complete, history_navigate,
    delete_word_before, open_in_vim,
)


class ModeHandler:
    """Dispatches keypresses based on the current TUI mode."""

    def handle_key(self, key: str, state: TUIState, screen) -> None:
        """Route *key* to the appropriate mode handler.

        *screen* is the Screen instance — handlers read/write its
        internal state (input_buf, cursor_pos, etc.) and call its
        refresh methods.
        """
        dispatch = {
            Mode.NORMAL:      self._handle_normal,
            Mode.INSERT:      self._handle_insert,
            Mode.VISUAL:      self._handle_visual,
            Mode.VISUAL_LINE: self._handle_visual,
            Mode.COMMAND:     self._handle_command,
            Mode.SEARCH:      self._handle_search,
        }
        handler = dispatch.get(state.mode, self._handle_normal)
        handler(key, state, screen)

    # ── Normal mode ───────────────────────────────────────────────────────────

    def _handle_normal(self, key: str, state: TUIState, screen) -> None:
        buf = screen._buffer
        layout = screen._layout

        # Multi-key: gg
        if state.pending_key == 'g':
            state.pending_key = ''
            if key == 'g':
                state.viewport_offset = 0
                state.cursor_row = 0
                state.auto_scroll = False
                screen._refresh_all()
                return
            # g + something else: ignore the pending g, process key normally

        if key == 'j' or key == KEY_DOWN:
            self._scroll_down(state, screen, 1)
            return

        if key == 'k' or key == KEY_UP:
            self._scroll_up(state, screen, 1)
            return

        if key == KEY_CTRL_D:
            half = max(1, layout.content_rows // 2)
            self._scroll_down(state, screen, half)
            return

        if key == KEY_CTRL_U:
            half = max(1, layout.content_rows // 2)
            self._scroll_up(state, screen, half)
            return

        if key == 'g':
            state.pending_key = 'g'
            return

        if key == 'G':
            total = buf.line_count()
            content_rows = layout.content_rows
            state.viewport_offset = max(0, total - content_rows)
            state.cursor_row = max(0, total - 1)
            state.auto_scroll = True
            screen._refresh_all()
            return

        if key == 'i':
            state.mode = Mode.INSERT
            state.auto_scroll = True
            # Pin to bottom when entering insert mode
            total = buf.line_count()
            content_rows = layout.content_rows
            state.viewport_offset = max(0, total - content_rows)
            screen._refresh_all()
            return

        if key == 'v':
            state.mode = Mode.VISUAL
            state.visual_anchor_row = state.cursor_row
            state.visual_anchor_col = state.cursor_col
            screen._refresh_all()
            return

        if key == 'V':
            state.mode = Mode.VISUAL_LINE
            state.visual_anchor_row = state.cursor_row
            state.visual_anchor_col = 0
            screen._refresh_all()
            return

        if key == ':':
            state.mode = Mode.COMMAND
            state.command_buf = []
            state.command_cursor = 0
            screen._refresh_all()
            return

        if key == '/':
            state.mode = Mode.SEARCH
            state.search_direction = 1
            state.search_buf = []
            screen._refresh_all()
            return

        if key == '?':
            state.mode = Mode.SEARCH
            state.search_direction = -1
            state.search_buf = []
            screen._refresh_all()
            return

        if key == 'n':
            self._search_next(state, screen, state.search_direction)
            return

        if key == 'N':
            self._search_next(state, screen, -state.search_direction)
            return

        if key == KEY_CTRL_C:
            raise KeyboardInterrupt

    # ── Insert mode ───────────────────────────────────────────────────────────

    def _handle_insert(self, key: str, state: TUIState, screen) -> None:
        if key == KEY_ESC:
            state.mode = Mode.NORMAL
            state.auto_scroll = False
            # Set cursor_row to current viewport bottom
            total = screen._buffer.line_count()
            state.cursor_row = min(
                state.viewport_offset + screen._layout.content_rows - 1,
                max(0, total - 1),
            )
            screen._refresh_all()
            return

        if key in KEY_ENTER:
            self._handle_submit(state, screen)
            return

        if key in KEY_ALT_ENTER:
            screen._input_buf.insert(screen._cursor_pos, '\n')
            screen._cursor_pos += 1
            screen._refresh_input()
            return

        if key == KEY_BACKSPACE:
            if screen._cursor_pos > 0:
                del screen._input_buf[screen._cursor_pos - 1]
                screen._cursor_pos -= 1
                screen._cmd_overlay_idx = -1
                screen._refresh_input()
            return

        if key in (KEY_CTRL_W, KEY_CTRL_BACKSPACE, KEY_ALT_BACKSPACE):
            screen._input_buf, screen._cursor_pos = delete_word_before(
                screen._input_buf, screen._cursor_pos
            )
            screen._cmd_overlay_idx = -1
            screen._refresh_input()
            return

        if key == KEY_DEL:
            if screen._cursor_pos < len(screen._input_buf):
                del screen._input_buf[screen._cursor_pos]
                screen._refresh_input()
            return

        if key == KEY_CTRL_C:
            if screen._input_buf:
                screen._input_buf.clear()
                screen._cursor_pos = 0
                screen._cmd_overlay_idx = -1
                screen._refresh_input()
                return
            raise KeyboardInterrupt

        if key == KEY_CTRL_V:
            new_buf = open_in_vim(screen._tty_fd, screen._cooked_attrs, screen._input_buf)
            screen._input_buf = new_buf
            screen._cursor_pos = len(new_buf)
            screen._refresh_all()
            return

        if key == KEY_UP:
            self._handle_insert_arrow_up(state, screen)
            return

        if key == KEY_DOWN:
            self._handle_insert_arrow_down(state, screen)
            return

        if key == KEY_RIGHT:
            if screen._cursor_pos < len(screen._input_buf):
                screen._cursor_pos += 1
                screen._refresh_input()
            return

        if key == KEY_LEFT:
            if screen._cursor_pos > 0:
                screen._cursor_pos -= 1
                screen._refresh_input()
            return

        if key in KEY_HOME:
            before = ''.join(screen._input_buf[:screen._cursor_pos])
            screen._cursor_pos = before.rfind('\n') + 1
            screen._refresh_input()
            return

        if key == KEY_CTRL_A:
            screen._cursor_pos = 0
            screen._refresh_input()
            return

        if key in KEY_END:
            rest = ''.join(screen._input_buf[screen._cursor_pos:])
            next_nl = rest.find('\n')
            screen._cursor_pos = (
                len(screen._input_buf) if next_nl == -1
                else screen._cursor_pos + next_nl
            )
            screen._refresh_input()
            return

        if key == KEY_CTRL_E:
            screen._cursor_pos = len(screen._input_buf)
            screen._refresh_input()
            return

        if key == KEY_CTRL_K:
            screen._input_buf = screen._input_buf[:screen._cursor_pos]
            screen._refresh_input()
            return

        if key == KEY_TAB:
            self._tab_complete(state, screen)
            return

        # Regular character
        if len(key) == 1 and ord(key) >= 32:
            screen._input_buf.insert(screen._cursor_pos, key)
            screen._cursor_pos += 1
            screen._cmd_overlay_idx = -1
            screen._refresh_input()

    def _handle_submit(self, state: TUIState, screen) -> None:
        """Handle Enter in insert mode."""
        # Line continuation with backslash
        if screen._cursor_pos > 0 and screen._input_buf[screen._cursor_pos - 1] == '\\':
            screen._input_buf[screen._cursor_pos - 1] = '\n'
            screen._refresh_input()
            return
        # If an overlay item is selected, complete to it
        matches = get_overlay_matches(''.join(screen._input_buf), screen._cmd_completions)
        if 0 <= screen._cmd_overlay_idx < len(matches):
            screen._input_buf = list(f'/{matches[screen._cmd_overlay_idx]}')
            screen._cursor_pos = len(screen._input_buf)
            screen._cmd_overlay_idx = -1
        result = ''.join(screen._input_buf)
        if result.strip():
            screen._history.insert(0, result)
            screen._save_history_entry(result)
        screen._in_prompt = False
        raise _SubmitException(result)

    def _handle_insert_arrow_up(self, state: TUIState, screen) -> None:
        matches = get_overlay_matches(''.join(screen._input_buf), screen._cmd_completions)
        if matches and screen._history_idx < 0:
            screen._cmd_overlay_idx = (screen._cmd_overlay_idx - 1) % len(matches)
            screen._refresh_input()
            return
        all_lines = ''.join(screen._input_buf).split('\n')
        before = ''.join(screen._input_buf[:screen._cursor_pos])
        lines_before = before.split('\n')
        cur_line = len(lines_before) - 1
        if cur_line == 0:
            idx, buf, pos = history_navigate(
                1, screen._history, screen._history_idx,
                screen._input_buf, screen._cursor_pos
            )
            screen._history_idx = idx
            screen._input_buf = buf
            screen._cursor_pos = pos
        else:
            cur_col = len(lines_before[-1])
            target_col = min(cur_col, len(all_lines[cur_line - 1]))
            screen._cursor_pos = (
                sum(len(all_lines[i]) + 1 for i in range(cur_line - 1)) + target_col
            )
        screen._refresh_input()

    def _handle_insert_arrow_down(self, state: TUIState, screen) -> None:
        matches = get_overlay_matches(''.join(screen._input_buf), screen._cmd_completions)
        if matches and screen._history_idx < 0:
            screen._cmd_overlay_idx = (screen._cmd_overlay_idx + 1) % len(matches)
            screen._refresh_input()
            return
        all_lines = ''.join(screen._input_buf).split('\n')
        before = ''.join(screen._input_buf[:screen._cursor_pos])
        lines_before = before.split('\n')
        cur_line = len(lines_before) - 1
        if cur_line == len(all_lines) - 1:
            idx, buf, pos = history_navigate(
                -1, screen._history, screen._history_idx,
                screen._input_buf, screen._cursor_pos
            )
            screen._history_idx = idx
            screen._input_buf = buf
            screen._cursor_pos = pos
        else:
            cur_col = len(lines_before[-1])
            target_col = min(cur_col, len(all_lines[cur_line + 1]))
            screen._cursor_pos = (
                sum(len(all_lines[i]) + 1 for i in range(cur_line + 1)) + target_col
            )
        screen._refresh_input()

    def _tab_complete(self, state: TUIState, screen) -> None:
        buf_str = ''.join(screen._input_buf)
        new_buf, new_idx = tab_complete(
            buf_str, screen._cmd_completions, screen._cmd_overlay_idx
        )
        if new_buf is not None:
            screen._input_buf = list(new_buf)
            screen._cursor_pos = len(screen._input_buf)
            screen._cmd_overlay_idx = new_idx
            screen._refresh_input()

    # ── Visual mode ───────────────────────────────────────────────────────────

    def _handle_visual(self, key: str, state: TUIState, screen) -> None:
        if key == KEY_ESC:
            state.mode = Mode.NORMAL
            screen._refresh_all()
            return

        if key == 'j' or key == KEY_DOWN:
            self._scroll_down(state, screen, 1)
            return

        if key == 'k' or key == KEY_UP:
            self._scroll_up(state, screen, 1)
            return

        if key == 'h' or key == KEY_LEFT:
            if state.cursor_col > 0:
                state.cursor_col -= 1
                screen._refresh_all()
            return

        if key == 'l' or key == KEY_RIGHT:
            state.cursor_col += 1
            screen._refresh_all()
            return

        if key == KEY_CTRL_D:
            half = max(1, screen._layout.content_rows // 2)
            self._scroll_down(state, screen, half)
            return

        if key == KEY_CTRL_U:
            half = max(1, screen._layout.content_rows // 2)
            self._scroll_up(state, screen, half)
            return

        if key == 'G':
            total = screen._buffer.line_count()
            state.cursor_row = max(0, total - 1)
            content_rows = screen._layout.content_rows
            state.viewport_offset = max(0, total - content_rows)
            screen._refresh_all()
            return

        if key == 'y':
            # Yank selection
            line_mode = (state.mode == Mode.VISUAL_LINE)
            text = screen._buffer.get_selection_text(
                state.visual_anchor_row, state.visual_anchor_col,
                state.cursor_row, state.cursor_col, line_mode,
            )
            state.yank_register = text
            clipboard_copy(text)
            state.mode = Mode.NORMAL
            screen._refresh_all()
            return

    # ── Command mode ──────────────────────────────────────────────────────────

    def _handle_command(self, key: str, state: TUIState, screen) -> None:
        if key == KEY_ESC:
            state.mode = Mode.NORMAL
            screen._refresh_all()
            return

        if key in KEY_ENTER:
            cmd = ''.join(state.command_buf).strip()
            state.mode = Mode.NORMAL
            if cmd:
                raise _CommandException(cmd)
            screen._refresh_all()
            return

        if key == KEY_BACKSPACE:
            if state.command_buf:
                del state.command_buf[state.command_cursor - 1]
                state.command_cursor -= 1
                screen._refresh_status()
            else:
                # Empty backspace cancels command mode
                state.mode = Mode.NORMAL
                screen._refresh_all()
            return

        if key == KEY_TAB:
            # Tab-complete command names
            cmd_str = ''.join(state.command_buf)
            from .input import _prefix_matches
            matches = _prefix_matches(cmd_str, screen._cmd_completions)
            if len(matches) == 1:
                state.command_buf = list(matches[0])
                state.command_cursor = len(state.command_buf)
                screen._refresh_status()
            elif len(matches) > 1:
                from .input import _common_prefix
                common = _common_prefix(matches)
                if len(common) > len(cmd_str):
                    state.command_buf = list(common)
                    state.command_cursor = len(state.command_buf)
                    screen._refresh_status()
            return

        # Regular character
        if len(key) == 1 and ord(key) >= 32:
            state.command_buf.insert(state.command_cursor, key)
            state.command_cursor += 1
            screen._refresh_status()

    # ── Search mode ───────────────────────────────────────────────────────────

    def _handle_search(self, key: str, state: TUIState, screen) -> None:
        if key == KEY_ESC:
            state.mode = Mode.NORMAL
            screen._refresh_all()
            return

        if key in KEY_ENTER:
            pattern = ''.join(state.search_buf)
            state.search_pattern = pattern
            state.search_matches = screen._buffer.search(pattern)
            state.search_match_idx = -1
            state.mode = Mode.NORMAL
            # Jump to first match
            if state.search_matches:
                self._search_next(state, screen, state.search_direction)
            else:
                screen._refresh_all()
            return

        if key == KEY_BACKSPACE:
            if state.search_buf:
                state.search_buf.pop()
                screen._refresh_all()
            else:
                state.mode = Mode.NORMAL
                screen._refresh_all()
            return

        if len(key) == 1 and ord(key) >= 32:
            state.search_buf.append(key)
            screen._refresh_all()

    # ── Scroll helpers ────────────────────────────────────────────────────────

    def _scroll_down(self, state: TUIState, screen, n: int) -> None:
        total = screen._buffer.line_count()
        content_rows = screen._layout.content_rows
        max_offset = max(0, total - content_rows)

        state.cursor_row = min(state.cursor_row + n, max(0, total - 1))
        # Ensure cursor stays visible
        if state.cursor_row >= state.viewport_offset + content_rows:
            state.viewport_offset = min(state.cursor_row - content_rows + 1, max_offset)
        state.auto_scroll = (state.viewport_offset >= max_offset)
        screen._refresh_all()

    def _scroll_up(self, state: TUIState, screen, n: int) -> None:
        state.cursor_row = max(state.cursor_row - n, 0)
        if state.cursor_row < state.viewport_offset:
            state.viewport_offset = state.cursor_row
        state.auto_scroll = False
        screen._refresh_all()

    def _search_next(self, state: TUIState, screen, direction: int) -> None:
        if not state.search_matches:
            return
        if direction > 0:
            # Find next match after cursor
            for i, m in enumerate(state.search_matches):
                if m > state.cursor_row:
                    state.search_match_idx = i
                    state.cursor_row = m
                    break
            else:
                # Wrap around
                state.search_match_idx = 0
                state.cursor_row = state.search_matches[0]
        else:
            # Find previous match before cursor
            for i in range(len(state.search_matches) - 1, -1, -1):
                if state.search_matches[i] < state.cursor_row:
                    state.search_match_idx = i
                    state.cursor_row = state.search_matches[i]
                    break
            else:
                state.search_match_idx = len(state.search_matches) - 1
                state.cursor_row = state.search_matches[-1]

        # Ensure cursor is visible
        content_rows = screen._layout.content_rows
        if state.cursor_row < state.viewport_offset:
            state.viewport_offset = state.cursor_row
        elif state.cursor_row >= state.viewport_offset + content_rows:
            state.viewport_offset = state.cursor_row - content_rows + 1
        screen._refresh_all()
