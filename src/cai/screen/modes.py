"""Mode state machine and per-mode key handlers for the TUI."""

import time

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
    history_navigate,
    delete_word_before, open_in_vim, open_buffer_in_vim,
)


# ── Word-motion & text-object helpers ────────────────────────────────────────

def _is_word_char(ch: str) -> bool:
    """True for characters that belong to a vim 'word' (alnum + underscore)."""
    return ch.isalnum() or ch == '_'


def _motion_w(plain: str, col: int) -> int:
    """Return column after moving forward one word (vim `w`)."""
    n = len(plain)
    if col >= n:
        return col
    # Skip current word characters
    if _is_word_char(plain[col]):
        while col < n and _is_word_char(plain[col]):
            col += 1
    elif not plain[col].isspace():
        # punctuation word
        while col < n and not plain[col].isspace() and not _is_word_char(plain[col]):
            col += 1
    # Skip whitespace
    while col < n and plain[col].isspace():
        col += 1
    return col


def _motion_b(plain: str, col: int) -> int:
    """Return column after moving backward one word (vim `b`)."""
    n = len(plain)
    if n == 0 or col <= 0:
        return 0
    col = min(col, n) - 1
    # Skip whitespace backward
    while col > 0 and plain[col].isspace():
        col -= 1
    if col < 0:
        return 0
    # Skip word or punctuation backward
    if _is_word_char(plain[col]):
        while col > 0 and _is_word_char(plain[col - 1]):
            col -= 1
    elif not plain[col].isspace():
        while col > 0 and not plain[col - 1].isspace() and not _is_word_char(plain[col - 1]):
            col -= 1
    return col


def _motion_e(plain: str, col: int) -> int:
    """Return column after moving to end of word (vim `e`)."""
    n = len(plain)
    if col >= n - 1:
        return max(0, n - 1)
    col += 1
    # Skip whitespace
    while col < n and plain[col].isspace():
        col += 1
    if col >= n:
        return max(0, n - 1)
    # Move to end of word or punctuation group
    if _is_word_char(plain[col]):
        while col + 1 < n and _is_word_char(plain[col + 1]):
            col += 1
    elif not plain[col].isspace():
        while col + 1 < n and not plain[col + 1].isspace() and not _is_word_char(plain[col + 1]):
            col += 1
    return col


def _textobj_inner_word(plain: str, col: int, big_word: bool = False) -> tuple[int, int] | None:
    """Return (start_col, end_col) for inner word under cursor."""
    if not plain or col >= len(plain):
        return None
    if big_word:
        # WORD: delimited by whitespace only
        if plain[col].isspace():
            return None
        sc = col
        while sc > 0 and not plain[sc - 1].isspace():
            sc -= 1
        ec = col
        while ec + 1 < len(plain) and not plain[ec + 1].isspace():
            ec += 1
        return sc, ec
    else:
        ch = plain[col]
        if _is_word_char(ch):
            sc = col
            while sc > 0 and _is_word_char(plain[sc - 1]):
                sc -= 1
            ec = col
            while ec + 1 < len(plain) and _is_word_char(plain[ec + 1]):
                ec += 1
            return sc, ec
        elif not ch.isspace():
            sc = col
            while sc > 0 and not plain[sc - 1].isspace() and not _is_word_char(plain[sc - 1]):
                sc -= 1
            ec = col
            while ec + 1 < len(plain) and not plain[ec + 1].isspace() and not _is_word_char(plain[ec + 1]):
                ec += 1
            return sc, ec
    return None


_BRACKET_PAIRS = {
    '(': ')', ')': '(',
    '<': '>', '>': '<',
}

_QUOTE_CHARS = {'"', "'", '`'}


def _textobj_inner_delimited(plain: str, col: int, char: str) -> tuple[int, int] | None:
    """Return (start_col, end_col) for inner content between matching delimiters.

    Handles quotes (" ' `) and brackets ( ) < >.
    """
    n = len(plain)
    if char in _QUOTE_CHARS:
        # Find the opening quote before/at col, and closing quote after
        # Strategy: find all positions of char, pair them left-to-right
        positions = [i for i, c in enumerate(plain) if c == char]
        if len(positions) < 2:
            return None
        # Find the pair that surrounds col
        for pi in range(0, len(positions) - 1, 2):
            start = positions[pi]
            end = positions[pi + 1]
            if start <= col <= end:
                if end - start <= 1:
                    return None  # empty
                return start + 1, end - 1
        return None

    # Bracket pairs
    open_ch = char if char in ('(', '<') else _BRACKET_PAIRS.get(char)
    close_ch = _BRACKET_PAIRS.get(open_ch, char) if open_ch else char
    if open_ch is None:
        return None

    # Search backward for opening bracket
    depth = 0
    start = -1
    for i in range(col, -1, -1):
        if plain[i] == close_ch and i != col:
            depth += 1
        elif plain[i] == open_ch:
            if depth == 0:
                start = i
                break
            depth -= 1
    if start == -1:
        return None

    # Search forward for closing bracket
    depth = 0
    end = -1
    for i in range(start + 1, n):
        if plain[i] == open_ch:
            depth += 1
        elif plain[i] == close_ch:
            if depth == 0:
                end = i
                break
            depth -= 1
    if end == -1:
        return None
    if end - start <= 1:
        return None
    return start + 1, end - 1


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

        # Multi-key: gg, zz/zt/zb
        if state.pending_key == 'z':
            state.pending_key = ''
            total = buf.line_count()
            content_rows = layout.content_rows
            if key == 'z':
                # zz — center cursor line in viewport
                state.viewport_offset = max(0, min(
                    state.cursor_row - content_rows // 2,
                    max(0, total - content_rows),
                ))
            elif key == 't':
                # zt — cursor line at top of viewport
                state.viewport_offset = max(0, min(
                    state.cursor_row,
                    max(0, total - content_rows),
                ))
            elif key == 'b':
                # zb — cursor line at bottom of viewport
                state.viewport_offset = max(0, state.cursor_row - content_rows + 1)
            else:
                return  # z + unknown: ignore
            state.auto_scroll = (state.viewport_offset >= max(0, total - content_rows))
            screen._refresh_all()
            return

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

        if key == 'h' or key == KEY_LEFT:
            if state.cursor_col > 0:
                state.cursor_col -= 1
                screen._refresh_all()
            return

        if key == 'l' or key == KEY_RIGHT:
            plain = buf.get_plain_text(state.cursor_row)
            if state.cursor_col < max(0, len(plain) - 1):
                state.cursor_col += 1
                screen._refresh_all()
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

        if key == 'z':
            state.pending_key = 'z'
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

        if key == 'w':
            self._word_motion(state, screen, 'w')
            return

        if key == 'b':
            self._word_motion(state, screen, 'b')
            return

        if key == 'e':
            self._word_motion(state, screen, 'e')
            return

        if key == '0':
            state.cursor_col = 0
            screen._refresh_all()
            return

        if key == '$':
            plain = buf.get_plain_text(state.cursor_row)
            state.cursor_col = max(0, len(plain) - 1)
            screen._refresh_all()
            return

        if key == '^':
            plain = buf.get_plain_text(state.cursor_row)
            col = 0
            while col < len(plain) and plain[col].isspace():
                col += 1
            state.cursor_col = col
            screen._refresh_all()
            return

        if key == 'n':
            self._search_next(state, screen, state.search_direction)
            return

        if key == 'N':
            self._search_next(state, screen, -state.search_direction)
            return

        if key == KEY_CTRL_V:
            self._open_buffer_in_vim(state, screen)
            return

        if key == KEY_CTRL_C:
            now = time.monotonic()
            if now - state.last_ctrl_c < 0.5:
                raise KeyboardInterrupt
            state.last_ctrl_c = now
            screen.write_status_hint("Press Ctrl-C again to quit")
            return

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
                screen._refresh_input()
            return

        if key in (KEY_CTRL_W, KEY_CTRL_BACKSPACE, KEY_ALT_BACKSPACE):
            screen._input_buf, screen._cursor_pos = delete_word_before(
                screen._input_buf, screen._cursor_pos
            )
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
                state.last_ctrl_c = 0.0
                screen._refresh_input()
                return
            now = time.monotonic()
            if now - state.last_ctrl_c < 0.5:
                raise KeyboardInterrupt
            state.last_ctrl_c = now
            screen.write_status_hint("Press Ctrl-C again to quit")
            return

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

        # ':' as first character → enter command mode
        if key == ':' and not screen._input_buf:
            state.mode = Mode.COMMAND
            state.command_buf = []
            state.command_cursor = 0
            screen._refresh_all()
            return

        # Regular character
        if len(key) == 1 and ord(key) >= 32:
            screen._input_buf.insert(screen._cursor_pos, key)
            screen._cursor_pos += 1
            screen._refresh_input()

    def _handle_submit(self, state: TUIState, screen) -> None:
        """Handle Enter in insert mode."""
        # Line continuation with backslash
        if screen._cursor_pos > 0 and screen._input_buf[screen._cursor_pos - 1] == '\\':
            screen._input_buf[screen._cursor_pos - 1] = '\n'
            screen._refresh_input()
            return
        result = ''.join(screen._input_buf)
        if result.strip():
            screen._history.insert(0, result)
            screen._save_history_entry(result)
        screen._in_prompt = False
        raise _SubmitException(result)

    def _handle_insert_arrow_up(self, state: TUIState, screen) -> None:
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

    # ── Visual mode ───────────────────────────────────────────────────────────

    def _handle_visual(self, key: str, state: TUIState, screen) -> None:
        # Handle pending 'i' for text objects
        if state.pending_key == 'i':
            state.pending_key = ''
            self._apply_text_object(key, state, screen)
            return

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

        if key == 'w':
            self._word_motion(state, screen, 'w')
            return

        if key == 'b':
            self._word_motion(state, screen, 'b')
            return

        if key == 'e':
            self._word_motion(state, screen, 'e')
            return

        if key == '0':
            state.cursor_col = 0
            screen._refresh_all()
            return

        if key == '$':
            plain = screen._buffer.get_plain_text(state.cursor_row)
            state.cursor_col = max(0, len(plain) - 1)
            screen._refresh_all()
            return

        if key == '^':
            plain = screen._buffer.get_plain_text(state.cursor_row)
            col = 0
            while col < len(plain) and plain[col].isspace():
                col += 1
            state.cursor_col = col
            screen._refresh_all()
            return

        if key == 'i':
            state.pending_key = 'i'
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
            self._command_tab_complete(state, screen)
            return

        # Regular character
        if len(key) == 1 and ord(key) >= 32:
            state.command_buf.insert(state.command_cursor, key)
            state.command_cursor += 1
            screen._refresh_status()

    def _command_tab_complete(self, state: TUIState, screen) -> None:
        """Tab-complete command names and sub-options in command mode."""
        completions = screen._cmd_completions
        if not completions:
            return
        text = ''.join(state.command_buf)

        # Check if we're completing a sub-option (e.g. "skill fr")
        parts = text.split(' ', 1)
        cmd_word = parts[0]

        if len(parts) == 2:
            # Sub-option completion: cmd_word is already determined
            if cmd_word not in completions:
                return
            sub_options = completions[cmd_word]
            if not sub_options:
                return
            sub_prefix = parts[1]
            matches = [s for s in sub_options if s.startswith(sub_prefix)]
            if not matches:
                return
            if len(matches) == 1:
                result = f'{cmd_word} {matches[0]}'
            else:
                common = matches[0]
                for m in matches[1:]:
                    i = 0
                    while i < len(common) and i < len(m) and common[i] == m[i]:
                        i += 1
                    common = common[:i]
                if len(common) > len(sub_prefix):
                    result = f'{cmd_word} {common}'
                else:
                    return
        else:
            # Top-level command completion
            matches = [c for c in completions if c.startswith(cmd_word)]
            if not matches:
                return
            if len(matches) == 1:
                result = matches[0]
                # Add trailing space if the command has sub-options
                if completions.get(matches[0]):
                    result += ' '
            else:
                common = matches[0]
                for m in matches[1:]:
                    i = 0
                    while i < len(common) and i < len(m) and common[i] == m[i]:
                        i += 1
                    common = common[:i]
                if len(common) > len(cmd_word):
                    result = common
                else:
                    return

        state.command_buf = list(result)
        state.command_cursor = len(state.command_buf)
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
        # Clamp cursor_col to new line length
        plain = screen._buffer.get_plain_text(state.cursor_row)
        if plain:
            state.cursor_col = min(state.cursor_col, len(plain) - 1)
        else:
            state.cursor_col = 0
        # Ensure cursor stays visible
        if state.cursor_row >= state.viewport_offset + content_rows:
            state.viewport_offset = min(state.cursor_row - content_rows + 1, max_offset)
        state.auto_scroll = (state.viewport_offset >= max_offset)
        screen._refresh_all()

    def _scroll_up(self, state: TUIState, screen, n: int) -> None:
        state.cursor_row = max(state.cursor_row - n, 0)
        # Clamp cursor_col to new line length
        plain = screen._buffer.get_plain_text(state.cursor_row)
        if plain:
            state.cursor_col = min(state.cursor_col, len(plain) - 1)
        else:
            state.cursor_col = 0
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

    # ── Word motions ─────────────────────────────────────────────────────────

    def _word_motion(self, state: TUIState, screen, motion: str) -> None:
        """Execute w/b/e motion, crossing line boundaries."""
        buf = screen._buffer
        total = buf.line_count()
        if total == 0:
            return
        plain = buf.get_plain_text(state.cursor_row)
        col = min(state.cursor_col, max(0, len(plain) - 1)) if plain else 0

        if motion == 'w':
            new_col = _motion_w(plain, col)
            if new_col >= len(plain) and state.cursor_row + 1 < total:
                state.cursor_row += 1
                plain = buf.get_plain_text(state.cursor_row)
                new_col = 0
                while new_col < len(plain) and plain[new_col].isspace():
                    new_col += 1
            state.cursor_col = min(new_col, max(0, len(plain) - 1)) if plain else 0
        elif motion == 'b':
            new_col = _motion_b(plain, col)
            if new_col == 0 and col == 0 and state.cursor_row > 0:
                state.cursor_row -= 1
                plain = buf.get_plain_text(state.cursor_row)
                new_col = max(0, len(plain) - 1) if plain else 0
            state.cursor_col = new_col
        elif motion == 'e':
            new_col = _motion_e(plain, col)
            if new_col <= col and state.cursor_row + 1 < total:
                state.cursor_row += 1
                plain = buf.get_plain_text(state.cursor_row)
                if plain:
                    new_col = 0
                    while new_col < len(plain) and plain[new_col].isspace():
                        new_col += 1
                    new_col = _motion_e(plain, new_col)
                else:
                    new_col = 0
            state.cursor_col = new_col

        # Ensure cursor stays visible
        content_rows = screen._layout.content_rows
        if state.cursor_row < state.viewport_offset:
            state.viewport_offset = state.cursor_row
        elif state.cursor_row >= state.viewport_offset + content_rows:
            state.viewport_offset = state.cursor_row - content_rows + 1
        screen._refresh_all()

    # ── Text objects (visual mode) ───────────────────────────────────────────

    def _apply_text_object(self, key: str, state: TUIState, screen) -> None:
        """Handle i + <key> text object in visual mode."""
        buf = screen._buffer
        plain = buf.get_plain_text(state.cursor_row)
        col = state.cursor_col
        result = None

        if key == 'w':
            result = _textobj_inner_word(plain, col, big_word=False)
        elif key == 'W':
            result = _textobj_inner_word(plain, col, big_word=True)
        elif key in ('"', "'", '`'):
            result = _textobj_inner_delimited(plain, col, key)
        elif key in ('(', ')', '<', '>'):
            result = _textobj_inner_delimited(plain, col, key)

        if result is not None:
            sc, ec = result
            # Set selection to cover the text object on the current line
            state.visual_anchor_row = state.cursor_row
            state.visual_anchor_col = sc
            state.cursor_col = ec
            if state.mode == Mode.VISUAL_LINE:
                state.mode = Mode.VISUAL  # text objects switch to char-wise
        screen._refresh_all()

    # ── Vim buffer viewer ────────────────────────────────────────────────────

    def _open_buffer_in_vim(self, state: TUIState, screen) -> None:
        """Open the content buffer in nvim with cursor at current position."""
        from .ansi import ALT_ENTER, ERASE_SCREEN, CUR_HIDE
        import sys
        open_buffer_in_vim(
            screen._tty_fd, screen._cooked_attrs,
            screen._buffer._lines,
            state.cursor_row, state.cursor_col,
        )
        # Re-enter alternate screen and restore TUI
        import tty
        tty.setraw(screen._tty_fd)
        sys.stdout.write(ALT_ENTER + ERASE_SCREEN + CUR_HIDE)
        sys.stdout.flush()
        screen._refresh_all()
