"""Raw key reading and stateless input helpers."""

import os
import select
import subprocess
import tempfile
import termios
import tty

from .ansi import KEY_CTRL_C, CUR_SHOW


def read_key(tty_fd: int) -> str:
    """Read one logical keypress from the given fd (handles escape sequences)."""
    ch = os.read(tty_fd, 1).decode('utf-8', errors='replace')
    if ch != '\033':
        return ch
    ready, _, _ = select.select([tty_fd], [], [], 0.05)
    if ready:
        rest = os.read(tty_fd, 16).decode('utf-8', errors='replace')
        return ch + rest
    return ch


def handle_listener_key(key: str, interrupt_event) -> None:
    """Signal interrupt on Ctrl-C during streaming."""
    if key == KEY_CTRL_C:
        interrupt_event.set()


def delete_word_before(buf: list, pos: int) -> tuple:
    """Ctrl-W / Ctrl-Backspace: delete word before cursor. Returns (buf, pos)."""
    buf = list(buf)
    while pos > 0 and buf[pos - 1] in ' \t':
        del buf[pos - 1]
        pos -= 1
    while pos > 0 and buf[pos - 1] not in ' \t\n':
        del buf[pos - 1]
        pos -= 1
    return buf, pos


def history_navigate(
    direction: int,
    history: list,
    history_idx: int,
    input_buf: list,
    cursor_pos: int,
) -> tuple:
    """Navigate command history. Returns (new_history_idx, new_buf, new_cursor_pos)."""
    new_idx = history_idx + direction

    if direction > 0 and new_idx < len(history):
        new_buf = list(history[new_idx])
        return new_idx, new_buf, len(new_buf)

    if direction < 0 and history_idx > 0:
        new_buf = list(history[history_idx - 1])
        return history_idx - 1, new_buf, len(new_buf)

    if direction < 0 and history_idx == 0:
        return -1, [], 0

    return history_idx, input_buf, cursor_pos


def _prefix_matches(prefix: str, completions: list) -> list:
    """Space-aware prefix matching for /commands.

    Without a space in the prefix (e.g. "skill") only top-level commands are
    shown — completions that contain a space are hidden so sub-commands don't
    flood the overlay until the user has committed to a command word.

    With a space in the prefix (e.g. "skill ") the space acts as a delimiter
    that opts into sub-command matching, so only completions that start with
    the full prefix (including the space) are returned.
    """
    if ' ' in prefix:
        return [c for c in completions if c.startswith(prefix)]
    return [c for c in completions if c.startswith(prefix) and ' ' not in c]


def get_overlay_matches(buf_str: str, completions: list) -> list:
    """Return command names whose prefix matches the current /cmd input."""
    if not buf_str.startswith('/') or '\n' in buf_str:
        return []
    prefix = buf_str[1:]
    return _prefix_matches(prefix, completions)


def tab_complete(
    buf_str: str,
    completions: list,
    overlay_idx: int,
) -> tuple:
    """Tab-complete /command. Returns (new_buf_str_or_None, new_overlay_idx)."""
    if not buf_str.startswith('/'):
        return None, overlay_idx
    current = buf_str[1:]
    matches = _prefix_matches(current, completions)

    if 0 <= overlay_idx < len(matches):
        return f'/{matches[overlay_idx]}', -1

    if len(matches) == 1:
        return f'/{matches[0]}', -1

    if len(matches) > 1:
        common = _common_prefix(matches)
        if len(common) > len(current):
            return f'/{common}', -1

    return None, overlay_idx


def _common_prefix(words: list) -> str:
    prefix = words[0]
    for w in words[1:]:
        i = 0
        while i < len(prefix) and i < len(w) and prefix[i] == w[i]:
            i += 1
        prefix = prefix[:i]
    return prefix


def open_in_vim(tty_fd: int, cooked_attrs, buf: list) -> list:
    """Open buf content in nvim; return updated character list."""
    content = ''.join(buf)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        termios.tcsetattr(tty_fd, termios.TCSADRAIN, cooked_attrs)
        import sys
        sys.stdout.write(CUR_SHOW)
        sys.stdout.flush()
        subprocess.run(['nvim', tmp])
        tty.setraw(tty_fd)
        with open(tmp, 'r') as f:
            new_content = f.read().rstrip('\n')
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return list(new_content)


def open_buffer_in_vim(tty_fd: int, cooked_attrs, lines: list[str], cursor_row: int, cursor_col: int) -> None:
    """Open content buffer in vim (read-only) with cursor at the given position."""
    from .ansi import ansi_strip, ALT_EXIT, ERASE_SCREEN
    import sys
    content = '\n'.join(ansi_strip(line) for line in lines)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        termios.tcsetattr(tty_fd, termios.TCSADRAIN, cooked_attrs)
        sys.stdout.write(ALT_EXIT + CUR_SHOW)
        sys.stdout.flush()
        # +line positions cursor; cursor_row is 0-based, vim is 1-based
        vim_row = cursor_row + 1
        vim_col = cursor_col + 1
        subprocess.run(['nvim', f'+{vim_row}', f'+normal! {vim_col}|', tmp])
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
