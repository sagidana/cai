"""raw key reading and stateless input helpers."""

import os
import select
import shlex
import subprocess
import sys
import tempfile
import termios
import tty

from .ansi import CUR_SHOW, ALT_EXIT, ansi_strip, PASTE_START, PASTE_END

# tag identifying a bracketed-paste event returned by read_key: (PASTE, text).
PASTE = 'paste'


def _read_byte(tty_fd):
    """read one more byte if it's already waiting, else ''. used to finish an
    escape sequence without blocking on a garbled/truncated one."""
    ready, _, _ = select.select([tty_fd], [], [], 0.05)
    if not ready:
        return ''
    return os.read(tty_fd, 1).decode('utf-8', errors='replace')


def _read_csi(tty_fd):
    """read a CSI sequence body (everything after the '\\033[') up to and
    including its final byte (0x40-0x7e). reading to the terminator - rather
    than a fixed-size slurp - keeps a variable-length mouse report whole and
    leaves a following batched event in the buffer for the next read."""
    seq = ''
    while True:
        b = _read_byte(tty_fd)
        if b == '':
            break
        seq += b
        if '\x40' <= b <= '\x7e':
            break
    return seq


def _read_paste(tty_fd):
    """slurp a bracketed-paste body (after PASTE_START) up to and excluding
    PASTE_END. reads raw bytes in chunks and decodes once, so multibyte UTF-8 in
    the paste is preserved. any input following PASTE_END in the same burst is
    dropped (a keystroke landing in the same read as a paste end is rare)."""
    data = b''
    end = PASTE_END.encode('utf-8')
    while True:
        ready, _, _ = select.select([tty_fd], [], [], 0.1)
        if not ready:
            break
        chunk = os.read(tty_fd, 4096)
        if chunk == b'':
            break
        data += chunk
        if end in data:
            data = data.split(end, 1)[0]
            break
    return data.decode('utf-8', errors='replace')


def read_key(tty_fd):
    """read one logical keypress from the given fd (handles escape sequences).

    a bracketed paste is returned as the tuple (PASTE, text) so the caller can
    insert the whole block literally instead of treating a pasted newline as
    Enter."""
    ch = os.read(tty_fd, 1).decode('utf-8', errors='replace')
    if ch != '\033':
        return ch
    nxt = _read_byte(tty_fd)
    if nxt == '':
        return ch
    if nxt == '[':
        seq = '\033[' + _read_csi(tty_fd)
        if seq == PASTE_START:
            return (PASTE, _read_paste(tty_fd))
        return seq
    if nxt == 'O':
        return '\033O' + _read_byte(tty_fd)
    return ch + nxt


def parse_mouse(seq):
    """decode an SGR mouse report (\\033[<b;x;y M|m) into
    (action, button, col, row), or None if seq isn't one. action is one of
    'wheel_up', 'wheel_down', 'press', 'drag', 'release'. col/row are 1-indexed
    terminal coordinates."""
    if not isinstance(seq, str):
        return None
    if not seq.startswith('\033[<'):
        return None
    if not (seq.endswith('M') or seq.endswith('m')):
        return None
    final = seq[-1]
    parts = seq[3:-1].split(';')
    if len(parts) != 3:
        return None
    try:
        b = int(parts[0])
        col = int(parts[1])
        row = int(parts[2])
    except ValueError:
        return None

    if b & 64:
        action = 'wheel_up'
        if b & 1:
            action = 'wheel_down'
        return action, b & 3, col, row

    button = b & 3
    if final == 'm':
        return 'release', button, col, row
    if b & 32:
        return 'drag', button, col, row
    return 'press', button, col, row


def input_pos_from_click(input_buf, vrow, tcol, prompt_prefix, cont_prefix, cols):
    """map a click in the input area to an index into input_buf. vrow is the
    0-based visual row within the input area, tcol the 0-based terminal column.
    mirrors layout.render_input's wrapping; clamps out-of-range clicks to the
    nearest valid position."""
    cols = max(1, cols)
    buf_str = ''.join(input_buf)
    lines = buf_str.split('\n')

    base = 0
    row_in_line = vrow
    target = -1
    for i, line in enumerate(lines):
        prefix = prompt_prefix
        if i > 0:
            prefix = cont_prefix
        vrows = max(1, (len(prefix) + len(line) + cols - 1) // cols)
        if row_in_line < vrows:
            target = i
            break
        row_in_line -= vrows
        base += len(line) + 1

    if target == -1:
        return len(input_buf)

    prefix = prompt_prefix
    if target > 0:
        prefix = cont_prefix
    char_in_line = row_in_line * cols + tcol - len(prefix)
    if char_in_line < 0:
        char_in_line = 0
    if char_in_line > len(lines[target]):
        char_in_line = len(lines[target])
    return base + char_in_line


def delete_word_before(buf, pos):
    """Ctrl-W / Ctrl-Backspace: delete word before cursor. returns (buf, pos)."""
    buf = list(buf)
    while pos > 0 and buf[pos - 1] in ' \t':
        del buf[pos - 1]
        pos -= 1
    while pos > 0 and buf[pos - 1] not in ' \t\n':
        del buf[pos - 1]
        pos -= 1
    return buf, pos


def history_navigate(direction, history, history_idx, input_buf, cursor_pos):
    """navigate command history.
    returns (new_history_idx, new_buf, new_cursor_pos)."""
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


def editor_argv(path, *, readonly=False, row=None, col=None):
    """build the command line for the user's editor: $EDITOR if set, nvim
    otherwise. read-only mode and cursor positioning are vim flags, so they
    are only added when the editor is vim-family; anything else just gets
    the path."""
    editor = os.environ.get('EDITOR', '').strip()
    if not editor:
        editor = 'nvim'
    argv = shlex.split(editor)
    name = os.path.basename(argv[0])
    if name in ('vi', 'vim', 'nvim', 'gvim'):
        if readonly:
            argv.append('-R')
        if row is not None:
            # +line positions the cursor; 'normal! N|' moves to column N.
            argv.append(f'+{row}')
            argv.append(f'+normal! {col}|')
    argv.append(path)
    return argv


def open_in_editor(tty_fd, cooked_attrs, buf):
    """open buf content in the editor; return updated character list."""
    content = ''.join(buf)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        termios.tcsetattr(tty_fd, termios.TCSADRAIN, cooked_attrs)
        sys.stdout.write(CUR_SHOW)
        sys.stdout.flush()
        subprocess.run(editor_argv(tmp))
        tty.setraw(tty_fd)
        with open(tmp, 'r') as f:
            new_content = f.read().rstrip('\n')
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return list(new_content)


def open_buffer_in_editor(tty_fd, cooked_attrs, lines, cursor_row, cursor_col):
    """open content buffer in the editor with cursor at the given position."""
    content = '\n'.join(ansi_strip(line) for line in lines)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        termios.tcsetattr(tty_fd, termios.TCSADRAIN, cooked_attrs)
        sys.stdout.write(ALT_EXIT + CUR_SHOW)
        sys.stdout.flush()
        # cursor_row is 0-based, vim is 1-based
        subprocess.run(editor_argv(tmp, row=cursor_row + 1, col=cursor_col + 1))
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
