"""ANSI escape code constants and text utilities."""

import re
import sys

_ANSI_RE = re.compile(
    r'\033'
    r'(?:'
    r'\[[0-9;?]*[mABCDEFGHJKLMPRSTXZ@`abcdefhilnpqrstux~]'   # CSI
    r'|[@-Z\\-_]'                                               # Fe / Fs
    r')'
)

# SGR (Select Graphic Rendition)
SGR_RESET          = '\033[m'
SGR_BOLD           = '\033[1m'
SGR_DIM            = '\033[2m'
SGR_REVERSE        = '\033[7m'
SGR_DIM_GRAY       = '\033[2;37m'
SGR_BOLD_RED       = '\033[1;31m'
SGR_CYAN           = '\033[36m'
SGR_GREEN          = '\033[32m'
SGR_RED            = '\033[31m'
SGR_YELLOW         = '\033[33m'
SGR_MAGENTA        = '\033[35m'
SGR_BOLD_AZURE     = '\033[1;38;5;45m'
SGR_AZURE_ON_DGRAY = '\033[38;5;45;48;5;238m'
SGR_REVERSE_YELLOW = '\033[7;33m'

# cursor control
CUR_SHOW      = '\033[?25h'
CUR_HIDE      = '\033[?25l'
CUR_HOME      = '\033[H'

CURSOR_BAR   = '\033[5 q'    # blinking bar (insert mode)
CURSOR_BLOCK = '\033[2 q'    # steady block (normal mode)
CURSOR_RESET = '\033[0 q'    # reset to default


def cur_move(r, c):
    return f'\033[{r};{c}H'


ERASE_SCREEN = '\033[2J'
ERASE_LINE   = '\033[K'

ALT_ENTER = '\033[?1049h'
ALT_EXIT  = '\033[?1049l'

# mouse reporting: 1000 = press/release, 1002 = motion while a button is held
# (drag), 1006 = SGR extended coordinates so x/y aren't capped at 223. these
# modes are independent of the alternate screen, so enabling once covers
# overlays too - only an external full-screen program (nvim) resets them.
MOUSE_ON  = '\033[?1000h\033[?1002h\033[?1006h'
MOUSE_OFF = '\033[?1000l\033[?1002l\033[?1006l'

KEY_ENTER          = ('\r', '\n')
KEY_ALT_ENTER      = ('\033\r', '\033\n')
KEY_BACKSPACE      = '\x7f'
KEY_CTRL_W         = '\x17'
KEY_CTRL_BACKSPACE = '\x08'
KEY_ALT_BACKSPACE  = '\033\x7f'
KEY_DEL            = '\033[3~'
KEY_CTRL_C         = '\x03'
KEY_CTRL_D         = '\x04'
KEY_CTRL_V         = '\x16'
KEY_CTRL_A         = '\x01'
KEY_CTRL_E         = '\x05'
KEY_CTRL_K         = '\x0b'
KEY_CTRL_L         = '\x0c'
KEY_CTRL_P         = '\x10'
KEY_CTRL_R         = '\x12'
KEY_CTRL_S         = '\x13'
KEY_CTRL_T         = '\x14'
KEY_CTRL_U         = '\x15'
KEY_CTRL_X         = '\x18'
KEY_ESC            = '\033'
KEY_UP             = '\033[A'
KEY_DOWN           = '\033[B'
KEY_RIGHT          = '\033[C'
KEY_LEFT           = '\033[D'
KEY_HOME           = ('\033[H', '\033[1~', '\033OH')
KEY_END            = ('\033[F', '\033[4~', '\033OF')
KEY_PGUP           = '\033[5~'
KEY_PGDN           = '\033[6~'
KEY_TAB            = '\t'


def ansi_strip(text):
    """remove ANSI escape sequences (for visual-width calculations)."""
    return _ANSI_RE.sub('', text)


def ansi_pad(text, width):
    """right-pad text to width visual columns, ignoring ANSI sequences.
    if the visible length already exceeds width, truncates via ansi_strip
    (losing color) so the result still fits. this keeps frame borders
    aligned when status lines contain colored tokens."""
    visible = len(ansi_strip(text))
    if visible < width:
        return text + ' ' * (width - visible)
    if visible == width:
        return text
    # overflow - strip ANSI to make truncation safe.
    return ansi_strip(text)[:width]


def osc52_copy(text):
    """return OSC 52 escape sequence to copy text to the system clipboard."""
    import base64
    encoded = base64.b64encode(text.encode('utf-8')).decode('ascii')
    return f'\033]52;c;{encoded}\033\\'


def clipboard_copy(text):
    """copy text to the system clipboard via OSC 52 + platform fallback."""
    sys.stdout.write(osc52_copy(text))
    sys.stdout.flush()

    # subprocess fallback for terminals that don't support OSC 52
    import platform
    import shutil
    import subprocess

    cmd = None
    if platform.system() == 'Darwin':
        cmd = 'pbcopy'
    elif shutil.which('xclip'):
        cmd = 'xclip'
    elif shutil.which('xsel'):
        cmd = 'xsel'
    if cmd is None: return

    argv = [cmd]
    if cmd == 'xclip':
        argv += ['-selection', 'clipboard']
    try:
        subprocess.run(argv,
                       input=text.encode('utf-8'),
                       check=True,
                       timeout=2,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except Exception:
        pass


def wrap_ansi(text, width):
    """wrap text (which may contain ANSI codes) to width visual columns.

    ANSI sequences are preserved verbatim; they contribute zero visual
    width. when a line is broken mid-style the active SGR is closed with
    \\033[m at the break point and re-opened at the start of the
    continuation line. returns a list of display lines (no trailing \\n)."""
    if width <= 0:
        if not text:
            return ['']
        return text.split('\n')

    lines = []
    active_style = ''

    for logical in text.split('\n'):
        current = []
        if active_style:
            current.append(active_style)
        col = 0
        i = 0
        n = len(logical)

        while i < n:
            ch = logical[i]

            if ch == '\033' and i + 1 < n:
                j = i + 1
                if logical[j] == '[':
                    # CSI sequence: ESC [ ... <final-byte>
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
                        j += 1
                    seq = logical[i:j]
                    # track the most recent SGR for style continuations
                    # across wraps
                    if seq.endswith('m'):
                        if seq in (SGR_RESET, '\033[0m'):
                            active_style = ''
                        else:
                            active_style = seq
                    current.append(seq)
                    i = j
                else:
                    current.append(logical[i:i + 2])
                    i += 2
            else:
                if col >= width:
                    # hard-wrap: close style, emit line, start continuation
                    if active_style:
                        current.append(SGR_RESET)
                    lines.append(''.join(current))
                    current = []
                    if active_style:
                        current.append(active_style)
                    col = 0
                current.append(ch)
                col += 1
                i += 1

        if active_style and current:
            current.append(SGR_RESET)
        lines.append(''.join(current))

    return lines
