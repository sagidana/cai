"""content buffer for the alternate-screen TUI: stores all rendered display
lines in memory, supports scrolling, search, selection, and re-wrapping on
terminal resize.

each appended segment can carry a gutter - a styled per-line prefix (the
colored message bar) re-applied to every wrapped display line, so the frame
survives wrapping and resize. the gutter glyph is fixed (GUTTER_GLYPH) so
blank detection and selection extraction can recognize gutter-only rows."""

import re
from .ansi import wrap_ansi, ansi_strip

# the visible gutter glyph. screen.py composes the styled prefix around it;
# the buffer recognizes it when deciding what a "blank" row is and when
# stripping the frame out of yanked text.
GUTTER_GLYPH = '▌'


class ContentBuffer:
    """stores terminal-rendered lines for the scrollable conversation area."""

    def __init__(self, cols):
        self._cols = max(1, cols)
        self._lines = []                 # wrapped display lines (with ANSI)
        self._raw_segments = []          # raw text segments for re-wrap
        self._gutters = []               # per-segment gutter prefix ('' = none)
        self._seg_lines = []             # per-segment wrapped line count
        self._partial = False            # last segment ended mid-line

    def append_text(self, text, gutter=''):
        """append text to the buffer. returns number of new display lines.
        gutter is a styled prefix painted at the start of every wrapped line
        of this segment (the message frame); it must render GUTTER_GLYPH."""
        if not text: return 0

        old_count = len(self._lines)

        # a gutter change mid-line first terminates the open line - two
        # gutters never share a display row.
        if self._partial and self._gutters and self._gutters[-1] != gutter:
            self.end_line()

        if self._partial and self._raw_segments:
            self._raw_segments[-1] += text
            new_wrapped = self._wrap_segment(self._raw_segments[-1], self._gutters[-1])
            prev_seg_lines = self._seg_lines[-1]
            self._lines = self._lines[:len(self._lines) - prev_seg_lines] + new_wrapped
            self._seg_lines[-1] = len(new_wrapped)
        else:
            self._raw_segments.append(text)
            self._gutters.append(gutter)
            new_wrapped = self._wrap_segment(text, gutter)
            self._lines.extend(new_wrapped)
            self._seg_lines.append(len(new_wrapped))

        self._partial = not text.endswith('\n')

        return len(self._lines) - old_count

    def end_line(self):
        """terminate a partial trailing line. the '\\n' joins the open
        segment, so the line keeps its gutter."""
        if not self._partial: return
        self.append_text('\n', gutter=self._gutters[-1])

    def trim_trailing_blanks(self):
        """drop trailing blank display lines - plain or gutter-only - so a
        new block's separator is exact no matter how many newlines the
        previous block ended with. raw segments are trimmed to match, so a
        resize rewrap reproduces the trimmed state."""
        if self._partial: return
        while self._raw_segments:
            last = self._raw_segments[-1]
            stripped = last.rstrip('\n')
            if ansi_strip(stripped) == '':
                # the segment renders nothing but newlines: drop it whole.
                self._drop_last_segment()
                continue
            trimmed = stripped + '\n'
            if trimmed == last: return
            self._replace_last_segment(trimmed)
            return

    def _drop_last_segment(self):
        count = self._seg_lines[-1]
        if count:
            self._lines = self._lines[:len(self._lines) - count]
        self._raw_segments.pop()
        self._gutters.pop()
        self._seg_lines.pop()

    def _replace_last_segment(self, text):
        prev_seg_lines = self._seg_lines[-1]
        self._raw_segments[-1] = text
        new_wrapped = self._wrap_segment(text, self._gutters[-1])
        self._lines = self._lines[:len(self._lines) - prev_seg_lines] + new_wrapped
        self._seg_lines[-1] = len(new_wrapped)

    def _wrap_segment(self, seg, gutter=''):
        # a trailing '\n' means "next write starts a new row", not "emit an
        # extra blank row". drop the phantom empty line wrap_ansi produces
        # from str.split('\n'), otherwise every write ending in \n leaves a
        # blank that the next write lands below.
        width = self._cols
        if gutter:
            width = max(1, self._cols - len(ansi_strip(gutter)))
        wrapped = wrap_ansi(seg, width)
        if seg.endswith('\n') and wrapped and ansi_strip(wrapped[-1]) == '':
            wrapped = wrapped[:-1]
        if gutter:
            prefixed = []
            for line in wrapped:
                prefixed.append(gutter + line)
            wrapped = prefixed
        return wrapped

    def rewrap(self, new_cols):
        """re-wrap all content for a new terminal width."""
        self._cols = max(1, new_cols)
        self._lines.clear()
        self._seg_lines = []
        for i, seg in enumerate(self._raw_segments):
            wrapped = self._wrap_segment(seg, self._gutters[i])
            self._lines.extend(wrapped)
            self._seg_lines.append(len(wrapped))

    def line_count(self):
        return len(self._lines)

    @staticmethod
    def _is_blank(line):
        """a display row that shows nothing: empty, or only the gutter bar."""
        plain = ansi_strip(line).strip()
        if plain == '': return True
        return plain == GUTTER_GLYPH

    def ends_blank(self):
        """True when the buffer ends on a blank display line - a write that
        wants one blank line of separation needs to add none."""
        if not self._lines: return False
        if self._partial: return False
        return self._is_blank(self._lines[-1])

    def get_lines(self, start, count):
        """return a slice of display lines for viewport rendering."""
        start = max(0, start)
        end = min(len(self._lines), start + count)
        return self._lines[start:end]

    def get_plain_text(self, line_idx):
        """return ANSI-stripped text for a single line."""
        if 0 <= line_idx < len(self._lines):
            return ansi_strip(self._lines[line_idx])
        return ''

    def search(self, pattern):
        """find every substring match of pattern. returns a list of
        (line_idx, start_col, end_col) tuples in buffer order. columns refer
        to the ANSI-stripped text so they line up with what the user sees on
        screen. zero-width matches are skipped - they would otherwise loop
        the cursor in place."""
        if not pattern: return []
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)

        matches = []
        for i, line in enumerate(self._lines):
            plain = ansi_strip(line)
            for m in rx.finditer(plain):
                if m.end() <= m.start(): continue
                matches.append((i, m.start(), m.end()))
        return matches

    def get_selection_text(self, start_row, start_col, end_row, end_col, line_mode):
        """extract plain text for the given selection range. the gutter
        prefix is not content: it is stripped from each line, and character
        columns (which are screen columns, gutter included) shift with it."""
        if start_row > end_row or (start_row == end_row and start_col > end_col):
            start_row, start_col, end_row, end_col = end_row, end_col, start_row, start_col

        start_row = max(0, start_row)
        end_row = min(len(self._lines) - 1, end_row)
        if start_row > end_row:
            return ''

        lines_out = []
        for i in range(start_row, end_row + 1):
            plain = ansi_strip(self._lines[i])
            offset = 0
            if plain.startswith(GUTTER_GLYPH + ' '):
                plain = plain[2:]
                offset = 2
            sc = max(0, start_col - offset)
            ec = max(0, end_col - offset)
            if line_mode:
                lines_out.append(plain)
            elif i == start_row and i == end_row:
                lines_out.append(plain[sc:ec + 1])
            elif i == start_row:
                lines_out.append(plain[sc:])
            elif i == end_row:
                lines_out.append(plain[:ec + 1])
            else:
                lines_out.append(plain)
        return '\n'.join(lines_out)

    def clear(self):
        self._lines.clear()
        self._raw_segments.clear()
        self._gutters.clear()
        self._seg_lines = []
        self._partial = False
