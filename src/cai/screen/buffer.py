"""content buffer for the alternate-screen TUI: stores all rendered display
lines in memory, supports scrolling, search, selection, and re-wrapping on
terminal resize."""

import re
from .ansi import wrap_ansi, ansi_strip


class ContentBuffer:
    """stores terminal-rendered lines for the scrollable conversation area."""

    def __init__(self, cols):
        self._cols = max(1, cols)
        self._lines = []                 # wrapped display lines (with ANSI)
        self._raw_segments = []          # raw text segments for re-wrap
        self._partial = False            # last segment ended mid-line
        self._last_seg_line_count = 0    # wrapped line count of the last segment

    def append_text(self, text):
        """append text to the buffer. returns number of new display lines."""
        if not text: return 0

        old_count = len(self._lines)

        if self._partial and self._raw_segments:
            self._raw_segments[-1] += text
            new_wrapped = self._wrap_segment(self._raw_segments[-1])
            prev_seg_lines = self._last_seg_line_count
            self._lines = self._lines[:len(self._lines) - prev_seg_lines] + new_wrapped
            self._last_seg_line_count = len(new_wrapped)
        else:
            self._raw_segments.append(text)
            new_wrapped = self._wrap_segment(text)
            self._lines.extend(new_wrapped)
            self._last_seg_line_count = len(new_wrapped)

        self._partial = not text.endswith('\n')

        return len(self._lines) - old_count

    def _wrap_segment(self, seg):
        # a trailing '\n' means "next write starts a new row", not "emit an
        # extra blank row". drop the phantom empty line wrap_ansi produces
        # from str.split('\n'), otherwise every write ending in \n leaves a
        # blank that the next write lands below.
        wrapped = wrap_ansi(seg, self._cols)
        if seg.endswith('\n') and wrapped and ansi_strip(wrapped[-1]) == '':
            wrapped = wrapped[:-1]
        return wrapped

    def rewrap(self, new_cols):
        """re-wrap all content for a new terminal width."""
        self._cols = max(1, new_cols)
        self._lines.clear()
        for seg in self._raw_segments:
            self._lines.extend(self._wrap_segment(seg))
        if self._raw_segments:
            self._last_seg_line_count = len(self._wrap_segment(self._raw_segments[-1]))

    def line_count(self):
        return len(self._lines)

    def ends_blank(self):
        """True when the buffer ends on a blank display line - a write that
        wants one blank line of separation needs to add none."""
        if not self._lines: return False
        if self._partial: return False
        return ansi_strip(self._lines[-1]) == ''

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
        """extract plain text for the given selection range."""
        if start_row > end_row or (start_row == end_row and start_col > end_col):
            start_row, start_col, end_row, end_col = end_row, end_col, start_row, start_col

        start_row = max(0, start_row)
        end_row = min(len(self._lines) - 1, end_row)
        if start_row > end_row:
            return ''

        lines_out = []
        for i in range(start_row, end_row + 1):
            plain = ansi_strip(self._lines[i])
            if line_mode:
                lines_out.append(plain)
            elif i == start_row and i == end_row:
                lines_out.append(plain[start_col:end_col + 1])
            elif i == start_row:
                lines_out.append(plain[start_col:])
            elif i == end_row:
                lines_out.append(plain[:end_col + 1])
            else:
                lines_out.append(plain)
        return '\n'.join(lines_out)

    def clear(self):
        self._lines.clear()
        self._raw_segments.clear()
        self._partial = False
        self._last_seg_line_count = 0
