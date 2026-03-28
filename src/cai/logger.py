"""
logger.py — Hierarchical JSONL logger + interactive TUI viewer for CAI.

Public API
----------
    init(path)       — initialise the module-level logger singleton
    log(level, msg)  — write one structured entry  (level >= 1)
    launch_tui(path) — open the full-screen interactive log viewer
"""

from __future__ import annotations

import json
import os
import re
import select
import signal
import sys
import termios
import threading
import time
import tty
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ── Defaults ─────────────────────────────────────────────────────────────────

LOG_PATH = "/tmp/cai/cai.log"

# ── Entry states ──────────────────────────────────────────────────────────────

LEAF          = 0   # no children ever observed
OPEN_PARENT   = 1   # has children; block not yet closed
CLOSED_PARENT = 2   # has children; block is fully bounded

# ── ANSI helpers ──────────────────────────────────────────────────────────────

_ALT_ENTER  = "\033[?1049h\033[H"
_ALT_LEAVE  = "\033[?1049l"
_HIDE_CUR   = "\033[?25l"
_SHOW_CUR   = "\033[?25h"
_CLEAR_LINE = "\033[2K\r"
_BOLD       = "\033[1m"
_DIM        = "\033[2m"
_REV        = "\033[7m"
_YELLOW_BOLD= "\033[1;33m"
_RESET      = "\033[0m"

# ── Level-based color palette (structural — one color per nesting depth) ──
_LEVEL_COLORS: tuple[str, ...] = (
    "\033[97m",   # lvl 1  — bright white   (root, most prominent)
    "\033[96m",   # lvl 2  — bright cyan
    "\033[92m",   # lvl 3  — bright green
    "\033[93m",   # lvl 4  — bright yellow
    "\033[95m",   # lvl 5  — bright magenta
    "\033[94m",   # lvl 6  — bright blue
    "\033[91m",   # lvl 7  — bright red
    "\033[37m",   # lvl 8  — normal white (fallback for deep nesting)
)


def _level_color(lvl: int) -> str:
    return _LEVEL_COLORS[(lvl - 1) % len(_LEVEL_COLORS)]


def _goto(row: int, col: int = 1) -> str:
    return f"\033[{row};{col}H"


def _term_size() -> tuple[int, int]:
    import shutil
    s = shutil.get_terminal_size((80, 24))
    return s.lines, s.columns


# ══════════════════════════════════════════════════════════════════════════════
# Logger — thread-safe JSONL writer
# ══════════════════════════════════════════════════════════════════════════════

class Logger:
    """Thread-safe append-only JSONL log writer."""

    def __init__(self, path: str = LOG_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, level: int, msg: str) -> None:
        record = {
            "ts":  datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "lvl": level,
            "msg": msg,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


# Module-level singleton
_instance: Logger | None = None

# Nesting level context variable — incremented by nest() context manager.
# log() adds this to the caller-supplied level so the same code produces
# deeper indentation when called from inside a harness block.
_base: ContextVar[int] = ContextVar('cai_log_base', default=0)


@contextmanager
def nest(delta: int = 1):
    """Context manager: increase the log nesting level by *delta* for all
    log() calls made within this block (including transitively called code).

    Usage::

        with nest(1):
            log(1, "this appears one level deeper than the caller's base")
    """
    tok = _base.set(_base.get() + delta)
    try:
        yield
    finally:
        _base.reset(tok)


def init(path: str = LOG_PATH) -> None:
    """Initialise the module-level logger (call once at program start)."""
    global _instance
    _instance = Logger(path)


def log(level: int, msg: str) -> None:
    """Write one log entry at *base_level + level*.  No-op if init() has not been called."""
    if _instance is not None:
        _instance.log(_base.get() + level, msg)


# ══════════════════════════════════════════════════════════════════════════════
# Entry — one parsed log record
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Entry:
    idx:   int
    ts:    str
    lvl:   int
    msg:   str
    state: int = LEAF      # LEAF | OPEN_PARENT | CLOSED_PARENT


# ══════════════════════════════════════════════════════════════════════════════
# LogViewer — the TUI
# ══════════════════════════════════════════════════════════════════════════════

class LogViewer:
    """
    Full-screen, dependency-free TUI log viewer.

    Thread model
    ────────────
      Reader thread  – tails the log file, ingests new Entry objects, sets
                       self._dirty to wake the render loop.
      Main thread    – render + input event loop.

    Shared state protected by self._lock: entries, _stack, fold_set.
    Main-thread-only state (no lock needed): visible, cursor, scroll_row,
      follow, depth_max, search_pat, search_matches.

    Scroll model
    ────────────
      scroll_row  — absolute screen-row offset of the top of the viewport.
                    One "screen row" = one terminal line.  Multi-line entries
                    occupy as many screen rows as they need.
      cursor      — index into self.visible (entry-level, not row-level).
                    j/k navigate entries; Ctrl-u/d scroll rows and then move
                    the cursor to the first entry in the new viewport.
    """

    def __init__(self, path: str) -> None:
        self.path        = path
        self.entries: list[Entry] = []
        self._stack:  list[int]  = []       # monotone stack of open ancestor indices
        self.fold_set: set[int]  = set()
        self.force_open_set: set[int] = set()   # explicitly unfolded; overrides depth_max
        self._auto_fold_set: set[int] = set()   # auto-folded while follow was off
        self._lock   = threading.Lock()
        self._dirty  = threading.Event()
        self._running = True

        # View state (main thread only)
        self.visible:   list[int] = []
        self.cursor     = 0         # index into self.visible  (entry-level)
        self.scroll_row = 0         # absolute screen-row at top of viewport
        self.follow     = True
        self.search_pat:     re.Pattern | None = None
        self.search_matches: list[int]         = []   # entry indices
        self.search_dir = 1         # 1 = forward, -1 = backward

        # Cached layout (updated at start of every _render call, main thread only)
        self._entry_row_start: list[int] = []   # abs row of first line of visible[i]
        self._row_spans:       list[int] = []   # number of screen rows for visible[i]
        self._total_rows       = 0              # sum of all _row_spans

        self._rows = 24
        self._cols = 80

        # /dev/tty for keyboard input even when stdin is piped
        self._tty_file = open("/dev/tty", "rb+", buffering=0)
        self._tty_fd   = self._tty_file.fileno()

    # ── Entry ingestion ───────────────────────────────────────────────────────

    def _ingest(self, entry: Entry) -> None:
        """
        Append entry to self.entries, update the monotone stack, and
        auto-fold newly discovered parents.  Must be called with self._lock held.
        """
        i   = entry.idx
        lvl = entry.lvl

        # Pop stack entries at same-or-deeper level → they are now closed.
        while self._stack and self.entries[self._stack[-1]].lvl >= lvl:
            closed_idx = self._stack.pop()
            if self.entries[closed_idx].state == OPEN_PARENT:
                self.entries[closed_idx].state = CLOSED_PARENT
            # LEAF stays LEAF (no children were ever ingested under it)

        # The current stack top is now the direct parent of this entry.
        if self._stack:
            parent_idx = self._stack[-1]
            parent = self.entries[parent_idx]
            if parent.state == LEAF:
                # First child arriving → promote to OPEN_PARENT.
                # Auto-fold only when the user is not actively following the
                # live tail; in follow mode keep everything expanded so new
                # output streams in fully visible.
                parent.state = OPEN_PARENT
                if not self.follow:
                    self.fold_set.add(parent_idx)
                    self._auto_fold_set.add(parent_idx)

        self.entries.append(entry)
        self._stack.append(i)

    # ── Visibility ────────────────────────────────────────────────────────────

    def _compute_visible(self) -> list[int]:
        """
        Return the list of entry indices that should be shown, respecting
        the current fold_set.  Called with self._lock held.
        """
        result: list[int] = []
        skip_until_lvl: int | None = None

        for e in self.entries:
            # Fold skip
            if skip_until_lvl is not None:
                if e.lvl <= skip_until_lvl:
                    skip_until_lvl = None   # exited the folded subtree
                else:
                    continue                # hidden child

            result.append(e.idx)

            if e.idx in self.fold_set and e.state != LEAF:
                skip_until_lvl = e.lvl     # hide everything deeper

        return result

    # ── Descendant helpers ────────────────────────────────────────────────────

    def _hidden_count(self, entries_snap: list[Entry], i: int) -> int:
        """Count entries that are children (direct or indirect) of entry i."""
        lvl = entries_snap[i].lvl
        count = 0
        for j in range(i + 1, len(entries_snap)):
            if entries_snap[j].lvl <= lvl:
                break
            count += 1
        return count

    def _descendants(self, i: int) -> list[int]:
        """Return indices of all descendant entries of entry i."""
        lvl = self.entries[i].lvl
        result = []
        for j in range(i + 1, len(self.entries)):
            if self.entries[j].lvl <= lvl:
                break
            result.append(j)
        return result

    # ── Fold actions ──────────────────────────────────────────────────────────

    def _toggle_fold(self, entry_idx: int) -> None:
        """Tab: toggle fold on this entry only; children keep their state."""
        e = self.entries[entry_idx]
        if e.state == LEAF:
            return
        if entry_idx in self.fold_set:
            # Unfolding — mark as explicitly opened so depth_max is bypassed.
            self.fold_set.discard(entry_idx)
            self._auto_fold_set.discard(entry_idx)
            self.force_open_set.add(entry_idx)
        else:
            # Folding — remove from force_open so depth override is dropped.
            self.fold_set.add(entry_idx)
            self._auto_fold_set.discard(entry_idx)
            self.force_open_set.discard(entry_idx)

    def _toggle_fold_recursive(self, entry_idx: int) -> None:
        """zA: toggle fold on this entry and all its descendants."""
        e = self.entries[entry_idx]
        descs = self._descendants(entry_idx)
        if entry_idx in self.fold_set:
            # Unfold: clear cursor + all descendants; mark as force-opened.
            self.fold_set.discard(entry_idx)
            self._auto_fold_set.discard(entry_idx)
            self.force_open_set.add(entry_idx)
            for j in descs:
                self.fold_set.discard(j)
                self._auto_fold_set.discard(j)
                if self.entries[j].state != LEAF:
                    self.force_open_set.add(j)
        else:
            # Fold: set cursor (if not LEAF) + all non-LEAF descendants.
            if e.state != LEAF:
                self.fold_set.add(entry_idx)
                self._auto_fold_set.discard(entry_idx)
                self.force_open_set.discard(entry_idx)
            for j in descs:
                if self.entries[j].state != LEAF:
                    self.fold_set.add(j)
                    self._auto_fold_set.discard(j)
                self.force_open_set.discard(j)

    def _apply_depth(self, depth: int) -> None:
        """
        One-time action: fold/unfold entries relative to the minimum level
        present in the log.  depth=1 shows only the shallowest level,
        depth=2 shows 2 levels deep from the minimum, etc.
        depth=0 unfolds everything.
        Must be called with self._lock held.
        """
        if depth == 0:
            self.fold_set.clear()
            self._auto_fold_set.clear()
            self.force_open_set.clear()
            return
        if not self.entries:
            return
        min_lvl = min(e.lvl for e in self.entries)
        # Absolute level cutoff: entries at this level and deeper get folded.
        cutoff = min_lvl + depth - 1
        for e in self.entries:
            if e.state == LEAF:
                continue
            if e.lvl < cutoff:
                # Unfold: let children be visible
                self.fold_set.discard(e.idx)
                self._auto_fold_set.discard(e.idx)
            else:
                # Fold: hide children (level >= cutoff)
                self.fold_set.add(e.idx)
                self._auto_fold_set.discard(e.idx)
                self.force_open_set.discard(e.idx)

    # ── Display-line helpers ──────────────────────────────────────────────────

    def _entry_display_lines(
        self,
        e: Entry,
        cols: int,
        fold_snap: set[int],
        entries_snap: list[Entry],
    ) -> list[tuple[str, str]]:
        """
        Return the screen lines for one entry as a list of (main_text, suffix).

          main_text — plain text (no ANSI); search highlight is applied by caller.
          suffix    — ready-to-print ANSI string appended after main_text
                      (used for the fold indicator; empty for normal lines).

        Folded entries always produce exactly 1 tuple.
        Unfolded entries expand \\n and wrap long lines; all display lines
        beyond the first are prefixed with  indent + "│ "  so boundaries
        between entries are always visible.
        """
        indent     = "  " * (e.lvl - 1)
        indent_len = len(indent)

        # ── Folded: one line ──────────────────────────────────────────────────
        if e.idx in fold_snap and e.state != LEAF:
            hidden        = self._hidden_count(entries_snap, e.idx)
            live          = " live\u2026" if e.state == OPEN_PARENT else ""
            ind_plain     = f"  \u25b6 [{hidden} hidden{live}]"
            ind_ansi      = f"  {_DIM}\u25b6 [{hidden} hidden{live}]{_RESET}"
            avail         = max(0, cols - indent_len - len(ind_plain))
            first_line    = e.msg.split("\n")[0]
            return [(indent + first_line[:avail], ind_ansi)]

        # ── Unfolded: full multi-line expansion ───────────────────────────────
        raw_lines = e.msg.split("\n")
        result: list[tuple[str, str]] = []

        for line_num, raw_line in enumerate(raw_lines):
            pfx        = indent if line_num == 0 else indent + "\u2502 "
            text_avail = max(1, cols - len(pfx))

            if not raw_line:
                result.append((pfx, ""))
                continue

            pos            = 0
            first_segment  = True
            while pos < len(raw_line):
                chunk = raw_line[pos : pos + text_avail]
                result.append((pfx + chunk, ""))
                pos += text_avail
                if first_segment and pos < len(raw_line):
                    # Wrapping within one raw line: use continuation prefix
                    pfx        = indent + "\u2502 "
                    text_avail = max(1, cols - len(pfx))
                first_segment = False

        return result if result else [(indent, "")]

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> None:
        rows, cols    = self._rows, self._cols
        content_rows  = rows - 2          # bottom 2 rows: status bar + help bar

        # ── Snapshot shared state ─────────────────────────────────────────────
        with self._lock:
            self.visible  = self._compute_visible()
            entries_snap  = list(self.entries)
            fold_snap     = set(self.fold_set)

        n = len(self.visible)

        # ── Build row-layout index ────────────────────────────────────────────
        # entry_row_start[i]  — absolute screen-row where visible[i] begins
        # row_spans[i]        — how many screen rows visible[i] occupies
        entry_row_start: list[int] = []
        row_spans:       list[int] = []
        # Also build a flat map: abs_row → (vis_i, line_idx, main_text, suffix)
        # We build it lazily during rendering to avoid two passes.
        total_rows = 0
        for i, entry_idx in enumerate(self.visible):
            e       = entries_snap[entry_idx]
            lines   = self._entry_display_lines(e, cols, fold_snap, entries_snap)
            span    = len(lines)
            entry_row_start.append(total_rows)
            row_spans.append(span)
            total_rows += span

        # Cache for _handle_key (zz/zt/zb, Ctrl-d/u need these)
        self._entry_row_start = entry_row_start
        self._row_spans       = row_spans
        self._total_rows      = total_rows

        # ── Follow mode ───────────────────────────────────────────────────────
        if self.follow and n > 0:
            self.cursor     = n - 1
            self.scroll_row = max(0, total_rows - content_rows)

        # ── Clamp cursor ──────────────────────────────────────────────────────
        if n == 0:
            self.cursor     = 0
            self.scroll_row = 0
        else:
            self.cursor = max(0, min(self.cursor, n - 1))
            cur_start   = entry_row_start[self.cursor]
            cur_end     = cur_start + row_spans[self.cursor]

            # Cursor entry scrolled above viewport → snap scroll to its first line
            if cur_start < self.scroll_row:
                self.scroll_row = cur_start
            # Cursor entry's first line scrolled below viewport → scroll down
            elif cur_start >= self.scroll_row + content_rows:
                entry_h = row_spans[self.cursor]
                if entry_h <= content_rows:
                    self.scroll_row = cur_end - content_rows
                else:
                    self.scroll_row = cur_start

        self.scroll_row = max(0, self.scroll_row)

        # ── Build screen_row_map ──────────────────────────────────────────────
        # screen_row_map[sr] = (vis_i, main_text, suffix) or None
        screen_row_map: list[tuple[int, str, str] | None] = [None] * content_rows

        for i, entry_idx in enumerate(self.visible):
            e           = entries_snap[entry_idx]
            lines       = self._entry_display_lines(e, cols, fold_snap, entries_snap)
            abs_start   = entry_row_start[i]
            abs_end     = abs_start + len(lines)

            # Skip entries entirely above or below viewport
            if abs_end <= self.scroll_row:
                continue
            if abs_start >= self.scroll_row + content_rows:
                break

            for li, (main_text, suffix) in enumerate(lines):
                abs_row = abs_start + li
                sr      = abs_row - self.scroll_row
                if 0 <= sr < content_rows:
                    screen_row_map[sr] = (i, main_text, suffix)

        # ── Draw content rows ─────────────────────────────────────────────────
        out: list[str] = [_HIDE_CUR]

        for sr in range(content_rows):
            out.append(_goto(sr + 1, 1))
            out.append(_CLEAR_LINE)

            cell = screen_row_map[sr]

            if cell is None:
                # Empty row — show "waiting" message centred when log is empty
                if n == 0 and sr == content_rows // 2:
                    waiting = f"Waiting for entries in {self.path} \u2026"
                    pad     = max(0, (cols - len(waiting)) // 2)
                    out.append(_DIM + " " * pad + waiting + _RESET)
                continue

            vis_i, main_text, suffix = cell
            is_cursor = (vis_i == self.cursor)
            lc = _level_color(entries_snap[self.visible[vis_i]].lvl)

            # Search highlight on main_text; restore level color after each match
            if self.search_pat:
                highlighted = self.search_pat.sub(
                    lambda m: f"{_YELLOW_BOLD}{m.group()}{_RESET}{lc}", main_text
                )
            else:
                highlighted = main_text

            if is_cursor:
                # Pad to full width so reverse-video fills the whole line
                pad     = max(0, cols - len(main_text))
                out.append(f"{_REV}{highlighted}{' ' * pad}{_RESET}{suffix}")
            else:
                out.append(f"{lc}{highlighted}{_RESET}{suffix}")

        # ── Status bar ────────────────────────────────────────────────────────
        out.append(_goto(rows - 1, 1))
        out.append(_CLEAR_LINE)

        pos_str    = f"{self.cursor + 1}/{n}" if n else "0/0"
        total_str  = f"[{len(entries_snap)} total]"
        follow_str = " \033[1;32mFOLLOW\033[0m" if self.follow else ""
        srch_str   = (f" /{self.search_pat.pattern}" if self.search_pat else "")
        status     = (
            f"\033[1;36mcai log\033[0m  {_DIM}{self.path}{_RESET}"
            f"  {pos_str} {total_str}"
            f"{follow_str}{srch_str}"
        )
        out.append(status[:cols])

        # ── Help bar ──────────────────────────────────────────────────────────
        out.append(_goto(rows, 1))
        out.append(_CLEAR_LINE)
        help_txt = (
            " Tab:fold  zA:fold-all  zz/zt/zb:align  /:search  ?:search\u2191  "
            "n/N:next/prev  0:depth1  1\u20138:depth2\u20139  9:all  F:follow  G:bottom  q:quit"
        )
        out.append(_DIM + help_txt[:cols] + _RESET)

        out.append(_SHOW_CUR)
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    # ── Keyboard input ────────────────────────────────────────────────────────

    def _read_key(self) -> str:
        """Read one logical keypress from /dev/tty (handles escape sequences)."""
        ch = os.read(self._tty_fd, 1).decode("utf-8", errors="replace")
        if ch == "\033":
            ready, _, _ = select.select([self._tty_fd], [], [], 0.05)
            if ready:
                rest = os.read(self._tty_fd, 16).decode("utf-8", errors="replace")
                return ch + rest
        return ch

    def _enter_search(self, direction: int) -> None:
        """Draw an inline search prompt and read a regex pattern."""
        rows, cols = self._rows, self._cols
        prompt_chr = "/" if direction == 1 else "?"
        buf = ""

        def _draw() -> None:
            sys.stdout.write(_goto(rows, 1) + _CLEAR_LINE + prompt_chr + buf)
            sys.stdout.flush()

        _draw()
        while True:
            raw = os.read(self._tty_fd, 16)
            ch  = raw.decode("utf-8", errors="replace")
            if ch in ("\r", "\n"):
                break
            elif ch in ("\x03", "\x1b"):
                buf = ""
                break
            elif ch in ("\x7f", "\x08"):
                buf = buf[:-1]
            else:
                buf += ch
            _draw()

        if buf:
            try:
                self.search_pat = re.compile(buf, re.IGNORECASE)
                self.search_dir = direction
                self._jump_to_match(direction)
            except re.error:
                pass

    def _update_search_matches(self) -> None:
        if not self.search_pat:
            self.search_matches = []
            return
        with self._lock:
            entries_snap = list(self.entries)
        self.search_matches = [
            e.idx for e in entries_snap if self.search_pat.search(e.msg)
        ]

    def _ancestors(self, i: int) -> list[int]:
        """Return indices of all ancestor entries of entry i.
        Must be called with self._lock held."""
        min_lvl_seen = self.entries[i].lvl
        result = []
        for j in range(i - 1, -1, -1):
            if self.entries[j].lvl < min_lvl_seen:
                result.append(j)
                min_lvl_seen = self.entries[j].lvl
        return result

    def _unfold_to(self, entry_idx: int) -> None:
        """Unfold all folded ancestors of entry_idx so it becomes visible.
        Must be called with self._lock held."""
        for anc in self._ancestors(entry_idx):
            if anc in self.fold_set:
                self.fold_set.discard(anc)
                self._auto_fold_set.discard(anc)
                self.force_open_set.add(anc)

    def _jump_to_match(self, direction: int) -> None:
        if not self.search_pat:
            return
        # Always recompute from current state so stale caches don't block jumps.
        with self._lock:
            entries_snap    = list(self.entries)
            current_visible = self._compute_visible()
        # Search ALL entries, not just visible ones.
        matches = [
            e.idx for e in entries_snap if self.search_pat.search(e.msg)
        ]
        self.search_matches = matches
        if not matches:
            return
        current = current_visible[self.cursor] if self.cursor < len(current_visible) else -1
        if direction == 1:
            ahead  = [m for m in matches if m > current]
            target = ahead[0] if ahead else matches[0]
        else:
            behind = [m for m in matches if m < current]
            target = behind[-1] if behind else matches[-1]
        # If target is hidden inside a folded node, unfold its ancestors.
        vis_set = set(current_visible)
        if target not in vis_set:
            with self._lock:
                self._unfold_to(target)
                new_visible = self._compute_visible()
        else:
            new_visible = current_visible
        try:
            self.cursor  = new_visible.index(target)
            self.visible = new_visible
            self.follow  = False
        except ValueError:
            pass

    # ── Input dispatch ────────────────────────────────────────────────────────

    def _handle_key(self, key: str) -> bool:
        """Process one keypress. Returns False to request quit."""
        rows, _ = self._rows, self._cols
        content_rows = rows - 2
        n = len(self.visible)

        # Save current entry so we can re-anchor cursor after fold operations.
        anchor = self.visible[self.cursor] if n > 0 else None

        # ── Quit ──────────────────────────────────────────────────────────────
        if key in ("q", "\x03"):
            return False

        # ── Vertical navigation ───────────────────────────────────────────────
        elif key in ("j", "\033[B"):          # down one entry
            if self.cursor < n - 1:
                self.cursor += 1
            self.follow = False
            # Scroll so the new cursor entry's first line is visible;
            # render() enforces this but we nudge scroll_row here too so
            # the screen doesn't jump to the entry start unnecessarily.
            if n > 0 and self.cursor < len(self._entry_row_start):
                cur_start = self._entry_row_start[self.cursor]
                if cur_start >= self.scroll_row + content_rows:
                    self.scroll_row = cur_start - content_rows + 1

        elif key in ("k", "\033[A"):          # up one entry
            if self.cursor > 0:
                self.cursor -= 1
            self.follow = False
            if n > 0 and self.cursor < len(self._entry_row_start):
                cur_start = self._entry_row_start[self.cursor]
                if cur_start < self.scroll_row:
                    self.scroll_row = cur_start

        elif key == "\x04":                   # Ctrl-d  half-page down (rows)
            step            = max(1, content_rows // 2)
            self.scroll_row = min(
                max(0, self._total_rows - content_rows),
                self.scroll_row + step,
            )
            # Move cursor to first entry that starts at or after new scroll_row
            self.cursor = self._first_entry_at_or_after(self.scroll_row)
            self.follow = False

        elif key == "\x15":                   # Ctrl-u  half-page up (rows)
            step            = max(1, content_rows // 2)
            self.scroll_row = max(0, self.scroll_row - step)
            self.cursor     = self._first_entry_at_or_after(self.scroll_row)
            self.follow     = False

        elif key == "g":                      # top
            self.cursor     = 0
            self.scroll_row = 0
            self.follow     = False

        elif key == "G":                      # bottom
            if n > 0:
                self.cursor     = n - 1
                self.scroll_row = max(0, self._total_rows - content_rows)
            self.follow = False

        elif key == "F":                      # toggle follow
            self.follow = not self.follow

        # ── Fold / unfold ─────────────────────────────────────────────────────
        elif key == "\t":                     # Tab — non-recursive toggle
            if n > 0:
                with self._lock:
                    self._toggle_fold(self.visible[self.cursor])
                self._reanchor(anchor)

        elif key == "z":                      # start of z* sequences
            next_key = self._read_key()
            if next_key == "A" and n > 0:
                with self._lock:
                    self._toggle_fold_recursive(self.visible[self.cursor])
                self._reanchor(anchor)
            elif next_key == "z" and self.cursor < len(self._entry_row_start):
                # zz — cursor entry centred in viewport
                cur_start   = self._entry_row_start[self.cursor]
                span        = self._row_spans[self.cursor] if self.cursor < len(self._row_spans) else 1
                self.scroll_row = max(0, cur_start - (content_rows - span) // 2)
                self.follow = False
            elif next_key == "t" and self.cursor < len(self._entry_row_start):
                # zt — cursor entry at top of viewport
                self.scroll_row = self._entry_row_start[self.cursor]
                self.follow     = False
            elif next_key == "b" and self.cursor < len(self._entry_row_start):
                # zb — cursor entry at bottom of viewport
                cur_start   = self._entry_row_start[self.cursor]
                span        = self._row_spans[self.cursor] if self.cursor < len(self._row_spans) else 1
                self.scroll_row = max(0, cur_start + span - content_rows)
                self.follow     = False

        # ── Search ────────────────────────────────────────────────────────────
        elif key == "/":
            self._enter_search(1)

        elif key == "?":
            self._enter_search(-1)

        elif key == "n":
            self._jump_to_match(self.search_dir)

        elif key == "N":
            self._jump_to_match(-self.search_dir)

        # ── Depth action (one-time fold to depth) ─────────────────────────────
        # 0 = first (shallowest) level only, 1–8 = 2–9 levels, 9 = all
        elif key in "0123456789":
            digit = int(key)
            with self._lock:
                self._apply_depth((digit + 1) % 10)
            self._reanchor(anchor)

        return True

    def _first_entry_at_or_after(self, abs_row: int) -> int:
        """
        Return the index into self.visible of the first entry whose first line
        is at or after abs_row.  Falls back to the last entry if none qualifies.
        """
        for i, start in enumerate(self._entry_row_start):
            if start >= abs_row:
                return i
        return max(0, len(self._entry_row_start) - 1)

    def _reanchor(self, anchor: int | None) -> None:
        """
        After the visible list changes (fold / depth), try to keep the cursor
        on the same entry it was on.  If that entry is now hidden, stay put
        (clamped).
        """
        with self._lock:
            new_vis = self._compute_visible()
        if anchor is not None and anchor in new_vis:
            self.cursor = new_vis.index(anchor)
        else:
            self.cursor = max(0, min(self.cursor, len(new_vis) - 1))

    # ── Background reader ─────────────────────────────────────────────────────

    def _reader(self) -> None:
        """Tail the log file and ingest new JSONL entries."""
        while self._running and not os.path.exists(self.path):
            self._dirty.set()   # trigger "waiting" render
            time.sleep(0.25)

        try:
            fh = open(self.path, "r", encoding="utf-8")
        except OSError:
            return

        try:
            while self._running:
                line = fh.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    entry = Entry(
                        idx=len(self.entries),
                        ts=obj.get("ts", ""),
                        lvl=max(1, int(obj.get("lvl", 1))),
                        msg=str(obj.get("msg", "")),
                    )
                    with self._lock:
                        self._ingest(entry)
                    self._dirty.set()
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
        finally:
            fh.close()

    # ── Main run loop ──────────────────────────────────────────────────────────

    def run(self) -> None:
        """Enter the TUI, run the event loop, restore terminal on exit."""
        old_attrs = termios.tcgetattr(self._tty_fd)

        def _cleanup() -> None:
            try:
                termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass
            sys.stdout.write(_ALT_LEAVE + _SHOW_CUR)
            sys.stdout.flush()
            self._tty_file.close()

        def _sigwinch(_sig: int, _frame: object) -> None:
            self._rows, self._cols = _term_size()
            self._dirty.set()

        signal.signal(signal.SIGWINCH, _sigwinch)
        self._rows, self._cols = _term_size()

        sys.stdout.write(_ALT_ENTER + _HIDE_CUR)
        sys.stdout.flush()

        tty.setraw(self._tty_fd)

        reader_thread = threading.Thread(target=self._reader, daemon=True)
        reader_thread.start()

        try:
            self._render()
            while True:
                # Wait for a keypress or new data (100 ms tick)
                ready, _, _ = select.select([self._tty_fd], [], [], 0.1)
                if ready:
                    key = self._read_key()
                    if not self._handle_key(key):
                        break
                    self._dirty.clear()
                    self._render()
                elif self._dirty.is_set():
                    self._dirty.clear()
                    self._render()
        finally:
            self._running = False
            _cleanup()


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def launch_tui(path: str = LOG_PATH) -> None:
    """Launch the full-screen log viewer (blocks until the user quits)."""
    viewer = LogViewer(path)
    viewer.run()
