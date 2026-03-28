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


def init(path: str = LOG_PATH) -> None:
    """Initialise the module-level logger (call once at program start)."""
    global _instance
    _instance = Logger(path)


def log(level: int, msg: str) -> None:
    """Write one log entry.  No-op if init() has not been called."""
    if _instance is not None:
        _instance.log(level, msg)


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
    Main-thread-only state (no lock needed): visible, cursor, scroll, follow,
      depth_max, search_pat, search_matches.
    """

    def __init__(self, path: str) -> None:
        self.path        = path
        self.entries: list[Entry] = []
        self._stack:  list[int]  = []       # monotone stack of open ancestor indices
        self.fold_set: set[int]  = set()
        self._lock   = threading.Lock()
        self._dirty  = threading.Event()
        self._running = True

        # View state (main thread only)
        self.visible:  list[int] = []
        self.cursor    = 0          # index into self.visible
        self.scroll    = 0          # top-of-screen index into self.visible
        self.follow    = True
        self.depth_max = 0          # 0 = show all levels
        self.search_pat:     re.Pattern | None = None
        self.search_matches: list[int]         = []   # entry indices
        self.search_dir = 1         # 1 = forward, -1 = backward

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
                # First child arriving → promote to OPEN_PARENT and auto-fold.
                parent.state = OPEN_PARENT
                self.fold_set.add(parent_idx)

        self.entries.append(entry)
        self._stack.append(i)

    # ── Visibility ────────────────────────────────────────────────────────────

    def _compute_visible(self) -> list[int]:
        """
        Return the list of entry indices that should be shown, respecting
        depth_max and the current fold_set.  Called with self._lock held.
        """
        result: list[int] = []
        skip_until_lvl: int | None = None

        for e in self.entries:
            # Depth filter (skip silently — does not affect fold-skip state
            # because any entry that could "close" a fold has lvl <= fold_lvl
            # <= depth_max and therefore passes this check).
            if self.depth_max > 0 and e.lvl > self.depth_max:
                continue

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
            self.fold_set.discard(entry_idx)
        else:
            self.fold_set.add(entry_idx)

    def _toggle_fold_recursive(self, entry_idx: int) -> None:
        """zA: toggle fold on this entry and all its descendants."""
        e = self.entries[entry_idx]
        descs = self._descendants(entry_idx)
        if entry_idx in self.fold_set:
            # Unfold: clear cursor + all descendants
            self.fold_set.discard(entry_idx)
            for j in descs:
                self.fold_set.discard(j)
        else:
            # Fold: set cursor (if not LEAF) + all non-LEAF descendants
            if e.state != LEAF:
                self.fold_set.add(entry_idx)
            for j in descs:
                if self.entries[j].state != LEAF:
                    self.fold_set.add(j)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> None:
        rows, cols = self._rows, self._cols
        content_rows = rows - 2     # bottom 2 rows: status bar + help bar

        # Snapshot shared state
        with self._lock:
            self.visible   = self._compute_visible()
            entries_snap   = list(self.entries)
            fold_snap      = set(self.fold_set)

        n = len(self.visible)

        # ── Scroll / cursor management ────────────────────────────────────────
        if self.follow and n > 0:
            self.cursor = n - 1
            self.scroll = max(0, n - content_rows)

        if n == 0:
            self.cursor = 0
            self.scroll = 0
        else:
            self.cursor = max(0, min(self.cursor, n - 1))
            if self.cursor < self.scroll:
                self.scroll = self.cursor
            if self.cursor >= self.scroll + content_rows:
                self.scroll = self.cursor - content_rows + 1

        out: list[str] = [_HIDE_CUR]

        # ── Content rows ──────────────────────────────────────────────────────
        for row in range(content_rows):
            vis_i = self.scroll + row
            out.append(_goto(row + 1, 1))
            out.append(_CLEAR_LINE)

            if n == 0 and row == content_rows // 2:
                waiting = f"Waiting for entries in {self.path} …"
                pad = max(0, (cols - len(waiting)) // 2)
                out.append(_DIM + " " * pad + waiting + _RESET)
                continue

            if vis_i >= n:
                continue

            entry_idx = self.visible[vis_i]
            e         = entries_snap[entry_idx]
            is_cursor = (vis_i == self.cursor)

            indent    = "  " * (e.lvl - 1)
            flat_msg  = e.msg.replace("\n", " ").replace("\r", "")

            # Fold indicator
            fold_suffix = ""
            if entry_idx in fold_snap and e.state != LEAF:
                hidden = self._hidden_count(entries_snap, entry_idx)
                live   = " live\u2026" if e.state == OPEN_PARENT else ""
                fold_suffix = f"  {_DIM}\u25b6 [{hidden} hidden{live}]{_RESET}"

            # Max usable width for text
            avail = cols - len(indent) - 2
            if avail < 1:
                avail = 1

            # Search highlight
            if self.search_pat:
                def _hl(m: re.Match) -> str:
                    return f"{_YELLOW_BOLD}{m.group()}{_RESET}"
                display_msg = self.search_pat.sub(_hl, flat_msg[:avail])
            else:
                display_msg = flat_msg[:avail]

            line_body = indent + display_msg + fold_suffix

            if is_cursor:
                out.append(f"{_REV}{indent}{display_msg}{_RESET}{fold_suffix}")
            else:
                out.append(line_body)

        # ── Status bar ────────────────────────────────────────────────────────
        out.append(_goto(rows - 1, 1))
        out.append(_CLEAR_LINE)

        pos_str    = f"{self.cursor + 1}/{n}" if n else "0/0"
        total_str  = f"[{len(entries_snap)} total]"
        follow_str = " \033[32mFOLLOW\033[0m"  if self.follow    else ""
        depth_str  = f" depth:{self.depth_max}" if self.depth_max else ""
        srch_str   = (f" /{self.search_pat.pattern}" if self.search_pat else "")
        status     = (
            f"{_BOLD}cai log{_RESET}  {self.path}"
            f"  {pos_str} {total_str}"
            f"{follow_str}{depth_str}{srch_str}"
        )
        out.append(status[:cols])

        # ── Help bar ──────────────────────────────────────────────────────────
        out.append(_goto(rows, 1))
        out.append(_CLEAR_LINE)
        help_txt = (
            " Tab:fold  zA:fold-all  zz/zt/zb:align  /:search  ?:search\u2191  "
            "n/N:next/prev  1\u20139:depth  0:all  f:follow  G:bottom  q:quit"
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
                self._update_search_matches()
                self._jump_to_match(direction)
            except re.error:
                pass

    def _update_search_matches(self) -> None:
        if not self.search_pat:
            self.search_matches = []
            return
        self.search_matches = [
            e.idx for e in self.entries if self.search_pat.search(e.msg)
        ]

    def _jump_to_match(self, direction: int) -> None:
        if not self.search_matches or not self.visible:
            return
        vis_set = set(self.visible)
        current = self.visible[self.cursor] if self.cursor < len(self.visible) else -1
        candidates = [m for m in self.search_matches if m in vis_set]
        if not candidates:
            return
        if direction == 1:
            ahead = [m for m in candidates if m > current]
            target = ahead[0] if ahead else candidates[0]
        else:
            behind = [m for m in candidates if m < current]
            target = behind[-1] if behind else candidates[-1]
        try:
            self.cursor = self.visible.index(target)
            self.follow = False
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
        elif key in ("j", "\033[B"):          # down
            if self.cursor < n - 1:
                self.cursor += 1
            self.follow = False
            if self.cursor >= self.scroll + content_rows:
                self.scroll = self.cursor - content_rows + 1

        elif key in ("k", "\033[A"):          # up
            if self.cursor > 0:
                self.cursor -= 1
            if self.cursor < self.scroll:
                self.scroll = self.cursor
            self.follow = False

        elif key == "\x04":                   # Ctrl-d  half-page down
            step = max(1, content_rows // 2)
            self.cursor = min(n - 1, self.cursor + step) if n else 0
            self.scroll = min(max(0, n - content_rows), self.scroll + step)
            self.follow = False

        elif key == "\x15":                   # Ctrl-u  half-page up
            step = max(1, content_rows // 2)
            self.cursor = max(0, self.cursor - step)
            self.scroll = max(0, self.scroll - step)
            self.follow = False

        elif key == "g":                      # top
            self.cursor = 0
            self.scroll = 0
            self.follow = False

        elif key == "G":                      # bottom + enable follow
            self.follow = True
            # render() will set cursor + scroll via follow logic

        elif key == "f":                      # toggle follow
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
            elif next_key == "z":             # zz — cursor to centre
                self.scroll = max(0, self.cursor - content_rows // 2)
                self.follow = False
            elif next_key == "t":             # zt — cursor to top
                self.scroll = self.cursor
                self.follow = False
            elif next_key == "b":             # zb — cursor to bottom
                self.scroll = max(0, self.cursor - content_rows + 1)
                self.follow = False

        # ── Search ────────────────────────────────────────────────────────────
        elif key == "/":
            self._enter_search(1)

        elif key == "?":
            self._enter_search(-1)

        elif key == "n":
            self._jump_to_match(self.search_dir)

        elif key == "N":
            self._jump_to_match(-self.search_dir)

        # ── Depth filter ──────────────────────────────────────────────────────
        elif key in "123456789":
            self.depth_max = int(key)
            self._reanchor(anchor)

        elif key == "0":
            self.depth_max = 0
            self._reanchor(anchor)

        return True

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
