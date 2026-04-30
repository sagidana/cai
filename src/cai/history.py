"""In-memory undo tree for messages[] mutations in interactive mode.

Plugs into the existing hook bus via the ``messages_mutated`` event:
``action_interactive`` registers ``("messages_mutated", tracker.on_event)`` on
``userconfig._user_hooks`` at startup and removes it on exit. Outside
interactive mode nobody registers a handler, so the instrumented
mutation sites (see ``cai.llm.fire_event``) iterate an empty match set.

The tracker captures a deep copy of messages on each distinct mutation
and keeps them as a tree: ``undo`` walks to the parent, ``redo`` to the
most-recently-taken child, and ``jump`` addresses any node by id. Undo
and jump rewrite the bound list in-place so callers never see a new
list object.
"""

import copy
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field


log = logging.getLogger("cai")


DEFAULT_CAP = 500


def _cap() -> int:
    raw = os.environ.get("CAI_HISTORY_CAP")
    if not raw:
        return DEFAULT_CAP
    try:
        v = int(raw)
        return v if v > 0 else DEFAULT_CAP
    except ValueError:
        return DEFAULT_CAP


def _content_hash(messages: list) -> str:
    """Stable hash of the message sequence for cheap dedup between records."""
    try:
        blob = json.dumps(messages, sort_keys=True, default=str)
    except Exception:
        blob = repr(messages)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class HistoryNode:
    id: int
    parent: int | None
    children: list[int] = field(default_factory=list)
    label: str = ""
    meta: dict = field(default_factory=dict)
    snapshot: list = field(default_factory=list)
    content_hash: str = ""
    ts: float = 0.0


class MessageHistoryTracker:
    """Per-session undo tree for the live ``messages`` list."""

    def __init__(self, messages: list) -> None:
        self._bind_messages = messages
        self._nodes: dict[int, HistoryNode] = {}
        self._next_id = 0
        self._head: int = -1
        self._root: int = -1
        self._record(messages, "init", {}, parent=None)
        self._root = self._head

    # ── public hook entrypoint ────────────────────────────────────────────
    def on_event(self, ctx: dict) -> None:
        """Hook handler for the ``messages_mutated`` event."""
        msgs = ctx.get("messages")
        if msgs is None or msgs is not self._bind_messages:
            return
        label = ctx.get("label", "")
        meta = ctx.get("meta") or {}
        self.record(msgs, label, meta)

    # ── recording ─────────────────────────────────────────────────────────
    def record(self, messages: list, label: str, meta: dict | None = None) -> int:
        """Record a new node unless the content is identical to HEAD."""
        h = _content_hash(messages)
        head_node = self._nodes.get(self._head)
        if head_node is not None and head_node.content_hash == h:
            return self._head
        return self._record(messages, label, meta or {}, parent=self._head, h=h)

    def _record(
        self,
        messages: list,
        label: str,
        meta: dict,
        *,
        parent: int | None,
        h: str | None = None,
    ) -> int:
        nid = self._next_id
        self._next_id += 1
        if h is None:
            h = _content_hash(messages)
        node = HistoryNode(
            id=nid,
            parent=parent,
            label=label,
            meta=dict(meta),
            snapshot=copy.deepcopy(messages),
            content_hash=h,
            ts=time.monotonic(),
        )
        self._nodes[nid] = node
        if parent is not None and parent in self._nodes:
            self._nodes[parent].children.append(nid)
        self._head = nid
        self._maybe_evict()
        return nid

    # ── navigation ────────────────────────────────────────────────────────
    def head(self) -> int:
        return self._head

    def node(self, nid: int) -> HistoryNode | None:
        return self._nodes.get(nid)

    def all_nodes(self) -> dict[int, HistoryNode]:
        return self._nodes

    def undo(self) -> bool:
        cur = self._nodes.get(self._head)
        if cur is None or cur.parent is None:
            return False
        self._jump_to(cur.parent)
        return True

    def redo(self) -> bool:
        cur = self._nodes.get(self._head)
        if cur is None or not cur.children:
            return False
        # Most-recently-added child is the "redo" target, matching how
        # branching after an undo pushes a new child to the end of the list.
        self._jump_to(cur.children[-1])
        return True

    def jump(self, nid: int) -> bool:
        if nid not in self._nodes:
            return False
        self._jump_to(nid)
        return True

    def _jump_to(self, nid: int) -> None:
        node = self._nodes[nid]
        self._bind_messages[:] = copy.deepcopy(node.snapshot)
        self._head = nid

    # ── rendering ─────────────────────────────────────────────────────────
    def layout(self) -> tuple[dict, dict, list, int]:
        """2D grid layout of the tree for git-graph-style display.

        Each node gets a ``(row, col)`` coordinate:

        * ``row`` = depth from the root (chronological / undo distance).
        * ``col`` = branch column. The first child of any node inherits
          the parent's col so the original timeline stays vertical;
          subsequent children claim fresh columns from a global counter,
          so a fork created by jumping back and mutating renders as a
          new column to the right of the spine.

        Returns a 4-tuple:

        * ``coords``  — ``{node_id: (row, col)}``
        * ``forks``   — ``{parent_id: [target_col, ...]}`` for parents
          with more than one child; the list holds the cols their newer
          children land in (used to draw ``──.`` fork connectors).
          Single-child parents are absent from this dict.
        * ``ordered`` — pre-order DFS of node ids (display order; used
          for j/k navigation in the overlay).
        * ``max_col`` — highest column index used (so the renderer knows
          how wide the grid is).
        """
        if self._root not in self._nodes:
            return {}, {}, [], 0
        coords: dict[int, tuple[int, int]] = {}
        forks: dict[int, list[int]] = {}
        ordered: list[int] = []
        next_col = [0]

        def visit(nid: int, row: int, col: int) -> None:
            coords[nid] = (row, col)
            ordered.append(nid)
            children = self._nodes[nid].children
            if not children:
                return
            # First child inherits the parent's column; each subsequent
            # child gets the next free column from the global counter.
            child_cols = [col]
            for _ in children[1:]:
                next_col[0] += 1
                child_cols.append(next_col[0])
            if len(children) > 1:
                forks[nid] = child_cols[1:]
            for child_id, child_col in zip(children, child_cols):
                visit(child_id, row + 1, child_col)

        visit(self._root, 0, 0)
        return coords, forks, ordered, next_col[0]

    # ── eviction ──────────────────────────────────────────────────────────
    def _maybe_evict(self) -> None:
        cap = _cap()
        if len(self._nodes) <= cap:
            return
        # Protect the path from root to HEAD (the "current branch spine").
        protected: set[int] = set()
        cur: int | None = self._head
        while cur is not None:
            protected.add(cur)
            cur = self._nodes[cur].parent
        # Evict oldest leaves not on the protected spine until under cap.
        while len(self._nodes) > cap:
            victim = self._oldest_unprotected_leaf(protected)
            if victim is None:
                break  # nothing else to drop
            self._drop(victim)

    def _oldest_unprotected_leaf(self, protected: set[int]) -> int | None:
        candidates = [
            n for n in self._nodes.values()
            if n.id not in protected and not n.children
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda n: n.ts)
        return candidates[0].id

    def _drop(self, nid: int) -> None:
        node = self._nodes.pop(nid, None)
        if node is None:
            return
        if node.parent is not None and node.parent in self._nodes:
            parent = self._nodes[node.parent]
            if nid in parent.children:
                parent.children.remove(nid)
