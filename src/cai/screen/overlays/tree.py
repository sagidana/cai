"""Generic tree overlay: a tree(1)-style hierarchy on the left, a live preview
on the right. Reusable chrome for any parent/child data - the :agents view is
the first caller, future hierarchical pickers can reuse it as-is.

The drawing is shared with the model/session picker (model._draw_model_overlay
renders the list+preview box); this module only flattens a node forest into
tree rows with ├──/└── connectors, keeps the structure stable across refreshes,
and drives the modal navigation (j/k, gg/G, ^U/^D, /search, Enter, an optional
action key, ESC).

A node is a dict the caller owns; it MUST carry 'id' (unique, hashable) and
'parent' (the id of its parent, or None for a root). Everything else - the row
label, its color, the preview - the caller supplies through callbacks, so this
module stays ignorant of what a node means. fetch_fn() is re-called every
~0.5s; a node that drops out of the fetch is KEPT (marked node['present'] =
False) so the tree never loses an entry while the view is open - the only thing
that moves between refreshes is the cursor.
"""

import select
import shutil
import signal
import sys
import termios
import time
import tty

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_HIDE,
    ERASE_SCREEN,
    SGR_BOLD_AZURE,
    KEY_ESC, KEY_ENTER, KEY_CTRL_C,
    KEY_CTRL_D, KEY_CTRL_K, KEY_CTRL_U, KEY_UP, KEY_DOWN,
)
from ..input import read_key, parse_mouse
from .model import _draw_model_overlay, _filter_and_sort, overlay_click_index
from .tools import _find_matches, _handle_search_key

_REFRESH = 0.5   # seconds between data refreshes while the view is open


def _flatten_tree(nodes):
    """nodes in insertion order -> [(node, prefix), ...] in tree(1) order: each
    root, then its descendants depth-first, with ├──/└──/│ connectors. sibling
    order is the insertion order. a node whose parent id is absent is treated as
    a root, so an orphaned subtree still shows."""
    by_id = {}
    order = []
    for node in nodes:
        by_id[node['id']] = node
        order.append(node['id'])

    children = {}
    roots = []
    for nid in order:
        parent = by_id[nid].get('parent')
        if parent is not None and parent in by_id:
            children.setdefault(parent, []).append(nid)
            continue
        roots.append(nid)

    rows = []

    def walk(nid, child_prefix, connector):
        rows.append((by_id[nid], connector))
        kids = children.get(nid, [])
        for i, kid in enumerate(kids):
            last = i == len(kids) - 1
            branch = '├── '
            cont = '│   '
            if last:
                branch = '└── '
                cont = '    '
            walk(kid, child_prefix + cont, child_prefix + branch)

    for nid in roots:
        walk(nid, '', '')
    return rows


def _flatten_history(nodes):
    """like _flatten_tree, but a node's FIRST child continues the trunk inline
    at the same indentation (a linear conversation renders flat), and only the
    extra children - the forks - branch off with ├──/└── connectors. the forks
    are drawn before the trunk continues, so a fork reads as a side branch off
    the main line."""
    by_id = {}
    order = []
    for node in nodes:
        by_id[node['id']] = node
        order.append(node['id'])

    children = {}
    roots = []
    for nid in order:
        parent = by_id[nid].get('parent')
        if parent is not None and parent in by_id:
            children.setdefault(parent, []).append(nid)
            continue
        roots.append(nid)

    rows = []

    def walk(nid, connector, cont):
        rows.append((by_id[nid], connector))
        kids = children.get(nid, [])
        if not kids:
            return
        trunk = kids[0]
        forks = kids[1:]
        for i, kid in enumerate(forks):
            last = i == len(forks) - 1
            branch = cont + '├── '
            child_cont = cont + '│   '
            if last:
                branch = cont + '└── '
                child_cont = cont + '    '
            walk(kid, branch, child_cont)
        walk(trunk, cont, cont)

    for nid in roots:
        walk(nid, '', '')
    return rows


def prompt_tree_overlay(screen,
                        fetch_fn,
                        *,
                        label_fn,
                        preview_fn,
                        color_fn=None,
                        is_self_fn=None,
                        action_fn=None,
                        flatten_fn=_flatten_tree,
                        title="tree",
                        hints='  j/k /:search ↵:open ESC:cancel'):
    """live tree picker. fetch_fn() -> list of node dicts (each with 'id' and
    'parent'); it is re-called every ~0.5s. label_fn(node) -> row text (no
    connectors); preview_fn(node, width, max_lines) -> preview lines; color_fn
    (node) -> an SGR color for the row (optional); is_self_fn(node) -> True for
    the 'you are here' node, drawn bold/azure (optional); action_fn(id) runs on
    Ctrl-K (optional). returns the selected node id on Enter, or None on cancel.
    a node that disappears from a later fetch is retained with node['present'] =
    False so the tree stays stable - render it accordingly in your callbacks.
    flatten_fn picks the row layout: the default nests every child; _flatten_history
    keeps the first child on the trunk so a linear chain renders flat."""
    nodes_by_id = {}
    order = []

    def _merge(fetched):
        present = set()
        for node in (fetched or []):
            nid = node['id']
            present.add(nid)
            node['present'] = True
            if nid not in nodes_by_id:
                order.append(nid)
            nodes_by_id[nid] = node
        for nid in order:
            if nid in present: continue
            nodes_by_id[nid]['present'] = False

    _merge(fetch_fn())
    if not nodes_by_id:
        return None

    flat = []          # [(node, connector), ...] in tree order, rebuilt each refresh
    labels = []        # display strings parallel to flat (connector + label)
    filtered = []
    selected_idx = 0
    prev_lines = {}
    first_draw = [True]
    resize_pending = [False]

    search_mode = False
    search_direction = 1
    search_pattern = ''
    search_buf = []
    search_matches = []
    search_match_idx = -1
    pre_search_idx = 0
    prev_key = ''

    def _rebuild_rows():
        nonlocal flat, labels, filtered
        ordered = []
        for nid in order:
            ordered.append(nodes_by_id[nid])
        flat = flatten_fn(ordered)
        labels = []
        for node, connector in flat:
            labels.append(f"{connector}{label_fn(node)}")
        filtered = _filter_and_sort(labels, '')

    def _selected_id():
        if flat and 0 <= selected_idx < len(flat):
            return flat[selected_idx][0]['id']
        return None

    def _row_colors():
        colors = []
        for node, _connector in flat:
            color = ''
            if color_fn is not None:
                color = color_fn(node) or ''
            if is_self_fn is not None and is_self_fn(node):
                color = SGR_BOLD_AZURE
            colors.append(color)
        return colors

    def preview_render(_label, width, max_lines):
        if not (flat and 0 <= selected_idx < len(flat)):
            return []
        return preview_fn(flat[selected_idx][0], width, max_lines)

    def _refresh_data(purge=False):
        nonlocal selected_idx, search_matches, search_match_idx
        keep_id = _selected_id()
        if purge:
            order.clear()
            nodes_by_id.clear()
        _merge(fetch_fn())
        _rebuild_rows()
        if keep_id is not None:
            for i, (node, _connector) in enumerate(flat):
                if node['id'] != keep_id: continue
                selected_idx = i
                break
        selected_idx = min(selected_idx, max(0, len(filtered) - 1))
        if search_pattern:
            search_matches = _find_matches(filtered, search_pattern)
            search_match_idx = min(search_match_idx, len(search_matches) - 1)

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        resize_pending[0] = True
        first_draw[0] = True

    def _redraw():
        _draw_model_overlay(
            screen._rows, screen._cols,
            filtered, selected_idx, search_buf,
            len(filtered), prev_lines, first_draw[0], title,
            preview_render,
            True, search_mode, search_matches, search_match_idx,
            search_direction, search_pattern,
            hints,
            _row_colors(),
        )
        first_draw[0] = False

    _rebuild_rows()

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()

    result = None
    try:
        signal.signal(signal.SIGWINCH, _on_resize)
        tty.setraw(screen._tty_fd)
        _redraw()
        last_refresh = time.monotonic()

        while True:
            if resize_pending[0]:
                resize_pending[0] = False
                _redraw()

            now = time.monotonic()
            if now - last_refresh >= _REFRESH:
                last_refresh = now
                _refresh_data()
                _redraw()

            rlist, _, _ = select.select([screen._tty_fd], [], [], 0.05)
            if not rlist:
                continue
            key = read_key(screen._tty_fd)

            mouse = parse_mouse(key)
            if mouse is not None:
                action, _button, mcol, mrow = mouse
                n = len(filtered)
                if action == 'wheel_up':
                    selected_idx = max(0, selected_idx - 1)
                elif action == 'wheel_down':
                    selected_idx = min(max(0, n - 1), selected_idx + 1)
                elif action == 'press':
                    idx = overlay_click_index(screen._rows, screen._cols, n,
                                              selected_idx, mrow, mcol)
                    if idx is not None:
                        selected_idx = idx
                _redraw()
                continue

            if search_mode:
                search_mode, selected_idx, search_pattern, search_buf, \
                    search_matches, search_match_idx = _handle_search_key(
                        key, filtered, search_mode, selected_idx,
                        search_pattern, search_buf, search_matches,
                        search_match_idx, pre_search_idx, search_direction,
                    )
                _redraw()
                continue

            if key == KEY_ESC or key == KEY_CTRL_C:
                break

            if key in KEY_ENTER:
                result = _selected_id()
                break

            if key == KEY_CTRL_K:
                if action_fn is not None:
                    nid = _selected_id()
                    if nid is not None:
                        action_fn(nid)
                        _refresh_data(purge=True)
                _redraw()
                continue

            n = len(filtered)
            if key in (KEY_UP, 'k'):
                selected_idx = max(0, selected_idx - 1)
            elif key in (KEY_DOWN, 'j'):
                selected_idx = min(max(0, n - 1), selected_idx + 1)
            elif key in ('/', '?'):
                search_mode = True
                search_direction = 1
                if key == '?':
                    search_direction = -1
                search_buf = []
                search_pattern = ''
                search_matches = []
                search_match_idx = -1
                pre_search_idx = selected_idx
            elif key == 'n' and search_matches:
                search_match_idx = (search_match_idx + search_direction) % len(search_matches)
                selected_idx = search_matches[search_match_idx]
            elif key == 'N' and search_matches:
                search_match_idx = (search_match_idx - search_direction) % len(search_matches)
                selected_idx = search_matches[search_match_idx]
            elif key == 'G':
                selected_idx = max(0, n - 1)
            elif key == 'g' and prev_key == 'g':
                selected_idx = 0
            elif key == KEY_CTRL_U:
                overhead = 4
                vis = max(5, int(screen._rows * 0.95)) - overhead
                selected_idx = max(0, selected_idx - max(1, vis // 2))
            elif key == KEY_CTRL_D:
                overhead = 4
                vis = max(5, int(screen._rows * 0.95)) - overhead
                selected_idx = min(max(0, n - 1), selected_idx + max(1, vis // 2))
            prev_key = key
            _redraw()
            continue

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return result
