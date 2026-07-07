"""Live sub-agents viewer: a tree(1)-style hierarchy + auto-refreshing preview.

This is a thin adapter over the generic tree overlay (overlays/tree.py): it maps
an agent node to a row label and a status color, marks the agent you are viewing
from (self) bold, and forwards the preview/kill callbacks the caller supplies.
The data (one node per agent across the registry, finished agents kept) and the
preview/stop logic live in cli.py - this module is pure presentation.

Keys: j/k move, gg/G jump, Ctrl-U/D half-page, /search (n/N to cycle), Enter
attaches to the selected agent read-only (a finished one opens its stored
transcript), Ctrl-K kills the selected running agent, ESC closes.
"""

from ..ansi import (
    SGR_GREEN, SGR_RED, SGR_YELLOW, SGR_DIM_GRAY,
)
from .tree import prompt_tree_overlay

_STATUS_COLOR = {
    'running':  SGR_YELLOW,
    'idle':     SGR_GREEN,
    'done':     SGR_GREEN,
    'finished': SGR_GREEN,
    'failed':   SGR_RED,
    'stopped':  SGR_DIM_GRAY,
}


def _status(node):
    status = node.get('status', '')
    # a node that dropped out of the registry (its process is gone) is kept by
    # the overlay with present=False - show it as finished, never blank.
    if not node.get('present', True):
        return 'finished'
    return status


def _label(node):
    name = node.get('name') or node['id']
    model = node.get('model')
    extra = ''
    if model:
        extra = f"  {model}"
    children = node.get('children_count', 0)
    kids = ''
    if children:
        kids = f"  ({children})"
    return f"{name}{extra}{kids} · {_status(node)}"


def _color(node):
    return _STATUS_COLOR.get(_status(node), '')


def prompt_agents_overlay(screen, fetch_fn, preview_fn, stop_fn=None, self_id=None):
    """live agents tree. fetch_fn() returns the node list (see
    cli._agent_tree_nodes); preview_fn(node, width, max_lines) renders the
    selected agent's conversation; stop_fn(id) interrupts a live agent (Ctrl-K);
    self_id marks the agent being viewed from (drawn bold). returns the selected
    agent id on Enter (the caller attaches to it), or None on cancel."""
    def _is_self(node):
        return self_id is not None and node['id'] == self_id

    return prompt_tree_overlay(
        screen,
        fetch_fn,
        label_fn=_label,
        preview_fn=preview_fn,
        color_fn=_color,
        is_self_fn=_is_self,
        action_fn=stop_fn,
        title="agents",
        hints='  j/k /:search ↵:attach ^K:kill ESC:cancel',
    )
