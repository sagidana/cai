"""Unit tests for the in-memory conversation-history tree behind the :history
view: segmentation into per-turn nodes, branch-on-fork ingest, the
continue-vs-wait rule, and prefix reconstruction."""

from cai import history
from cai.history import HistoryTree
from cai.screen.overlays.tree import _flatten_history


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text, tool_calls=None):
    msg = {"role": "assistant", "content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def _call(call_id, name):
    return {"id": call_id, "function": {"name": name, "arguments": "{}"}}


def _linear_convo():
    return [
        _user("hi"),
        _assistant("", [_call("c1", "ls")]),
        _tool("c1", "files"),
        _assistant("", [_call("c2", "cat")]),
        _tool("c2", "body"),
        _assistant("done"),
        _user("again"),
        _assistant("", [_call("c3", "ls")]),
        _tool("c3", "files"),
        _assistant("ok"),
    ]


def test_segment_groups_user_and_assistant_rounds():
    spans = history._segment(_linear_convo())
    shape = []
    for role, has_tools, msgs in spans:
        shape.append((role, has_tools, len(msgs)))
    assert shape == [
        ("user", False, 1),
        ("assistant", True, 2),
        ("assistant", True, 2),
        ("assistant", False, 1),
        ("user", False, 1),
        ("assistant", True, 2),
        ("assistant", False, 1),
    ]


def test_segment_skips_system_and_keeps_tool_image_in_round():
    convo = [
        {"role": "system", "content": "sys"},
        _user("hi"),
        _assistant("", [_call("c1", "shot")]),
        _tool("c1", "ok"),
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
        _assistant("done"),
    ]
    spans = history._segment(convo)
    shape = []
    for role, has_tools, msgs in spans:
        shape.append((role, len(msgs)))
    # the image-bearing user message stays inside the assistant round, so there
    # are three turns (user, assistant+tool+image, assistant), not four.
    assert shape == [("user", 1), ("assistant", 3), ("assistant", 1)]


def test_linear_ingest_one_node_per_turn():
    tree = HistoryTree()
    tree.ingest(_linear_convo())
    assert len(tree.nodes()) == 7
    last = tree.nodes()[-1]
    assert last["is_head"]
    assert last["role"] == "assistant"
    assert not last["has_tools"]


def test_node_roles_and_tools():
    tree = HistoryTree()
    tree.ingest(_linear_convo())
    nodes = tree.nodes()
    summary = []
    for node in nodes:
        summary.append((node["role"], node["has_tools"]))
    assert summary == [
        ("user", False),
        ("assistant", True),
        ("assistant", True),
        ("assistant", False),
        ("user", False),
        ("assistant", True),
        ("assistant", False),
    ]


def test_fork_creates_a_branch_and_retains_old_tail():
    tree = HistoryTree()
    tree.ingest(_linear_convo())
    before = len(tree.nodes())
    # fork back to the 4th node (the first "done" assistant) and regrow a
    # different second user turn.
    fork_node = tree.nodes()[3]
    prefix = tree.prefix_messages(fork_node["id"])
    regrown = prefix + [_user("different"), _assistant("new answer")]
    tree.ingest(regrown)
    nodes = tree.nodes()
    # old branch (3 nodes after the fork point) is retained; two new nodes are
    # added under the fork point.
    assert len(nodes) == before + 2
    # the fork node now has two children.
    children = []
    for node in nodes:
        if node["parent"] == fork_node["id"]:
            children.append(node)
    assert len(children) == 2
    # head moved to the new tail.
    head = None
    for node in nodes:
        if node["is_head"]:
            head = node
    assert head["role"] == "assistant"
    assert tree.prefix_messages(head["id"])[-1]["content"] == "new answer"


def test_ingest_resets_when_first_turn_changes():
    tree = HistoryTree()
    tree.ingest(_linear_convo())
    tree.ingest([_user("brand new"), _assistant("hello")])
    nodes = tree.nodes()
    assert len(nodes) == 2
    assert nodes[0]["role"] == "user"
    assert tree.prefix_messages(nodes[0]["id"])[0]["content"] == "brand new"


def test_empty_conversation_clears_tree():
    tree = HistoryTree()
    tree.ingest(_linear_convo())
    tree.ingest([])
    assert tree.nodes() == []


def test_prefix_messages_is_root_to_node():
    tree = HistoryTree()
    tree.ingest(_linear_convo())
    third = tree.nodes()[2]
    prefix = tree.prefix_messages(third["id"])
    # user + two assistant rounds (each assistant + its tool result) = 5 msgs.
    assert len(prefix) == 5
    assert prefix[0]["role"] == "user"
    assert prefix[-1]["role"] == "tool"


def test_should_continue_truth_table():
    tree = HistoryTree()
    tree.ingest(_linear_convo())
    nodes = tree.nodes()
    # node 0: ends on user prompt -> continue
    assert tree.should_continue(nodes[0]["id"])
    # node 1: ends on a completed tool result -> continue
    assert tree.should_continue(nodes[1]["id"])
    # node 3: ends on a final assistant reply -> wait
    assert not tree.should_continue(nodes[3]["id"])


def _connectors(tree):
    out = []
    for node, conn in _flatten_history(tree.nodes()):
        out.append(conn)
    return out


def test_flatten_is_flat_without_forks():
    tree = HistoryTree()
    tree.ingest(_linear_convo())
    # a linear conversation renders with no connectors at all.
    assert _connectors(tree) == ["", "", "", "", "", "", ""]


def test_flatten_nests_only_the_fork_branch():
    tree = HistoryTree()
    tree.ingest(_linear_convo())
    fork_node = tree.nodes()[3]
    prefix = tree.prefix_messages(fork_node["id"])
    tree.ingest(prefix + [_user("different"), _assistant("new answer")])
    rows = _flatten_history(tree.nodes())
    forked = []
    trunk = []
    for node, conn in rows:
        if conn:
            forked.append(node)
            continue
        trunk.append(node)
    # exactly the two regrown nodes are nested under the fork; everything else
    # stays on the flat trunk.
    assert len(forked) == 2
    assert len(trunk) == len(tree.nodes()) - 2
    # the fork point carries one branch glyph; its linear continuation is
    # indented under it without a second glyph.
    glyphed = []
    for _node, conn in rows:
        if "── " in conn:
            glyphed.append(conn)
    assert len(glyphed) == 1


def test_should_continue_pending_tool_call():
    tree = HistoryTree()
    convo = [_user("hi"), _assistant("", [_call("c1", "ls")])]
    tree.ingest(convo)
    head = tree.nodes()[-1]
    # an assistant message with tool_calls but no results yet -> continue.
    assert tree.should_continue(head["id"])
