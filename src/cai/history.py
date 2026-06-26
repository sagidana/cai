import json


# the in-memory, branching record of one interactive conversation, built from
# the live message list. nodes are returnable points: one per user message and
# one per assistant round (an assistant message plus its trailing tool results).
# re-ingesting a forked-and-regrown conversation keeps the old branch and hangs
# the divergent tail under the fork point, so the tree mirrors what the agent
# explored. it lives only for the current TUI instance.


def _msg_key(msg):
    data = {}
    data["role"] = msg.get("role")
    data["content"] = msg.get("content")
    data["tool_calls"] = msg.get("tool_calls")
    data["tool_call_id"] = msg.get("tool_call_id")
    return json.dumps(data, sort_keys=True, default=str)


def _span_key(span):
    keys = []
    for msg in span:
        keys.append(_msg_key(msg))
    return tuple(keys)


def _segment(messages):
    """split a message list into per-turn spans. each span is
    [role, has_tools, msgs]: a user message, or an assistant message and the
    tool results that follow it. the synthetic user message that carries tool
    images back to the model (it follows a tool result mid-round) stays inside
    the assistant round rather than starting a new turn."""
    spans = []
    current = None
    prev_role = None
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            prev_role = role
            continue
        if role == "user" and prev_role == "tool":
            if current is not None:
                current[2].append(msg)
            prev_role = role
            continue
        if role == "user":
            current = ["user", False, [msg]]
            spans.append(current)
            prev_role = role
            continue
        if role == "assistant":
            has_tools = bool(msg.get("tool_calls"))
            current = ["assistant", has_tools, [msg]]
            spans.append(current)
            prev_role = role
            continue
        if current is not None:
            current[2].append(msg)
        prev_role = role
    return spans


class HistoryNode:
    def __init__(self, id, parent, role, has_tools, span, key):
        self.id = id
        self.parent = parent
        self.role = role
        self.has_tools = has_tools
        self.span = span
        self.key = key
        self.children = []


class HistoryTree:
    def __init__(self):
        self._nodes = {}
        self._order = []
        self._root_id = None
        self._head_id = None
        self._counter = 0

    def _reset(self):
        self._nodes = {}
        self._order = []
        self._root_id = None
        self._head_id = None

    def _add_node(self, parent_id, role, has_tools, span, key):
        nid = self._counter
        self._counter += 1
        node = HistoryNode(nid, parent_id, role, has_tools, span, key)
        self._nodes[nid] = node
        self._order.append(nid)
        if parent_id is None:
            self._root_id = nid
        else:
            self._nodes[parent_id].children.append(nid)
        return nid

    def _find_child(self, parent_id, key):
        candidates = []
        if parent_id is None:
            if self._root_id is not None:
                candidates = [self._root_id]
        else:
            candidates = self._nodes[parent_id].children
        for cid in candidates:
            if self._nodes[cid].key == key:
                return cid
        return None

    def ingest(self, messages):
        """fold the current message list into the tree. a conversation that no
        longer shares its first turn (cleared, loaded, or a resumed session)
        drops the old tree and starts fresh, so stale branches never linger."""
        spans = _segment(messages)
        if not spans:
            self._reset()
            return
        first_key = _span_key(spans[0][2])
        if self._root_id is None or self._nodes[self._root_id].key != first_key:
            self._reset()
        parent_id = None
        last_id = None
        for role, has_tools, span in spans:
            key = _span_key(span)
            match = self._find_child(parent_id, key)
            if match is None:
                match = self._add_node(parent_id, role, has_tools, span, key)
            parent_id = match
            last_id = match
        self._head_id = last_id

    def nodes(self):
        out = []
        for nid in self._order:
            node = self._nodes[nid]
            entry = {}
            entry["id"] = node.id
            entry["parent"] = node.parent
            entry["role"] = node.role
            entry["has_tools"] = node.has_tools
            entry["is_head"] = (node.id == self._head_id)
            out.append(entry)
        return out

    def prefix_messages(self, node_id):
        """the full message list from the root down to node_id - the snapshot
        the conversation forks back to."""
        chain = []
        nid = node_id
        while nid is not None:
            chain.append(nid)
            nid = self._nodes[nid].parent
        chain.reverse()
        messages = []
        for nid in chain:
            messages.extend(self._nodes[nid].span)
        return messages

    def should_continue(self, node_id):
        """whether forking to node_id should re-enter the agentic loop. it does
        unless the snapshot ends on a final assistant reply (one with no pending
        tool calls) - i.e. user prompts, completed tool results, and pending
        tool calls all resume the loop."""
        prefix = self.prefix_messages(node_id)
        if not prefix:
            return False
        last = prefix[-1]
        role = last.get("role")
        if role == "user":
            return True
        if role == "tool":
            return True
        if role == "assistant":
            if last.get("tool_calls"):
                return True
            return False
        return True
