"""tui: the interactive, full-screen terminal UI.

Wraps the reused screen/ package (an alternate-screen, vim-modal viewport) around
an Agent that is served over a unix socket (UnixWiredAgent) and driven purely as
a wire client through AgentClient - the TUI never touches the agent object once it
is served, so the same client surface will drive a remote agent over --attach. The
main thread owns the terminal and blocks in Screen.prompt(); a worker thread
SUBMITs prompts over the client's streaming connection and renders the EVENT/RESULT
stream into the viewport. Control ops (messages, tools, skills, model, save, load)
go over a separate, lock-guarded control connection.

The status line follows the reference's pattern: the worker only updates raw
signals (busy, the kind of the last delta, when the last token arrived); a
single status thread samples them every _REFRESH_INTERVAL and renders the line,
so a stream that goes quiet falls back to 'waiting' on its own without any event
to trigger it.

The `:`-commands cover the session: :models / :messages / :history / :sessions /
:save / :load, plus :tools and :skills to toggle which tools and skills the agent
uses. Each picker is a screen overlay; the mutating ones (:messages, :history,
:sessions, :load, :tools, :skills) are refused while a run is in flight so they
never race the worker's view of the conversation or the tool registry.
"""
import logging
import os
import queue
import threading
import time

from cai import usage
from cai.channel import connect
from cai.commands import CommandContext
from cai.environment import Environment
from cai.events import EventType
from cai.screen import Screen
from cai.screen.overlays import config as overlay_config
from cai.screen.overlays.config import Setting
from cai.session import SessionsRegistry
from cai.ui import BaseUI
from cai.wire import Wire


log = logging.getLogger("cai")

# jobs-queue sentinel: a continue turn (re-enter the agentic loop with no new user
# message), enqueued after a :history fork that lands on an unfinished point.
_CONTINUE = object()


# command palette (Ctrl-P) and command-mode (:) completion entries.
_PALETTE_COMMANDS = [
    ("models", "switch the model (pin favorites)"),
    ("tools", "enable / disable tools"),
    ("skills", "activate / deactivate skills"),
    ("messages", "view / edit / delete the conversation"),
    ("history", "fork back to an earlier point in this conversation"),
    ("continue", "continue the conversation without a new prompt"),
    ("agents", "live view of sub-agents"),
    ("system", "view / edit the system prompt"),
    ("sessions", "load a saved session"),
    ("config", "edit the live session settings"),
    ("save", "save the session (optional path)"),
    ("load", "load a session file (path)"),
    ("clear", "clear the conversation view"),
    ("quit", "exit the interactive session"),
]
# commands that take an argument: picking them in the palette pre-fills command
# mode (':save ') instead of dispatching immediately.
_PALETTE_ARG_COMMANDS = ("save", "load")

# how often the status thread resamples the signals and repaints the line.
_REFRESH_INTERVAL = 0.1
# a stream is "stalled" once this many seconds pass with no new token; the status
# then falls back from responding/reasoning to waiting.
_STALL_SECONDS = 3.0
# how long a ctx.ui.status(...) note stays on the status line before the normal
# busy/idle readout takes over again.
_NOTE_SECONDS = 4.0
# context-window size used for the ctx % readout when config.json sets no
# 'default_context_size'. matches the reference's fallback.
_DEFAULT_CONTEXT_SIZE = 1_000_000


def _short_args(tool_args):
    """one-line preview of a tool call's arguments for the viewport."""
    if not tool_args:
        return ""
    parts = []
    for key in tool_args:
        text = str(tool_args[key])
        if len(text) > 60:
            text = text[:60] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _status_text(model, state):
    """the left-hand status string: 'model | state'. the screen adds the
    '-- MODE --' prefix and scroll % itself."""
    parts = []
    if model:
        parts.append(model)
    parts.append(state)
    return " | ".join(parts)


class _Transcript:
    """renders agent events into the conversation viewport with uniform
    spacing: exactly one blank line between logical blocks.

    a block is a user prompt, an assistant message, a reasoning trace, a tool
    exchange, or a one-off note. consecutive CONTENT (or REASONING) chunks
    stream into one block, and a tool call chains with its result (and any
    directly following tool exchange) into one tool block - only a change of
    unit starts a new block (block=True). a block's opening chunk is stripped
    of leading newlines (the boundary is owned by the screen, which also
    trims whatever trailing newlines the previous block streamed), so the
    model's own paragraph breaks survive while the gaps between blocks never
    depend on the newlines each event happened to carry."""

    def __init__(self, screen, settings):
        self._screen = screen
        self._settings = settings   # the live env Settings (:config edits it)
        self._streaming = None   # the open unit: "content"/"reasoning"/"tool"/None

    def event(self, event):
        if event.type == EventType.CONTENT:
            self._stream(event.text or "", "content", Screen.LLM)
            return
        if event.type == EventType.REASONING:
            if not self._settings.show_reasoning:
                return
            self._stream(event.text or "", "reasoning", Screen.REASONING)
            return
        if event.type == EventType.USER:
            self.note(f"> {(event.text or '').rstrip()}\n", Screen.USER)
            return
        if event.type == EventType.TOOL_CALL:
            self._stream(f"-> {event.tool_name}({_short_args(event.tool_args)})\n",
                         "tool", Screen.TOOL)
            return
        if event.type == EventType.TOOL_RESULT:
            kind = Screen.TOOL
            if event.is_error:
                kind = Screen.ERROR
            result = event.tool_result or ""
            self._stream(f"<- {event.tool_name}: {len(result)} chars\n", "tool", kind)
            return

    def _stream(self, text, unit, kind):
        # a fresh unit starts a new block; the same unit continues the open
        # one. the opening chunk sheds its leading newlines - a chunk that is
        # nothing but newlines does not open the block at all, the next real
        # chunk does.
        block = unit != self._streaming
        if block:
            text = text.lstrip('\n')
            if not text:
                return
        self._streaming = unit
        self._screen.write(text, kind=kind, block=block)

    def note(self, text, kind):
        """write a self-contained block (a prompt, a tool line, a notice),
        always separated from what precedes it by one blank line."""
        self._streaming = None
        self._screen.write(text, kind=kind, block=True)


class _Status:
    """holds the live status signals and renders the status line.

    the worker updates the signals as the agent streams (busy/idle, the kind of
    the last delta, and when the last token arrived); the status thread calls
    refresh() on a timer. keeping the decision in refresh() - rather than
    painting on each event - is what lets a quiet stream fall back to 'waiting'
    on the clock alone."""

    def __init__(self, screen, model, registry, fallback_limit):
        self._screen = screen
        self._model = model
        # the context-window limit for the ctx % readout: the current model's
        # cached context_length from the registry when known, else this fallback.
        self._registry = registry
        self._fallback_limit = fallback_limit
        self._lock = threading.Lock()
        self._busy = False
        self._phase = None         # None | "tool" | "waiting"
        self._stream_kind = None   # None | "responding" | "reasoning"
        self._last_char = 0.0      # time.monotonic() of the last streamed token
        self._tokens = 0           # exact tokens from the api's usage channel (0 until a sample)
        self._context_limit = fallback_limit
        self._limit_model = None   # the model _context_limit was resolved for
        self._resolve_limit()
        # the latest real usage sample (tokens measured when the conversation
        # held sample_chars chars), surfaced to the :messages overlay so its
        # per-message token math matches the status line.
        self._sample_tokens = 0
        self._sample_chars = 0
        self._last = None          # last (text, right) painted; unchanged is a no-op
        self._note = ""            # transient ctx.ui.status note shown over the state
        self._note_at = 0.0        # time.monotonic() the note was set

    # --- signal updates, called by the worker thread as it streams a run ---

    def busy(self):
        with self._lock:
            self._busy = True
            self._phase = "waiting"
            self._stream_kind = None
            self._last_char = time.monotonic()

    def idle(self):
        with self._lock:
            self._busy = False

    def stream(self, kind):
        with self._lock:
            self._stream_kind = kind
            self._phase = None
            self._last_char = time.monotonic()

    def tool(self):
        with self._lock:
            self._phase = "tool"

    def tool_done(self):
        # tool finished; the model is called again, so we are waiting on it.
        with self._lock:
            self._phase = "waiting"
            self._stream_kind = None

    def set_tokens(self, tokens):
        # the exact token count from the api's usage channel, pushed by the
        # worker when a real sample arrives; we only format it into the readout.
        with self._lock:
            self._tokens = tokens

    def set_note(self, message):
        # a hook's ctx.ui.status(...) reaches here over the wire; show it on the
        # status line for _NOTE_SECONDS, then let the normal readout resume.
        with self._lock:
            self._note = message or ""
            self._note_at = time.monotonic()

    def _resolve_limit(self):
        # cache the current model's context window (cache-only registry read),
        # falling back when it is unknown. caller manages locking.
        limit = None
        if self._registry is not None:
            limit = self._registry.context_length(self._model)
        if limit:
            self._context_limit = int(limit)
        else:
            self._context_limit = self._fallback_limit
        self._limit_model = self._model

    def refresh_limit(self):
        # re-resolve the limit for the current model - called once the catalogue
        # cache has been warmed in the background so a freshly-known
        # context_length is picked up.
        with self._lock:
            self._resolve_limit()

    def set_model(self, model):
        # reflect a model switch (the :models picker) on the status line, and
        # re-resolve its context-window limit.
        with self._lock:
            self._model = model
            self._resolve_limit()

    def set_sample(self, sample_tokens, sample_chars):
        with self._lock:
            self._sample_tokens = sample_tokens
            self._sample_chars = sample_chars

    def ctx_snapshot(self):
        # (context_limit, sample_tokens, sample_chars) for the :messages overlay.
        with self._lock:
            return self._context_limit, self._sample_tokens, self._sample_chars

    # --- render, called by the status thread on a timer ---

    def _state(self, now):
        # one decision ladder over the signals (mirrors the reference's
        # _refresh_status): a stream quiet past _STALL_SECONDS reads as waiting.
        if not self._busy:
            return "idle"
        if self._phase == "tool":
            return "tools"
        if self._stream_kind and now - self._last_char <= _STALL_SECONDS:
            return self._stream_kind
        return "waiting"

    def refresh(self):
        with self._lock:
            if self._limit_model != self._model:
                self._resolve_limit()
            now = time.monotonic()
            if self._note and now - self._note_at < _NOTE_SECONDS:
                text = self._note
            else:
                text = _status_text(self._model, self._state(now))
            right = usage.format_ctx(self._tokens, self._context_limit)
            key = (text, right)
            if key == self._last:
                return
            self._last = key
        # paint outside the lock: set_status takes the screen's own render lock.
        self._screen.set_status(text, right)


class _StatusLoop(threading.Thread):
    """repaints the status line every _REFRESH_INTERVAL until stop is set.

    sampling on a timer (rather than only on events) is what makes a stalled
    stream fall back to 'waiting' - no token arrives to trigger a refresh, so
    this loop is the only thing that notices the silence."""

    def __init__(self, status, stop):
        super().__init__(daemon=True, name="cai-tui-status")
        self._status = status
        self._stop_event = stop

    def run(self):
        while not self._stop_event.wait(_REFRESH_INTERVAL):
            self._status.refresh()


class AgentClient:
    """drives a served agent purely over the wire. a streaming connection carries
    runs (SUBMIT and the EVENT/RESULT stream the worker drains); a separate,
    lock-guarded control connection carries the request/response state ops. the
    TUI talks to the agent only through this, so the very same surface drives a
    local served agent today and a remote one over --attach later."""

    def __init__(self, path):
        self.stream = Wire(connect(path))
        self._ctrl = Wire(connect(path))
        self._ctrl_lock = threading.Lock()

    def submit(self, text):
        self.stream.send_submit(text)

    def continue_run(self):
        """re-enter the agentic loop with no new user turn. a None-text SUBMIT
        runs agent.run(None) on the host - used after a :history fork lands on an
        unfinished point (a user turn, a tool result, or a pending tool call)."""
        self.stream.send_submit(None)

    def steer(self, text):
        self.stream.send_steer(text)

    def interrupt(self):
        self.stream.send_interrupt()

    def recv(self):
        return self.stream.recv()

    def _call(self, op, value=None):
        """one request/response control op, serialized so the worker's token
        polls never interleave with a main-thread overlay op on the one
        control connection. returns the op's value, or None on failure."""
        with self._ctrl_lock:
            ok, result, error = self._ctrl.control(op, value)
        if not ok:
            log.warning("control %r failed: %s", op, error)
            return None
        return result

    def get_info(self):
        return self._call("get_info") or {}

    def get_messages(self):
        return self._call("get_messages") or []

    def set_messages(self, messages):
        self._call("set_messages", messages)

    def get_selected_tools(self):
        return self._call("get_selected_tools") or []

    def get_available_tools(self):
        return self._call("get_available_tools") or []

    def set_selected_tools(self, names):
        self._call("set_selected_tools", names)

    def get_selected_skills(self):
        return self._call("get_selected_skills") or []

    def get_available_skills(self):
        return self._call("get_available_skills") or []

    def set_selected_skills(self, names):
        self._call("set_selected_skills", names)

    def set_model(self, model):
        self._call("set_model", model)

    def set_system_prompt_base(self, base):
        self._call("set_system_prompt_base", base)

    def save(self, path):
        return self._call("save", path)

    def load(self, path):
        with self._ctrl_lock:
            ok, _result, error = self._ctrl.control("load", path)
        if not ok:
            log.warning("control 'load' failed: %s", error)
        return ok

    def close(self):
        try:
            self.stream.channel.close()
        except OSError:
            pass
        try:
            self._ctrl.channel.close()
        except OSError:
            pass


class ScreenUI(BaseUI):
    """the TUI client's UI: renders a served hook's prompts against the Screen.

    one-way ops draw immediately (notify -> a transcript line, status -> the
    status line). the input ops bridge to the main thread via
    Screen.submit_request, which services the request from inside prompt() by
    running the matching overlay and handing back the human's answer; a closed
    screen / no UI yields the headless default."""

    def __init__(self, screen, status, transcript):
        self._screen = screen
        self._status = status
        self._transcript = transcript

    def confirm(self, message, *, default=False, detail=""):
        request = {}
        request["kind"] = "confirm"
        request["title"] = message
        request["body"] = detail
        result = self._screen.submit_request(request)
        if result is None:
            return default
        return bool(result)

    def select(self, message, options, *, default=None, detail=""):
        options = list(options)
        request = {}
        request["kind"] = "select"
        request["title"] = message
        request["options"] = options
        result = self._screen.submit_request(request)
        if result is None:
            return BaseUI.select(self, message, options, default=default, detail=detail)
        return result

    def text(self, message, *, default="", secret=False):
        request = {}
        request["kind"] = "text"
        request["title"] = message
        request["default"] = default
        request["secret"] = secret
        return self._screen.submit_request(request)

    def notify(self, message, *, level="info"):
        kind = Screen.META
        if level == "error":
            kind = Screen.ERROR
        self._transcript.note(f"[{level}] {message}\n", kind)

    def status(self, message):
        self._status.set_note(message)


class _Worker(threading.Thread):
    """drains submitted prompts and streams each run into the screen, all over the
    wire client.

    one job at a time: the queue serialises runs so the agent is never driven
    concurrently. set_busy() brackets each run so the screen and the Ctrl-C
    interrupt gate know a response is in flight; the _Status signals it updates
    drive the status line, painted by the status thread."""

    def __init__(self, client, screen, jobs, stop, status, settings):
        super().__init__(daemon=True, name="cai-tui-worker")
        self._client = client
        self._screen = screen
        self._jobs = jobs
        self._stop_event = stop
        self._status = status
        self._transcript = _Transcript(screen, settings)
        self._ui = ScreenUI(screen, status, self._transcript)
        self._sample_tokens = 0
        self._sample_chars = 0
        self._interrupted = False

    def mark_interrupted(self):
        """flag the in-flight run as Ctrl-C'd so _run_one notes it once the run
        unwinds. set from the key thread; read on the worker thread."""
        self._interrupted = True

    def run(self):
        while not self._stop_event.is_set():
            try:
                text = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if text is None:
                break
            self._run_one(text)

    def _update_status(self, event):
        if event.type == EventType.CONTENT:
            self._status.stream("responding")
        elif event.type == EventType.REASONING:
            self._status.stream("reasoning")
        elif event.type == EventType.TOOL_CALL:
            self._status.tool()
        elif event.type == EventType.TOOL_RESULT:
            self._status.tool_done()
        elif event.type == EventType.USAGE:
            self._on_usage(event)

    def _on_usage(self, event):
        report = event.usage or {}
        tokens = report.get("total_tokens")
        if not tokens:
            tokens = report.get("prompt_tokens", 0) + report.get("completion_tokens", 0)
        if not tokens: return
        # the api's exact count - the official readout. also kept as the
        # calibration sample the :messages overlay uses for its per-message math.
        messages = self._client.get_messages()
        self._status.set_tokens(tokens)
        self._sample_tokens = tokens
        self._sample_chars = usage.message_chars(messages)
        self._status.set_sample(self._sample_tokens, self._sample_chars)

    def _run_one(self, text):
        self._interrupted = False
        self._screen.set_busy(True)
        self._status.busy()
        ui = self._ui
        try:
            if text is _CONTINUE:
                self._client.continue_run()
            else:
                self._client.submit(text)
            result = None
            while result is None:
                messages = self._client.recv()
                if messages is None:
                    break
                for msg in messages:
                    if self._client.stream.answer(msg, ui):
                        continue
                    kind = msg.get("type")
                    if kind == Wire.EVENT:
                        event = Wire.event_from_dict(msg["event"])
                        self._transcript.event(event)
                        self._update_status(event)
                        continue
                    if kind == Wire.RESULT:
                        result = msg.get("text")
                        break
            if result and result.startswith("Error:"):
                self._transcript.note(f"[{result}]\n", Screen.ERROR)
            if self._interrupted:
                self._transcript.note("[interrupted]\n", Screen.META)
        except OSError as e:
            self._transcript.note(f"[error: {e}]\n", Screen.ERROR)
        finally:
            self._screen.set_busy(False)
            self._status.idle()


def _open_messages(screen, client, status):
    """open the :messages overlay over a snapshot of the conversation and write
    back any edits, both over the wire."""
    msgs = client.get_messages()
    if not msgs:
        screen.write("[no messages yet]\n", kind=Screen.META, block=True)
        return
    context_size, prompt_tokens, sample_chars = status.ctx_snapshot()
    edited, _estimate, modified = screen.prompt_messages_overlay(
        msgs,
        context_size=context_size,
        prompt_tokens=prompt_tokens,
        sample_chars=sample_chars)
    if modified:
        client.set_messages(edited)


def _history_node_label(node):
    if node["role"] == "user":
        return "user"
    if node["has_tools"]:
        return "assistant + tools"
    return "assistant"


def _history_node_color(node):
    from cai.screen.ansi import SGR_CYAN, SGR_GREEN

    if node["role"] == "user":
        return SGR_GREEN
    return SGR_CYAN


def _history_node_preview(tree, node, width, max_lines):
    from cai.screen.render import _preview_lines

    convo = []
    for message in tree.prefix_messages(node["id"]):
        if message.get("role") == "system": continue
        convo.append(message)
    head = f"{len(convo)} messages"[:width]
    lines = [head, ""]
    body = max(1, max_lines - len(lines))
    lines.extend(_preview_lines(convo, width, body))
    return lines[:max_lines]


def _open_history(screen, client, status, jobs):
    """open the :history fork view over a snapshot of the conversation. on Enter,
    rewind the live conversation to the chosen node, repaint the viewport to match,
    and - unless the snapshot ends on a final assistant reply - re-enter the
    agentic loop with a continue turn. the tree lives on the screen across opens;
    ingest() drops and rebuilds it when the conversation no longer shares its first
    turn (a clear / load / resume), so stale branches never linger."""
    from cai.history import HistoryTree
    from cai.screen.overlays.tree import _flatten_history

    msgs = client.get_messages()
    if not msgs:
        screen.write("[no history yet]\n", kind=Screen.META, block=True)
        return
    tree = screen._convo_history_tree
    if tree is None:
        tree = HistoryTree()
        screen._convo_history_tree = tree
    tree.ingest(msgs)
    if not tree.nodes():
        screen.write("[no history yet]\n", kind=Screen.META, block=True)
        return

    def _preview(node, width, max_lines):
        return _history_node_preview(tree, node, width, max_lines)

    def _is_head(node):
        return node["is_head"]

    sel = screen.prompt_tree_overlay(
        tree.nodes,
        label_fn=_history_node_label,
        preview_fn=_preview,
        color_fn=_history_node_color,
        is_self_fn=_is_head,
        flatten_fn=_flatten_history,
        title="history",
        hints='  j/k /:search ↵:fork ESC:cancel')
    if sel is None:
        return
    client.set_messages(tree.prefix_messages(sel))
    screen.clear_buffer()
    _replay_messages(screen, client.get_messages())
    if tree.should_continue(sel):
        jobs.put(_CONTINUE)


def _open_models(screen, client, status, registry):
    """open the :models picker and switch the agent (over the wire) to the chosen
    model. the catalogue (and prices) come from the registry, which fetches+caches
    from the provider; pins are persisted by the overlay through the same
    registry."""
    ids = registry.model_ids()
    if not ids:
        screen.write("[no models available]\n", kind=Screen.META, block=True)
        return
    picked = screen.prompt_model_overlay(ids, prices=registry.prices(), favorites=True)
    if picked:
        client.set_model(picked)
        status.set_model(picked)


def _replay_messages(screen, messages):
    """write a loaded conversation into the (cleared) viewport so the user sees
    what they switched to. only user/assistant turns are rendered."""
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            content = str(content)
        if role == "user":
            screen.write(f"> {content}\n", kind=Screen.USER, block=True)
        elif role == "assistant":
            screen.write(content + "\n", kind=Screen.LLM, block=True)


def _session_preview(path, width, max_lines):
    """the picker's right-hand preview: a '<n> messages' header then the
    conversation tail, colored by role. shares the renderer the live :agents
    preview uses (render._preview_lines), so assistant turns whose text lives in
    tool calls / structured content render the same here as there, instead of the
    blank rows a naive content read produces."""
    from cai.screen.ansi import SGR_DIM_GRAY, SGR_RESET
    from cai.screen.render import _preview_lines

    if not path:
        return []
    try:
        payload = SessionsRegistry.read_flow(path)
    except (OSError, ValueError):
        return [f"{SGR_DIM_GRAY}could not read session{SGR_RESET}"]
    convo = []
    for message in (payload.get("messages") or []):
        if message.get("role") == "system": continue
        convo.append(message)
    head = f"{len(convo)} messages"[:width]
    lines = [head, ""]
    body = max(1, max_lines - len(lines))
    lines.extend(_preview_lines(convo, width, body))
    return lines[:max_lines]


def _load_session(screen, client, status, path, cfg):
    """load a .flow into the agent over the wire, then refresh the view: clear the
    viewport, replay the conversation, and follow the restored model. shared by
    :load, the :sessions picker, and the --continue/--sessions resume flags."""
    if not client.load(path):
        screen.write(f"[load error] {path}\n", kind=Screen.ERROR, block=True)
        return
    status.set_model(client.get_info().get("model", ""))
    screen.clear_buffer()
    _replay_messages(screen, client.get_messages())
    _refresh_chips(screen, client, cfg)
    screen.write(f"[loaded {os.path.basename(path)}]\n", kind=Screen.META, block=True)


def _session_tree_nodes():
    """saved sessions as a hierarchy: each .flow declares the ids of the
    sub-agents it launched (their own '<id>.flow' stems), so a child nests under
    the parent that declared it. nodes link by id; a declared child whose own
    flow was deleted becomes a dead-end leaf rather than vanishing. the .flow
    list is read from disk - a client-side file concern."""
    nodes = {}
    declared = []
    for path in SessionsRegistry.list_sessions():
        try:
            payload = SessionsRegistry.read_flow(path)
        except (OSError, ValueError):
            continue
        node_id = os.path.splitext(os.path.basename(path))[0]
        node = {}
        node["id"] = node_id
        node["parent"] = None
        node["name"] = SessionsRegistry.session_label(path)
        node["path"] = path
        node["dead_end"] = False
        nodes[node_id] = node
        for child_id in (payload.get("children") or []):
            declared.append((node_id, child_id))

    for parent_id, child_id in declared:
        child = nodes.get(child_id)
        if child is None:
            leaf = {}
            leaf["id"] = child_id
            leaf["parent"] = parent_id
            leaf["name"] = child_id
            leaf["path"] = None
            leaf["dead_end"] = True
            nodes[child_id] = leaf
            continue
        child["parent"] = parent_id
    return list(nodes.values())


def _open_sessions(screen, client, status, cfg):
    """open the :sessions picker (a tree of sessions and their sub-agent
    sessions) and load the chosen saved session. building the tree reads .flow
    files off disk; the load itself goes over the wire. Ctrl-K deletes the
    highlighted session's .flow; a pruned dead-end has no transcript to delete."""
    from cai.screen.ansi import SGR_DIM_GRAY, SGR_RESET

    if not SessionsRegistry.list_sessions():
        screen.write("[no saved sessions]\n", kind=Screen.META, block=True)
        return

    def _label(node):
        name = node.get("name") or node["id"]
        if node.get("dead_end"):
            return f"{name} · pruned"
        return name

    def _color(node):
        if node.get("dead_end"):
            return SGR_DIM_GRAY
        return ""

    def _preview(node, width, max_lines):
        if node.get("path") is None:
            return [f"{SGR_DIM_GRAY}pruned - no saved transcript{SGR_RESET}"]
        return _session_preview(node["path"], width, max_lines)

    def _delete(nid):
        for node in _session_tree_nodes():
            if node["id"] != nid: continue
            path = node.get("path")
            if path is None: return
            try:
                os.remove(path)
            except OSError:
                pass
            return

    sel = screen.prompt_tree_overlay(
        _session_tree_nodes,
        label_fn=_label,
        preview_fn=_preview,
        color_fn=_color,
        action_fn=_delete,
        title="sessions",
        hints='  j/k /:search ↵:resume ^K:delete ESC:cancel',
    )
    if sel is None:
        return
    for node in _session_tree_nodes():
        if node["id"] != sel: continue
        path = node.get("path")
        if path is None:
            return
        _load_session(screen, client, status, path, cfg)
        return


def _bool_setting(label, obj, attr):
    return Setting(label=label,
                   kind=overlay_config.BOOL,
                   get=lambda: getattr(obj, attr),
                   set=lambda v: setattr(obj, attr, bool(v)))


def _int_setting(label, obj, attr):
    """an integer field. empty input clears to 0; non-numbers report an error
    the overlay shows instead of writing. the overlay's undo replays the native
    int through set(), which str()/int() round-trips cleanly."""
    def write(value):
        text = str(value).strip()
        if text == "":
            setattr(obj, attr, 0)
            return None
        try:
            setattr(obj, attr, int(text))
        except ValueError:
            return "not a number"
        return None
    return Setting(label=label,
                   kind=overlay_config.INT,
                   get=lambda: getattr(obj, attr),
                   set=write)


def _list_setting(label, obj, attr):
    """a list of names edited as a comma-separated string. set() accepts either
    the edited text or a native list (the overlay's undo replays the displayed
    string, which splits back to the same list)."""
    def write(value):
        if isinstance(value, (list, tuple)):
            setattr(obj, attr, list(value))
            return None
        names = []
        for part in str(value).split(","):
            part = part.strip()
            if part:
                names.append(part)
        setattr(obj, attr, names)
        return None
    return Setting(label=label,
                   kind=overlay_config.STRING,
                   get=lambda: ", ".join(getattr(obj, attr) or []),
                   set=write)


def _open_config(screen, client, cfg):
    """the :config overlay - edit the env's live Settings (cai.settings) in
    place. edits apply to this session only; permanent config lives in init.py."""
    settings = []
    settings.append(_bool_setting("show reasoning", cfg, "show_reasoning"))
    settings.append(_bool_setting("show chips", cfg, "show_chips"))
    settings.append(_int_setting("tool result max chars", cfg, "tool_result_max_chars"))
    settings.append(_bool_setting("auto save sessions", cfg, "auto_save_sessions"))
    settings.append(_int_setting("max sessions mb", cfg, "max_sessions_mb"))
    settings.append(_list_setting("skills", cfg, "skills"))
    settings.append(_list_setting("tools", cfg, "tools"))
    screen.prompt_config_overlay(settings)
    _refresh_chips(screen, client, cfg)


def _save_session(screen, client, path, cfg):
    """save the agent (over the wire) to path, or its default <name>.flow when
    path is empty. the written path comes back from the control op."""
    written = client.save(path or None)
    if written is None:
        screen.write("[save error]\n", kind=Screen.ERROR, block=True)
        return
    SessionsRegistry.prune(cfg.max_sessions_mb)
    screen.write(f"[saved {written}]\n", kind=Screen.META, block=True)


def _agent_control(name, op):
    """connect to an agent's socket and run one read-only control op; returns its
    value, or None if the socket is gone/unreachable. the :agents view reads
    everything this way - solely over ~/.config/cai/agents/*.sock."""
    from cai.agents_registry import AgentsRegistry

    try:
        channel = connect(AgentsRegistry.sock_path(name))
    except OSError:
        return None
    wire = Wire(channel)
    try:
        ok, value, _error = wire.control(op)
    except OSError:
        ok, value = False, None
    finally:
        channel.close()
    if not ok:
        return None
    return value


def _open_agents(screen, client):
    """open the live sub-agents tree, built solely from the agents sockets. each
    live agent answers get_info (name, model, the ids of its children); the tree
    is linked from those children lists, and a child with no live socket of its
    own (finished, torn down) still shows as a leaf. preview reads a live agent's
    conversation over its socket; Enter opens it read-only; Ctrl-K interrupts a
    live sub-agent. the TUI's own agent is the root, marked self."""
    from cai.agents_registry import AgentsRegistry
    from cai.screen.ansi import SGR_DIM_GRAY, SGR_RESET
    from cai.screen.render import _preview_lines

    self_name = client.get_info().get("name", "")

    def _nodes():
        nodes = {}
        for name in AgentsRegistry.list_names():
            info = _agent_control(name, "get_info")
            if info is None: continue
            node = {}
            node["id"] = name
            node["parent"] = None
            node["name"] = info.get("name") or name
            node["model"] = info.get("model", "")
            node["children_ids"] = info.get("children") or []
            node["status"] = "running"
            node["live"] = True
            nodes[name] = node
        for node in list(nodes.values()):
            for cid in node["children_ids"]:
                if cid in nodes: continue
                leaf = {}
                leaf["id"] = cid
                leaf["parent"] = None
                leaf["name"] = cid
                leaf["model"] = ""
                leaf["children_ids"] = []
                leaf["status"] = "done"
                leaf["live"] = False
                nodes[cid] = leaf
        for node in nodes.values():
            for cid in node["children_ids"]:
                child = nodes.get(cid)
                if child is None: continue
                child["parent"] = node["id"]
        root = nodes.get(self_name)
        if root is not None:
            root["status"] = "idle"
            if screen._busy:
                root["status"] = "running"
        for node in nodes.values():
            node["children_count"] = len(node["children_ids"])
        return list(nodes.values())

    def _messages_for(node):
        # prefer the live transcript over the socket; once an agent finishes its
        # socket is gone, so fall back to the '<id>.flow' it autosaved (its id is
        # the flow stem). non-system messages only - the leading composed prompt
        # is not worth previewing.
        if node.get("live"):
            messages = _agent_control(node["id"], "get_messages")
            if messages is not None:
                return messages
        try:
            payload = SessionsRegistry.read_flow(SessionsRegistry.session_path(node["id"]))
        except (OSError, ValueError):
            return None
        convo = []
        for message in (payload.get("messages") or []):
            if message.get("role") == "system": continue
            convo.append(message)
        return convo

    def _preview(node, width, max_lines):
        status = node.get("status", "")
        if not node.get("present", True):
            status = "finished"
        lines = [status[:width], ""]
        messages = _messages_for(node)
        if messages is None:
            lines.append(f"{SGR_DIM_GRAY}no transcript{SGR_RESET}")
            return lines[:max_lines]
        body = max(1, max_lines - len(lines))
        lines.extend(_preview_lines(messages, width, body))
        return lines[:max_lines]

    def _stop(name):
        if name == self_name:
            return
        try:
            channel = connect(AgentsRegistry.sock_path(name))
        except OSError:
            return
        try:
            Wire(channel).send_interrupt()
        finally:
            channel.close()

    sel = screen.prompt_agents_overlay(_nodes, _preview, stop_fn=_stop,
                                       self_id=self_name)
    if sel is None:
        return
    chosen = None
    for node in _nodes():
        if node["id"] != sel: continue
        chosen = node
        break
    if chosen is None:
        return
    messages = _messages_for(chosen)
    if messages is None:
        screen.write("[no transcript for this agent]\n", kind=Screen.META, block=True)
        return
    screen.prompt_messages_overlay(messages)


def _chip_lines(skills, tools):
    """the chips widget body: one pill per row, the skills column (pink)
    to the left of the tools column (cyan). pills in a column share one width
    so each column reads as a block; a row whose skills cell is empty paints
    nothing there, so the conversation shows through."""
    from cai.screen.ansi import SGR_PINK_ON_DGRAY, SGR_CYAN_ON_DGRAY, SGR_RESET

    skill_width = 0
    for name in skills:
        skill_width = max(skill_width, len(name))
    tool_width = 0
    for name in tools:
        tool_width = max(tool_width, len(name))

    lines = []
    count = max(len(skills), len(tools))
    for i in range(count):
        parts = []
        if i < len(skills):
            name = skills[i].ljust(skill_width)
            parts.append(f'{SGR_PINK_ON_DGRAY} {name} {SGR_RESET}')
        if tools:
            if i < len(tools):
                name = tools[i].ljust(tool_width)
                parts.append(f'{SGR_CYAN_ON_DGRAY} {name} {SGR_RESET}')
            elif i < len(skills):
                # keep the skill pill in its column: blank out the tools cell
                # so the pill doesn't drift to the right edge.
                parts.append(' ' * (tool_width + 2))
        lines.append(' '.join(parts))
    return lines


def _refresh_chips(screen, client, cfg):
    """rebuild the hover chips widget from the agent's live selection (over
    the wire): the active skills and tools, one pill per row. the show_chips
    setting (:config) turns the widget off entirely."""
    if not cfg.show_chips:
        screen.remove_widget("chips")
        return
    skills = sorted(client.get_selected_skills())
    tools = sorted(client.get_selected_tools())
    lines = _chip_lines(skills, tools)
    if not lines:
        screen.remove_widget("chips")
        return
    screen.add_widget("chips", lines)


def _running_subagents(client):
    """display names of the live sub-agents descended from this session's
    agent, read solely over the agents sockets (like the :agents view):
    every live agent answers get_info, and the descendants are walked
    through the children lists starting from this agent."""
    from cai.agents_registry import AgentsRegistry

    infos = {}
    for name in AgentsRegistry.list_names():
        info = _agent_control(name, "get_info")
        if info is None: continue
        infos[name] = info
    self_name = client.get_info().get("name", "")
    self_info = infos.get(self_name) or {}

    running = []
    pending = list(self_info.get("children") or [])
    seen = set()
    while pending:
        child_id = pending.pop(0)
        if child_id in seen: continue
        seen.add(child_id)
        info = infos.get(child_id)
        if info is None: continue
        running.append(info.get("name") or child_id)
        pending.extend(info.get("children") or [])
    return running


def _agent_chip_lines(names):
    """the agents widget body: one yellow pill per running sub-agent, all
    padded to one width so the column reads as a block."""
    from cai.screen.ansi import SGR_YELLOW_ON_DGRAY, SGR_RESET

    width = 0
    for name in names:
        width = max(width, len(name))
    lines = []
    for name in names:
        text = name.ljust(width)
        lines.append(f'{SGR_YELLOW_ON_DGRAY} {text} {SGR_RESET}')
    return lines


# how often the agents-chips loop re-reads the agents sockets.
_AGENTS_REFRESH = 1.0


class _AgentsChipsLoop(threading.Thread):
    """keeps the 'agents' hover widget in sync with the live sub-agents by
    polling the agents sockets every _AGENTS_REFRESH seconds. polling is the
    only source: a sub-agent starting or finishing is not an event on this
    run's stream. the widget is only touched when the pill lines actually
    change, so an idle session repaints nothing."""

    def __init__(self, screen, client, cfg, stop):
        super().__init__(daemon=True, name="cai-tui-agents-chips")
        self._screen = screen
        self._client = client
        self._cfg = cfg
        self._stop_event = stop
        self._last = []

    def run(self):
        while not self._stop_event.wait(_AGENTS_REFRESH):
            lines = []
            if self._cfg.show_chips:
                try:
                    lines = _agent_chip_lines(_running_subagents(self._client))
                except OSError:
                    continue
            if lines == self._last: continue
            self._last = lines
            if not lines:
                self._screen.remove_widget("agents")
                continue
            self._screen.add_widget("agents", lines)


def _tool_label(name):
    """the origin column for a tool entry: the MCP server name (the part before
    '__'), or 'tool' for an unprefixed function tool."""
    if "__" in name:
        return name.split("__", 1)[0]
    return "tool"


def _open_tools(screen, client, cfg):
    """open the :tools overlay and apply the new selection, all over the wire.
    get_available_tools already unions the catalogue with the agent's own
    registered tools, so it is the full selectable set."""
    active = set(client.get_selected_tools())
    available = set(client.get_available_tools())
    if not available:
        screen.write("[no tools available]\n", kind=Screen.META, block=True)
        return
    entries = []
    for name in sorted(available):
        entries.append((name, _tool_label(name)))
    new_selected = screen.prompt_tools_overlay(entries, active)
    client.set_selected_tools(sorted(new_selected))
    _refresh_chips(screen, client, cfg)


def _open_skills(screen, client, cfg):
    """open the :skills overlay and apply the new selection, all over the wire."""
    available = set(client.get_available_skills())
    active = set(client.get_selected_skills())
    if not available:
        screen.write("[no skills available]\n", kind=Screen.META, block=True)
        return
    new_active = screen.prompt_skills_overlay(sorted(available), active)
    client.set_selected_skills(sorted(new_active))
    _refresh_chips(screen, client, cfg)


def _system_prompt_nodes(composed, base):
    nodes = []

    readonly = {}
    readonly["id"] = "readonly"
    readonly["parent"] = None
    readonly["label"] = "full prompt (read-only)"
    readonly["text"] = composed
    readonly["empty"] = "(no system prompt set)"
    nodes.append(readonly)

    edit = {}
    edit["id"] = "base"
    edit["parent"] = None
    edit["label"] = "base (edit — in memory)"
    edit["text"] = base
    edit["empty"] = "(empty)"
    nodes.append(edit)

    return nodes


def _system_preview(node, width, max_lines):
    from cai.screen.ansi import wrap_ansi, SGR_RESET, SGR_DIM_GRAY

    text = node["text"]
    if not text:
        return [f"{SGR_DIM_GRAY}{node['empty']}{SGR_RESET}"]
    return wrap_ansi(text, width)[:max_lines]


def _open_system(screen, client):
    """`:system` overlay. two options: the full composed prompt (read-only), and
    the agent's system-prompt base, editable in nvim. saving the base pushes it
    to the live agent (which recomposes the prompt with the current skills); the
    change is in memory only and never touches disk."""
    info = client.get_info()
    composed = info.get("system_prompt") or ""
    base = info.get("system_prompt_base") or ""
    nodes = _system_prompt_nodes(composed, base)

    def _fetch():
        return nodes

    def _label(node):
        return node["label"]

    # loop so closing the editor returns to the system view, not the main
    # conversation; only ESC at the overlay exits.
    while True:
        sel = screen.prompt_tree_overlay(
            _fetch,
            label_fn=_label,
            preview_fn=_system_preview,
            title="system prompt",
            hints='  j/k ↵:open ESC:cancel')
        if sel is None:
            return
        if sel == "readonly":
            if not composed:
                screen.write("[no system prompt set]\n", kind=Screen.META, block=True)
                continue
            screen.view_in_editor(composed, suffix='.md')
            continue
        edited = screen.edit_in_editor(base, suffix='.md')
        if edited is None:
            continue
        client.set_system_prompt_base(edited)
        base = edited
        info = client.get_info()
        composed = info.get("system_prompt") or ""
        base = info.get("system_prompt_base") or ""
        nodes = _system_prompt_nodes(composed, base)


def _refresh_tokens(status):
    # resync the ctx readout after a client-side message change (e.g. :clear) the
    # worker is not in the loop for: the official count drops to '?' since no api
    # sample covers the new messages until the next turn produces one.
    status.set_tokens(0)


def _handle_command(screen, client, status, registry, jobs, env, cmd):
    """dispatch a `:`-command. returns True to quit the loop, else False.

    cmd is the raw command string (the text after ':'); the first token is the
    command name. builtin names are handled here; an unrecognized one falls back
    to a command registered through cai.command (env.commands()). the idle
    gates keep conversation-mutating commands off a live run; reads and config
    switches are allowed while busy."""
    if cmd == "q" or cmd == "quit":
        return True
    parts = cmd.split(" ", 1)
    head = parts[0]
    arg = ""
    if len(parts) > 1:
        arg = parts[1].strip()
    if head == "clear":
        if screen._busy:
            screen.write("[busy — :clear when idle]\n", kind=Screen.META, block=True)
            return False
        client.set_messages([])
        screen.clear_buffer()
        _refresh_tokens(status)
        return False
    if head == "save":
        path = ""
        if arg:
            path = os.path.expanduser(arg)
        _save_session(screen, client, path, env.settings)
        return False
    if head == "load":
        if screen._busy:
            screen.write("[busy — :load when idle]\n", kind=Screen.META, block=True)
            return False
        if not arg:
            screen.write("[usage: :load <path>]\n", kind=Screen.META, block=True)
            return False
        _load_session(screen, client, status, os.path.expanduser(arg), env.settings)
        return False
    if head == "models":
        _open_models(screen, client, status, registry)
        return False
    if head == "config":
        _open_config(screen, client, env.settings)
        return False
    if head == "tools":
        if screen._busy:
            screen.write("[busy — open :tools when idle]\n", kind=Screen.META, block=True)
            return False
        _open_tools(screen, client, env.settings)
        return False
    if head == "skills":
        if screen._busy:
            screen.write("[busy — open :skills when idle]\n", kind=Screen.META, block=True)
            return False
        _open_skills(screen, client, env.settings)
        return False
    if head == "messages":
        if screen._busy:
            screen.write("[busy — open :messages when idle]\n", kind=Screen.META, block=True)
            return False
        _open_messages(screen, client, status)
        return False
    if head == "history":
        if screen._busy:
            screen.write("[busy — open :history when idle]\n", kind=Screen.META, block=True)
            return False
        _open_history(screen, client, status, jobs)
        return False
    if head == "continue":
        if screen._busy:
            screen.write("[busy — :continue when idle]\n", kind=Screen.META, block=True)
            return False
        if not client.get_messages():
            screen.write("[nothing to continue]\n", kind=Screen.META, block=True)
            return False
        jobs.put(_CONTINUE)
        return False
    if head == "agents":
        _open_agents(screen, client)
        return False
    if head == "system":
        if screen._busy:
            screen.write("[busy — open :system when idle]\n", kind=Screen.META, block=True)
            return False
        _open_system(screen, client)
        return False
    if head == "sessions":
        if screen._busy:
            screen.write("[busy — open :sessions when idle]\n", kind=Screen.META, block=True)
            return False
        _open_sessions(screen, client, status, env.settings)
        return False
    command = env.commands().get(head)
    if command is not None:
        ctx = CommandContext(arg, client, screen)
        try:
            command.fn(ctx)
        except Exception:
            log.exception("command :%s raised", head)
            screen.write(f"[command :{head} failed]\n", kind=Screen.META, block=True)
        return False
    screen.write(f"[unknown command: :{head}]\n", kind=Screen.META, block=True)
    return False


def run(*,
        model=None,
        system_prompt=None,
        tools=None,
        skills=None,
        reasoning_effort=None,
        temperature=None,
        max_steps=None,
        resume_path=None,
        pick_session=False,
        initial_prompt=None):
    """launch the interactive TUI around a fresh in-process Agent. blocks until
    the user quits (:q, Ctrl-C, or EOF). returns the process exit code.

    resume_path loads that saved session at startup (--continue); pick_session
    opens the :sessions picker at startup (--sessions). either resumes the chosen
    session in place, so autosave writes back to it.

    initial_prompt, when given, is submitted as the first user turn at startup
    (e.g. `cai -i -- hello`) - the TUI comes up, runs it, then takes over input."""
    from cai import config
    from cai.agent import Agent
    from cai.hooks import HookEvent
    from cai.models import ModelsRegistry

    env = Environment.default().load()
    # the cai.settings skills / tools are auto-activated on every CLI run, merged
    # in on top of any --skill / --tool the user passed.
    skills = Environment.merge_activations(skills, env.settings.skills)
    tools = Environment.merge_activations(tools, env.settings.tools)

    # autosave: persist the session to <name>.flow on every conversation
    # mutation, driven solely by hooks - AFTER_RUN (the final answer landed) and
    # MESSAGES_MUTATED (tool results appended mid-run, plus out-of-run edits like
    # a :history fork) together cover every change a run makes (a no-tool turn
    # ends in AFTER_RUN; a tool step appends results and fires MESSAGES_MUTATED).
    # loading a session fires MESSAGES_LOADED, not MESSAGES_MUTATED, so a plain
    # load - whose messages came straight off disk - is not written back and does
    # not bump the file's mtime. the agent comes through ctx.data, and the hook
    # runs on the worker thread mid-run, so it sees a consistent conversation and
    # never races the main thread.
    def _autosave(ctx):
        if not env.settings.auto_save_sessions: return
        saved = (ctx.data or {}).get("agent")

        if saved is None: return
        if not SessionsRegistry.has_real_messages(saved.get_messages()): return

        try:
            saved.save()
            SessionsRegistry.prune(env.settings.max_sessions_mb)
        except OSError:
            pass

    autosave_hooks = []
    autosave_hooks.append((HookEvent.AFTER_RUN, _autosave))
    autosave_hooks.append((HookEvent.MESSAGES_MUTATED, _autosave))

    agent = Agent(model=model,
                  env=env,
                  system_prompt=system_prompt,
                  tools=tools or [],
                  skills=skills or [],
                  hooks=autosave_hooks,
                  reasoning_effort=reasoning_effort,
                  temperature=temperature,
                  max_steps=max_steps,
                  tool_result_max_chars=env.settings.tool_result_max_chars)

    from cai.wired_agent import UnixWiredAgent
    server = UnixWiredAgent(agent)
    threading.Thread(target=server.serve, daemon=True, name="cai-tui-serve").start()
    client = AgentClient(server.path)

    screen = Screen()
    jobs = queue.Queue()
    stop = threading.Event()

    palette = list(_PALETTE_COMMANDS)
    commands = env.commands()
    for name in sorted(commands):
        palette.append((name, commands[name].help))
    completions = {}
    for name, _help in palette:
        completions[name] = []
    screen.set_palette_commands(palette, _PALETTE_ARG_COMMANDS)
    screen.set_cmd_completions(completions)

    from cai.api import OpenAiApi
    cfg = config.load_config()
    fallback_limit = int(config.load_optional("default_context_size", _DEFAULT_CONTEXT_SIZE))
    registry = ModelsRegistry(OpenAiApi(cfg.base_url,
                                        config.load_api_key(),
                                        ssl_verify=config.load_optional("ssl_verify", True)))
    model = client.get_info().get("model", "")
    status = _Status(screen, model, registry, fallback_limit)
    status.refresh()   # initial idle paint, before the status thread takes over

    # warm the catalogue cache in the background so the current model's
    # context_length is saved and the ctx % can derive from it - without
    # blocking startup or the first prompt. models() respects the daily cache,
    # so this is a network call at most once a day.
    def _warm_models():
        registry.models()
        status.refresh_limit()

    threading.Thread(target=_warm_models, daemon=True, name="cai-tui-models-warm").start()

    def _on_interrupt():
        # Ctrl-C on an empty prompt: cancel an in-flight run if there is one,
        # and tell the screen we consumed the interrupt so it skips its quit
        # double-tap. otherwise let the default handling run.
        if screen._busy:
            worker.mark_interrupted()
            client.interrupt()
            return True
        return False

    screen.set_interrupt_handler(_on_interrupt)

    def _on_recall():
        # Ctrl-X: take back the last user prompt (e.g. one just Ctrl-C'd before
        # the model answered). only when idle and the conversation actually ends
        # on a plain user message - an interrupted turn that left a partial
        # assistant reply, or a mid-run conversation, is left untouched. returns
        # the text for the screen to drop into the input box; None does nothing.
        if screen._busy:
            return None
        msgs = client.get_messages()
        if not msgs:
            return None
        last = msgs[-1]
        if last.get("role") != "user":
            return None
        content = last.get("content")
        if not isinstance(content, str):
            return None
        client.set_messages(msgs[:-1])
        return content

    screen.set_recall_handler(_on_recall)

    worker = _Worker(client, screen, jobs, stop, status, env.settings)
    worker.start()
    status_loop = _StatusLoop(status, stop)
    status_loop.start()
    agents_chips = _AgentsChipsLoop(screen, client, env.settings, stop)
    agents_chips.start()

    screen.write(f"cai — model {model}. type to chat; :q to quit.\n",
                 kind=Screen.META, block=True)
    _refresh_chips(screen, client, env.settings)

    # resume at startup: --continue loads the resolved session directly;
    # --sessions opens the picker. nothing is running yet, so no idle gate.
    if resume_path:
        _load_session(screen, client, status, resume_path, env.settings)
    elif pick_session:
        _open_sessions(screen, client, status, env.settings)

    # a prompt passed on the command line with -i: hand it to the worker as the
    # first turn. the worker echoes the USER event and streams the answer just
    # as if the user had typed it, then the input loop below takes over.
    if initial_prompt and initial_prompt.strip():
        jobs.put(initial_prompt)

    try:
        while not stop.is_set():
            user_input = screen.prompt("> ")
            if user_input is None:
                continue
            if screen._command_result is not None:
                cmd = screen._command_result
                screen._command_result = None
                if _handle_command(screen, client, status, registry, jobs, env, cmd):
                    break
                continue
            # '!text' steers the in-flight run; with nothing running it is just
            # an ordinary prompt with the bang stripped.
            if user_input.startswith("!"):
                steer_text = user_input[1:]
                if screen._busy:
                    client.steer(steer_text)
                    continue
                user_input = steer_text
            if not user_input.strip():
                continue
            jobs.put(user_input)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop.set()
        jobs.put(None)
        server.close()
        worker.join(timeout=2)
        status_loop.join(timeout=2)
        agents_chips.join(timeout=2)
        client.close()
        screen.close()
        agent.close()
    return 0
