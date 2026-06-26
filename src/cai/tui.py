"""tui: the interactive, full-screen terminal UI.

Wraps the reused screen/ package (an alternate-screen, vim-modal viewport) around
a single in-process Agent. The main thread owns the terminal and blocks in
Screen.prompt(); a worker thread drains submitted prompts off a queue, runs the
agent, and streams its events into the viewport. This is the in-process analogue
of the socket attach loop the reference frontend uses - there is no wire here,
the worker simply iterates the agent's RunHandle.

The status line follows the reference's pattern: the worker only updates raw
signals (busy, the kind of the last delta, when the last token arrived); a
single status thread samples them every _REFRESH_INTERVAL and renders the line,
so a stream that goes quiet falls back to 'waiting' on its own without any event
to trigger it.

The `:`-commands cover the session: :models / :messages / :sessions / :save /
:load, plus :tools and :skills to toggle which tools and skills the agent uses.
Each picker is a screen overlay; the mutating ones (:messages, :sessions, :load,
:tools, :skills) are refused while a run is in flight so they never race the
worker's view of the conversation or the tool registry.
"""
import os
import queue
import threading
import time

from cai import usage
from cai.events import EventType
from cai.screen import Screen
from cai.session import SessionsRegistry


# command palette (Ctrl-P) and command-mode (:) completion entries.
_PALETTE_COMMANDS = [
    ("models", "switch the model (pin favorites)"),
    ("tools", "enable / disable tools"),
    ("skills", "activate / deactivate skills"),
    ("messages", "view / edit / delete the conversation"),
    ("sessions", "load a saved session"),
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


def _write_event(screen, event):
    """render one agent event into the conversation viewport, by kind."""
    if event.type == EventType.CONTENT:
        screen.write(event.text or "", kind=Screen.LLM)
        return
    if event.type == EventType.REASONING:
        screen.write(event.text or "", kind=Screen.REASONING)
        return
    if event.type == EventType.TOOL_CALL:
        line = f"\n-> {event.tool_name}({_short_args(event.tool_args)})\n"
        screen.write(line, kind=Screen.TOOL)
        return
    if event.type == EventType.TOOL_RESULT:
        kind = Screen.TOOL
        if event.is_error:
            kind = Screen.ERROR
        result = event.tool_result or ""
        screen.write(f"<- {event.tool_name}: {len(result)} chars\n", kind=kind)
        return


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
        self._tokens = 0           # estimated tokens in the conversation so far
        self._context_limit = fallback_limit
        self._limit_model = None   # the model _context_limit was resolved for
        self._resolve_limit()
        # the latest real usage sample (tokens measured when the conversation
        # held sample_chars chars), surfaced to the :messages overlay so its
        # per-message token math matches the status line.
        self._sample_tokens = 0
        self._sample_chars = 0
        self._last = None          # last (text, right) painted; unchanged is a no-op

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
        # the conversation's estimated token count, computed by the worker (it
        # owns the agent's messages); we only format it into the ctx readout.
        with self._lock:
            self._tokens = tokens

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
            text = _status_text(self._model, self._state(time.monotonic()))
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


class _Worker(threading.Thread):
    """drains submitted prompts and streams each agent run into the screen.

    one job at a time: the queue serialises runs so the agent is never driven
    concurrently. set_busy() brackets each run so the screen and the Ctrl-C
    interrupt gate know a response is in flight; the _Status signals it updates
    drive the status line, painted by the status thread."""

    def __init__(self, agent, screen, jobs, stop, status):
        super().__init__(daemon=True, name="cai-tui-worker")
        self._agent = agent
        self._screen = screen
        self._jobs = jobs
        # NB: not self._stop - that shadows threading.Thread._stop, which join()
        # calls internally (it would then try to call this Event and crash).
        self._stop_event = stop
        self._status = status
        # the latest real usage measurement, paired with the message size at
        # which it was taken, so between-turn estimates scale from a true count.
        self._sample_tokens = 0
        self._sample_chars = 0

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
        # pair the turn's real token count with the conversation's current char
        # size, so the between-turn estimate scales from a true measurement.
        report = event.usage or {}
        tokens = report.get("total_tokens")
        if not tokens:
            tokens = report.get("prompt_tokens", 0) + report.get("completion_tokens", 0)
        if tokens:
            self._sample_tokens = tokens
            self._sample_chars = usage.message_chars(self._agent.messages)
            self._status.set_sample(self._sample_tokens, self._sample_chars)

    def _push_tokens(self):
        tokens = usage.estimate_tokens(self._agent.messages,
                                       self._sample_tokens,
                                       self._sample_chars)
        self._status.set_tokens(tokens)

    def _run_one(self, text):
        # echo the submitted prompt only now, as the run starts, so the user
        # line lands in order just above the streamed answer.
        self._screen.write(f"> {text}\n\n", kind=Screen.USER)
        self._screen.set_busy(True)
        self._status.busy()
        self._push_tokens()   # reflect the just-added user turn immediately
        try:
            run = self._agent.run(text)
            for event in run:
                _write_event(self._screen, event)
                self._update_status(event)
            self._screen.write("\n", kind=Screen.DEFAULT)
        except Exception as e:
            self._screen.write(f"\n[error: {e}]\n", kind=Screen.ERROR)
        finally:
            self._screen.set_busy(False)
            self._status.idle()
            self._push_tokens()   # fold in the assistant turn + the fresh sample


def _open_messages(screen, agent, status):
    """open the :messages overlay over a snapshot of the conversation and write
    back any edits. idle-gated by the caller: it reads and replaces
    agent.messages directly, which would race a streaming run."""
    msgs = list(agent.get_messages())
    if not msgs:
        screen.write("[no messages yet]\n", kind=Screen.META)
        return
    context_size, prompt_tokens, sample_chars = status.ctx_snapshot()
    edited, _estimate, modified = screen.prompt_messages_overlay(
        msgs,
        context_size=context_size,
        prompt_tokens=prompt_tokens,
        sample_chars=sample_chars)
    if modified:
        agent.set_messages(edited)


def _open_models(screen, agent, status, registry):
    """open the :models picker and switch the agent to the chosen model. the
    catalogue (and prices) come from the registry, which fetches+caches from the
    provider; pins are persisted by the overlay through the same registry."""
    ids = registry.model_ids()
    if not ids:
        screen.write("[no models available]\n", kind=Screen.META)
        return
    picked = screen.prompt_model_overlay(ids, prices=registry.prices(), favorites=True)
    if picked:
        agent.model = picked
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
            screen.write(f"> {content}\n\n", kind=Screen.USER)
        elif role == "assistant":
            screen.write(content + "\n\n", kind=Screen.LLM)


def _session_preview(path, width, max_lines):
    """the picker's right-hand preview: up to max_lines '<role>: <snippet>' rows
    read from the flow at path."""
    if not path:
        return []
    try:
        payload = SessionsRegistry.read_flow(path)
    except (OSError, ValueError):
        return []
    lines = []
    for message in (payload.get("messages") or []):
        role = message.get("role")
        if role == "system": continue
        content = message.get("content")
        if not isinstance(content, str):
            content = str(content)
        snippet = " ".join(content.split())[:width]
        lines.append(f"{role}: {snippet}")
        if len(lines) >= max_lines:
            break
    return lines


def _session_name(path):
    """the agent name behind a .flow path: its basename without the extension."""
    base = os.path.basename(path)
    if base.endswith(".flow"):
        base = base[:-len(".flow")]
    return base


def _load_session(screen, agent, status, path):
    """load a .flow into the agent in place, then refresh the view: clear the
    viewport, replay the conversation, and follow the restored model. shared by
    :load, the :sessions picker, and the --continue/--sessions resume flags.
    caller idle-gates (load mutates messages)."""
    try:
        agent.load(path)
    except (OSError, ValueError) as e:
        screen.write(f"[load error] {e}\n", kind=Screen.ERROR)
        return
    # adopt the loaded file's identity so autosave writes back to it (resume in
    # place) instead of forking to the fresh agent's own <name>.flow.
    agent.name = _session_name(path)
    status.set_model(agent.model)
    screen.clear_buffer()
    _replay_messages(screen, agent.get_messages())
    screen.write(f"[loaded {os.path.basename(path)}]\n", kind=Screen.META)


def _open_sessions(screen, agent, status):
    """open the :sessions picker and load the chosen saved session in place."""
    paths = SessionsRegistry.list_sessions()
    if not paths:
        screen.write("[no saved sessions]\n", kind=Screen.META)
        return
    labels = []
    by_label = {}
    for path in paths:
        label = SessionsRegistry.session_label(path)
        labels.append(label)
        by_label[label] = path

    def _preview(label, width, max_lines):
        return _session_preview(by_label.get(label), width, max_lines)

    picked = screen.prompt_session_overlay(labels, preview_fn=_preview)
    if not picked:
        return
    path = by_label.get(picked)
    if path is None:
        return
    _load_session(screen, agent, status, path)


def _save_session(screen, agent, path):
    """save the agent to path (or its default <name>.flow when path is empty)."""
    try:
        written = agent.save(path or None)
    except OSError as e:
        screen.write(f"[save error] {e}\n", kind=Screen.ERROR)
        return
    screen.write(f"[saved {written}]\n", kind=Screen.META)


def _tool_label(name):
    """the origin column for a tool entry: the MCP server name (the part before
    '__'), or 'tool' for an unprefixed function tool."""
    if "__" in name:
        return name.split("__", 1)[0]
    return "tool"


def _open_tools(screen, agent):
    """open the :tools overlay and apply the new selection. every active tool is
    shown selected - the agent's own tools and the ones its skills pulled in;
    set_tools then diffs the registry to match. discovery spawns each MCP server
    briefly to list its tools, so the caller idle-gates it - and mutating the
    toolset would race a run anyway."""
    from cai.tools import ToolRegistry
    active = set(agent.get_tools())
    available = set(ToolRegistry.available_tools())
    available |= active
    if not available:
        screen.write("[no tools available]\n", kind=Screen.META)
        return
    entries = []
    for name in sorted(available):
        entries.append((name, _tool_label(name)))
    new_selected = screen.prompt_tools_overlay(entries, active)
    agent.set_tools(sorted(new_selected))


def _open_skills(screen, agent):
    """open the :skills overlay and apply the new selection. set_skills rebuilds
    the skill layer (its tools + the system prompt), so the caller idle-gates it
    to keep it off the worker's view."""
    from cai.skills import SkillsRegistry
    names = set(SkillsRegistry.available_skills())
    active = set(agent.get_skills())
    names |= active
    if not names:
        screen.write("[no skills available]\n", kind=Screen.META)
        return
    new_active = screen.prompt_skills_overlay(sorted(names), active)
    agent.set_skills(sorted(new_active))


def _handle_command(screen, agent, status, registry, cmd):
    """dispatch a `:`-command. returns True to quit the loop, else False.

    cmd is the raw command string (the text after ':'); the first token is the
    command name."""
    if cmd == "q" or cmd == "quit":
        return True
    parts = cmd.split(" ", 1)
    head = parts[0]
    arg = ""
    if len(parts) > 1:
        arg = parts[1].strip()
    if head == "clear":
        screen.clear_buffer()
        return False
    if head == "save":
        # a snapshot read; safe while busy. arg is an optional path.
        path = ""
        if arg:
            path = os.path.expanduser(arg)
        _save_session(screen, agent, path)
        return False
    if head == "load":
        # loading replaces the live conversation, so only while idle.
        if screen._busy:
            screen.write("[busy — :load when idle]\n", kind=Screen.META)
            return False
        if not arg:
            screen.write("[usage: :load <path>]\n", kind=Screen.META)
            return False
        _load_session(screen, agent, status, os.path.expanduser(arg))
        return False
    if head == "models":
        # switching only affects the next run, so this is safe while busy.
        _open_models(screen, agent, status, registry)
        return False
    if head == "tools":
        # changing the toolset mutates the live registry, so only while idle.
        if screen._busy:
            screen.write("[busy — open :tools when idle]\n", kind=Screen.META)
            return False
        _open_tools(screen, agent)
        return False
    if head == "skills":
        # rebuilding the skill layer mutates the live registry, so only idle.
        if screen._busy:
            screen.write("[busy — open :skills when idle]\n", kind=Screen.META)
            return False
        _open_skills(screen, agent)
        return False
    if head == "messages":
        # editing the live conversation mid-run would race the worker, so only
        # while idle. opening it under a busy run is refused, not queued.
        if screen._busy:
            screen.write("[busy — open :messages when idle]\n", kind=Screen.META)
            return False
        _open_messages(screen, agent, status)
        return False
    if head == "sessions":
        # loading replaces the live conversation, so only while idle (same race
        # as :messages).
        if screen._busy:
            screen.write("[busy — open :sessions when idle]\n", kind=Screen.META)
            return False
        _open_sessions(screen, agent, status)
        return False
    screen.write(f"[unknown command: :{head}]\n", kind=Screen.META)
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
        pick_session=False):
    """launch the interactive TUI around a fresh in-process Agent. blocks until
    the user quits (:q, Ctrl-C, or EOF). returns the process exit code.

    resume_path loads that saved session at startup (--continue); pick_session
    opens the :sessions picker at startup (--sessions). either resumes the chosen
    session in place, so autosave writes back to it."""
    from cai import config
    from cai.agent import Agent
    from cai.hooks import HookEvent
    from cai.models import ModelsRegistry

    # autosave: persist the session to <name>.flow on every conversation
    # mutation, driven solely by hooks - AFTER_RUN (the final answer landed) and
    # MESSAGES_MUTATED (tool results appended mid-run) together cover every
    # change a run makes (a no-tool turn ends in AFTER_RUN; a tool step appends
    # results and fires MESSAGES_MUTATED). the agent comes through ctx.data, and
    # the hook runs on the worker thread mid-run, so it sees a consistent
    # conversation and never races the main thread.
    def _autosave(ctx):
        saved = (ctx.data or {}).get("agent")

        if saved is None: return
        if not SessionsRegistry.has_real_messages(saved.get_messages()): return

        try:
            saved.save()
        except OSError:
            pass

    autosave_hooks = []
    autosave_hooks.append((HookEvent.AFTER_RUN, _autosave))
    autosave_hooks.append((HookEvent.MESSAGES_MUTATED, _autosave))

    agent = Agent(model=model,
                  system_prompt=system_prompt,
                  tools=tools or [],
                  skills=skills or [],
                  hooks=autosave_hooks,
                  reasoning_effort=reasoning_effort,
                  temperature=temperature,
                  max_steps=max_steps)

    screen = Screen()
    jobs = queue.Queue()
    stop = threading.Event()

    completions = {}
    for name, _help in _PALETTE_COMMANDS:
        completions[name] = []
    screen.set_palette_commands(_PALETTE_COMMANDS, _PALETTE_ARG_COMMANDS)
    screen.set_cmd_completions(completions)

    fallback_limit = int(config.load_optional("default_context_size", _DEFAULT_CONTEXT_SIZE))
    registry = ModelsRegistry(agent.api)
    status = _Status(screen, agent.model, registry, fallback_limit)
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
            agent.stop()
            return True
        return False

    screen.set_interrupt_handler(_on_interrupt)

    worker = _Worker(agent, screen, jobs, stop, status)
    worker.start()
    status_loop = _StatusLoop(status, stop)
    status_loop.start()

    screen.write(f"cai — model {agent.model}. type to chat; :q to quit.\n\n",
                 kind=Screen.META)

    # resume at startup: --continue loads the resolved session directly;
    # --sessions opens the picker. nothing is running yet, so no idle gate.
    if resume_path:
        _load_session(screen, agent, status, resume_path)
    elif pick_session:
        _open_sessions(screen, agent, status)

    try:
        while not stop.is_set():
            user_input = screen.prompt("> ")
            if user_input is None:
                continue
            if screen._command_result is not None:
                cmd = screen._command_result
                screen._command_result = None
                if _handle_command(screen, agent, status, registry, cmd):
                    break
                continue
            # '!text' steers the in-flight run; with nothing running it is just
            # an ordinary prompt with the bang stripped.
            if user_input.startswith("!"):
                steer_text = user_input[1:]
                if screen._busy:
                    agent.steer(steer_text)
                    continue
                user_input = steer_text
            if not user_input.strip():
                continue
            jobs.put(user_input)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop.set()
        agent.stop()
        jobs.put(None)
        worker.join(timeout=2)
        status_loop.join(timeout=2)
        screen.close()
        agent.close()
    return 0
