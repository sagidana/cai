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

Minimal by design: the only commands are :q/:quit and :clear, and no overlays
beyond the core loop are wired yet. Model/config/messages/tools pickers and
tool-approval gating are added in later layers.
"""
import queue
import threading
import time

from cai import usage
from cai.events import EventType
from cai.screen import Screen


# command palette (Ctrl-P) and command-mode (:) completion entries.
_PALETTE_COMMANDS = [
    ("messages", "view / edit / delete the conversation"),
    ("clear", "clear the conversation view"),
    ("quit", "exit the interactive session"),
]

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

    def __init__(self, screen, model, context_limit):
        self._screen = screen
        self._model = model
        self._context_limit = context_limit
        self._lock = threading.Lock()
        self._busy = False
        self._phase = None         # None | "tool" | "waiting"
        self._stream_kind = None   # None | "responding" | "reasoning"
        self._last_char = 0.0      # time.monotonic() of the last streamed token
        self._tokens = 0           # estimated tokens in the conversation so far
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


def _handle_command(screen, agent, status, cmd):
    """dispatch a `:`-command. returns True to quit the loop, else False.

    cmd is the raw command string (the text after ':'); the first token is the
    command name."""
    if cmd == "q" or cmd == "quit":
        return True
    head = cmd.split(" ", 1)[0]
    if head == "clear":
        screen.clear_buffer()
        return False
    if head == "messages":
        # editing the live conversation mid-run would race the worker, so only
        # while idle. opening it under a busy run is refused, not queued.
        if screen._busy:
            screen.write("[busy — open :messages when idle]\n", kind=Screen.META)
            return False
        _open_messages(screen, agent, status)
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
        max_steps=None):
    """launch the interactive TUI around a fresh in-process Agent. blocks until
    the user quits (:q, Ctrl-C, or EOF). returns the process exit code."""
    from cai import config
    from cai.agent import Agent

    agent = Agent(model=model,
                  system_prompt=system_prompt,
                  tools=tools or [],
                  skills=skills or [],
                  reasoning_effort=reasoning_effort,
                  temperature=temperature,
                  max_steps=max_steps)

    screen = Screen()
    jobs = queue.Queue()
    stop = threading.Event()

    completions = {}
    for name, _help in _PALETTE_COMMANDS:
        completions[name] = []
    screen.set_palette_commands(_PALETTE_COMMANDS)
    screen.set_cmd_completions(completions)

    context_limit = int(config.load_optional("default_context_size", _DEFAULT_CONTEXT_SIZE))
    status = _Status(screen, agent.model, context_limit)
    status.refresh()   # initial idle paint, before the status thread takes over

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

    try:
        while not stop.is_set():
            user_input = screen.prompt("> ")
            if user_input is None:
                continue
            if screen._command_result is not None:
                cmd = screen._command_result
                screen._command_result = None
                if _handle_command(screen, agent, status, cmd):
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
