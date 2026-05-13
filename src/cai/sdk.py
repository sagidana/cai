"""
sdk.py — programmatic SDK surface for cai.

Public classes: Harness, Agent, Event.

The SDK is additive: importing cai.sdk does not touch cli or the TUI,
and `call_llm` in llm.py is invoked through its public callback API.
"""

from __future__ import annotations

import contextvars
import itertools
import json
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Optional

from cai import core
from cai import logger as _cai_logger

log = logging.getLogger("cai.sdk")

# Auto-generated names for anonymous Harness and agent() calls, so log
# records in the TUI have stable, identifying labels.
_harness_seq = itertools.count(1)
_block_seq = itertools.count(1)


# ─── Event ────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """One item yielded while streaming an Agent run.

    Consumers switch on ``.type`` and read the relevant fields:
    - content / reasoning: ``.text``
    - tool_call:           ``.tool_name``, ``.tool_args``, ``.tool_call_id``
    - tool_result:         ``.tool_name``, ``.tool_result``, ``.tool_call_id``, ``.is_error``
    """
    type: Literal["content", "reasoning", "tool_call", "tool_result"]
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None
    is_error: bool = False


# ─── Agent ────────────────────────────────────────────────────────────────────

_DONE = object()  # sentinel pushed onto the queue to signal iterator end


class Agent:
    """Lazy, single-consumption handle for an agent run.

    Construct directly to drive an agent over an explicit ``messages`` array
    (the caller controls what goes into the context), or get one back from
    ``Harness.agent()`` which builds ``messages`` from the harness's own state.

    Iterating yields Event objects as the agent runs. Reading any final-state
    attribute implicitly drains the iterator first. ``wait()`` drains without
    iterating. ``stop()`` aborts the run.
    """

    def __init__(self,
                 messages: list,
                 model: str,
                 system_prompt: str = "",
                 tools: Optional[list] = None,
                 max_turns: Optional[int] = None,
                 strict_format: Optional[str] = None,
                 block_name: str = "",
                 log_ctx: Optional[contextvars.Context] = None,
                 hooks: Optional[list] = None,
                 log_path: Optional[str] = None,
                 harness_name: Optional[str] = None):
        self._input_messages = messages
        self._system_prompt = system_prompt
        # OpenAI-format tool schema list for call_llm. Default-None pattern
        # avoids the mutable-default footgun.
        self._tools = tools if tools is not None else []
        self._model = model
        self._max_turns = max_turns
        self._strict_format = strict_format
        self._hooks = hooks
        # Log plumbing: name used in BLOCK records, and a contextvars
        # snapshot so the worker thread inherits the caller's nesting level
        # (and the caller's active logger, unless log_path overrides below).
        self._block_name = block_name or f"block-{next(_block_seq)}"
        self._harness_name = harness_name
        self._log_ctx = log_ctx if log_ctx is not None else contextvars.copy_context()

        # Per-Agent log scoping. When log_path is supplied, the worker thread
        # writes to that file instead of the surrounding scope's logger. The
        # rebind happens *inside* self._log_ctx so it does not leak to the
        # caller. If a parent logger is active in the caller's scope, we drop
        # a one-line breadcrumb there so the parent file isn't silently empty.
        # If no parent and no log_path either, fall back to the default
        # LOG_PATH so standalone Agent runs always land somewhere.
        self._log_path = log_path
        if log_path is not None:
            agent_logger = _cai_logger.get_logger(log_path)
            parent_logger = _cai_logger._active()
            if parent_logger is not None and parent_logger is not agent_logger:
                _cai_logger.log(1, (
                    f"AGENT REDIRECT  name={self._block_name!r}  "
                    f"path={log_path!r}"
                ))
            # Rebind _current inside the captured context only — the outer
            # scope keeps whatever logger it had.
            self._log_ctx.run(_cai_logger._current.set, agent_logger)
        elif _cai_logger._active() is None:
            # No log_path, no parent, no module default — bring up the
            # default so the run's records are not silently dropped.
            _cai_logger.init(_cai_logger.LOG_PATH)

        # Thread + queue for streaming events out of llm.call_llm
        self._queue: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._interrupt = threading.Event()
        self._started = False
        self._done = False

        # Final fields, populated as the run progresses
        self._text: str = ""
        self._reasoning: str = ""
        self._messages: list = list(messages)  # defensive copy
        self._tool_calls: list = []
        self._finish_reason: str = ""
        self._usage: dict = {}
        self._error: Optional[str] = None

    # ─── iteration / blocking API ─────────────────────────────────────────────

    def __iter__(self) -> Iterator[Event]:
        if self._done:
            return iter(())
        self._ensure_started()
        return self._drain()

    def _drain(self) -> Iterator[Event]:
        while True:
            item = self._queue.get()
            if item is _DONE:
                self._done = True
                return
            yield item

    def wait(self) -> "Agent":
        """Drain the iterator to completion. Idempotent."""
        for _ in self:
            pass
        return self

    def stop(self) -> None:
        """Abort the run. Safe to call at any time; no-op if already done."""
        if self._done:
            return
        self._interrupt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # Drain anything the worker pushed between interrupt and exit.
        try:
            while True:
                item = self._queue.get_nowait()
                if item is _DONE:
                    break
        except queue.Empty:
            pass
        self._done = True
        self._finish_reason = "stopped"   # user intent overrides worker's reason
        self._error = None

    # ─── final-state properties (auto-wait) ───────────────────────────────────

    @property
    def text(self) -> str:
        self.wait()
        return self._text

    @property
    def reasoning(self) -> str:
        self.wait()
        return self._reasoning

    @property
    def messages(self) -> list:
        self.wait()
        return self._messages

    @property
    def tool_calls(self) -> list:
        self.wait()
        return self._tool_calls

    @property
    def finish_reason(self) -> str:
        self.wait()
        return self._finish_reason

    @property
    def usage(self) -> dict:
        self.wait()
        return self._usage

    @property
    def error(self) -> Optional[str]:
        self.wait()
        return self._error

    # ─── internals ────────────────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        # Self-sufficient bootstrap for the Agent-only flow: llm.call_llm
        # below requires llm.setup() to have run. Idempotent — if a Harness
        # (or an earlier Agent) already bootstrapped, this returns the cached
        # context and does no work. Picks up cai.config overrides set either
        # directly or via a prior cai.load_init() call.
        core.bootstrap()
        # Run the worker inside the caller's contextvars snapshot so the
        # log-nesting level propagates across the thread boundary.
        self._thread = threading.Thread(
            target=self._log_ctx.run, args=(self._run,), daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Worker thread body: drive llm.call_llm, translate its callbacks
        into Event objects pushed onto the queue, populate final fields."""
        from cai import llm

        # Emit the BLOCK header from inside the worker thread so it lands in
        # whichever logger is active in this context (Harness's, Agent's own,
        # or the default). For Harness-built Agents, ``harness_name`` was
        # passed in; for standalone Agents it is None and that field is
        # omitted from the record.
        tool_names = [t["function"]["name"] for t in (self._tools or [])]
        header = (
            f"BLOCK  name={self._block_name!r}  "
            f"model={self._model}  strict_format={self._strict_format}  "
            f"tools={tool_names}"
        )
        if self._harness_name is not None:
            header = (
                f"BLOCK  name={self._block_name!r}  "
                f"harness={self._harness_name!r}  "
                f"model={self._model}  strict_format={self._strict_format}  "
                f"tools={tool_names}"
            )
        _cai_logger.log(1, header)
        # Find the last user turn to record as the prompt — works for both
        # Harness-built Agents (prompt appended by Harness.agent) and
        # standalone Agents constructed with caller-owned messages.
        prompt_text = ""
        for m in reversed(self._input_messages):
            if m.get("role") == "user":
                prompt_text = m.get("content") or ""
                break
        _cai_logger.log(2, f"BLOCK PROMPT  {prompt_text}")

        # Prepend system prompt (call_llm expects it as a message, not a param).
        messages = list(self._input_messages)
        if self._system_prompt:
            messages.insert(0, {"role": "system", "content": self._system_prompt})
        self._messages = messages   # so the caller sees mutations live

        def stream_cb(chunk):
            if chunk:
                self._text += chunk
                self._queue.put(Event(type="content", text=chunk))

        def reasoning_cb(chunk):
            if chunk is None:   # end-of-reasoning sentinel from llm.py
                return
            self._reasoning += chunk
            self._queue.put(Event(type="reasoning", text=chunk))

        def event_cb(ev):
            if ev["type"] == "tool_call":
                self._tool_calls.append(ev)
                self._queue.put(Event(
                    type="tool_call",
                    tool_name=ev["name"],
                    tool_args=ev["args"],
                    tool_call_id=ev["id"],
                ))
            elif ev["type"] == "tool_result":
                self._queue.put(Event(
                    type="tool_result",
                    tool_name=ev["name"],
                    tool_call_id=ev["id"],
                    tool_result=ev["result"],
                    is_error=ev["is_error"],
                ))
            elif ev["type"] == "usage":
                # Latest turn's usage. prompt_tokens reflects current context
                # depth; completion_tokens/total_tokens are this turn only.
                self._usage = ev["usage"]

        _cai_logger.push_nest(1)
        try:
            # call_llm mutates `messages` in place and returns the final
            # assistant content string on success. llm.py's own structured
            # log records (TURN, TOOL CALL, [assistant], …) appear as
            # children of the BLOCK header thanks to the nest push above.
            content = llm.call_llm(
                messages=messages,
                model=self._model,
                tools=self._tools,
                max_turns=self._max_turns,
                strict_format=self._strict_format,
                stream_callback=stream_cb,
                reasoning_callback=reasoning_cb,
                event_callback=event_cb,
                interrupt_event=self._interrupt,
                hooks=self._hooks,
            )
            # content is the last assistant turn's text; self._text already
            # accumulated it via stream_cb, but override with the returned
            # value as the authoritative final text.
            self._text = content or self._text
            # call_llm appends intermediate tool-calling turns to `messages`
            # but returns the terminal assistant turn without appending it.
            # Add it so r.messages is self-contained and enrich() picks up
            # the final reply without the caller also having to pass r.text.
            if self._text:
                messages.append({"role": "assistant", "content": self._text})
            self._finish_reason = "stop"
            _cai_logger.log(1, f"BLOCK RESULT  name={self._block_name!r}  "
                            f"len={len(self._text)}\n{self._text}")
        except llm.LLMError as e:
            self._error = str(e)
            self._finish_reason = "error"
            _cai_logger.log(1, f"BLOCK ERROR  name={self._block_name!r}  {e}")
        except Exception as e:
            log.exception("sdk.Agent worker thread failed")
            self._error = f"{type(e).__name__}: {e}"
            self._finish_reason = "error"
            _cai_logger.log(1, f"BLOCK ERROR  name={self._block_name!r}  "
                            f"{type(e).__name__}: {e}")
        finally:
            _cai_logger.pop_nest(1)
            self._queue.put(_DONE)


# ─── Harness ──────────────────────────────────────────────────────────────────

class Harness:
    """A configured agent session.

    Constructor-immutable (except for ``.messages``). Per-call overrides on
    ``agent()`` layer on top of harness state: ``system_prompt``/``skills``/
    ``tools``/``functions`` append (union); ``model`` overrides.
    """

    def __init__(self, *,
                 system_prompt: Optional[str] = None,
                 skills: Optional[list] = None,
                 tools: Optional[list] = None,
                 functions: Optional[list] = None,
                 model: Optional[str] = None,
                 mcp_servers: Optional[list] = None,
                 name: Optional[str] = None,
                 log_path: Optional[str] = None,
                 hooks: Optional[list] = None):
        import cai.tools as _cai_tools

        # 0. Logger setup.
        # - If log_path was provided, this Harness gets its own scoped logger:
        #   every log() call made from within its scope (including from any
        #   Agent that doesn't override) lands in that file. Same path used
        #   twice → same Logger instance (so nesting stays consistent).
        # - If log_path is None and no module-level default exists yet, fall
        #   back to the default path. Pre-existing default is left alone so
        #   the CLI's --log-path keeps winning when SDK is used inside it.
        self._logger_token = None
        if log_path is not None:
            scoped = _cai_logger.get_logger(log_path)
            self._logger_token = _cai_logger._current.set(scoped)
        elif _cai_logger._instance is None:
            _cai_logger.init(_cai_logger.LOG_PATH)

        # 1. Bootstrap — loads config, registers internal MCP, builds APIs,
        #    wires up llm module state.
        ctx = core.bootstrap()
        self._ctx = ctx

        # Public identifier used in structured log records. Auto-generated
        # when not supplied so the TUI can still distinguish concurrent
        # harnesses.
        self._name = name or f"harness-{next(_harness_seq)}"

        # 2. Extra MCP servers
        for cmd in (mcp_servers or []):
            _cai_tools.register_server(cmd)

        # 3. User functions — registered via in-process local-function path.
        self._functions: list = list(functions or [])
        if self._functions:
            _cai_tools.register_local_functions(self._functions)

        # 4. Skills: pulls in skill tool names + skill prompts.
        # Retain the *names* so save()/load() can round-trip without baking
        # the composed system prompt into the flow file.
        self._skills: list = list(skills or [])
        skill_tool_names, skill_prompts = core.load_skills(self._skills)

        # 5. Assembled system prompt via the shared composer.
        # Retain the user-supplied base so skill mutations can recompose
        # without losing it (and so flow files can round-trip).
        self._system_prompt_base = system_prompt
        self._system_prompt = core.compose_system_prompt(
            system_prompt, skill_prompts)

        # 6. Tool allowlist. Refresh available_tools after any registrations.
        # Only tools explicitly listed (or pulled in by skills) are exposed —
        # omitting `tools=` yields an empty toolset, not "everything".
        ctx.available_tools = _cai_tools.get_all_tools()
        all_names = [t["function"]["name"] for t in ctx.available_tools]
        allowlist = set(tools or []) | set(skill_tool_names)
        self._tools = [n for n in all_names if n in allowlist]

        # 7. Model
        self._model = model or ctx.config.get("model")

        # 8. Hooks. No hooks run by default. For the built-in context-budget
        # behaviour, opt in by passing hooks=[("after_turn", mask_hook),
        # ("after_turn", compact_hook)] — both are exported from the cai
        # package.
        self._hooks = hooks

        # Caller-owned conversation state
        self.messages: list = []

        _cai_logger.log(1, f"HARNESS {self._name}  model={self._model}  "
                        f"tools={len(self._tools)}")
        # Push the log nesting so every BLOCK / ENRICHMENT / user log() call
        # made through this Harness folds as a child of the HARNESS record.
        # close() (or __exit__) pops this; __del__ is a safety net.
        _cai_logger.push_nest(1)
        self._nest_active = True

    # ─── read-only introspection ──────────────────────────────────────────────

    @property
    def tools(self) -> list:
        return list(self._tools)

    @property
    def available_tools(self) -> list:
        """Names of every tool registered across all MCP servers (internal +
        external). This is the superset you can draw from when passing
        ``tools=[...]`` to ``Harness`` or ``agent()``. Contrast with ``.tools``,
        which is only the allowlisted subset this harness was configured with.
        """
        import cai.tools as _cai_tools
        return [t["function"]["name"] for t in _cai_tools.get_all_tools()]

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def model(self) -> str:
        return self._model

    @property
    def functions(self) -> list:
        return list(self._functions)

    # ─── agent ────────────────────────────────────────────────────────────────

    def agent(self, *,
                  prompt: Optional[str] = None,
                  system_prompt: Optional[str] = None,
                  skills: Optional[list] = None,
                  tools: Optional[list] = None,
                  functions: Optional[list] = None,
                  model: Optional[str] = None,
                  strict_format: Optional[str] = None,
                  name: Optional[str] = None,
                  hooks: Optional[list] = None) -> Agent:
        """Run an agent over the harness's current context.

        The run starts from a defensive copy of ``self.messages``. If
        ``prompt`` is supplied it is appended as a user turn. To control the
        context explicitly (without going through ``self.messages``), construct
        ``cai.Agent`` directly.
        """
        import cai.tools as _cai_tools

        # Build effective messages from the harness's own state, then append
        # the prompt as a user turn if supplied. Use enrich() to populate
        # self.messages between calls.
        eff_messages: list = list(self.messages)
        if prompt is not None:
            eff_messages.append({"role": "user", "content": prompt})

        # Resolve per-call additions.
        call_skill_tool_names, call_skill_prompts = core.load_skills(skills or [])

        # system_prompt: append to harness base.
        eff_system = self._system_prompt
        if system_prompt:
            eff_system = f"{eff_system}\n\n{system_prompt}" if eff_system else system_prompt
        for p in call_skill_prompts:
            eff_system = f"{eff_system}\n\n{p}" if eff_system else p

        # functions append — register and surface them alongside MCP tools.
        call_fn_names: set = set()
        if functions:
            _cai_tools.register_local_functions(functions)
            call_fn_names = {fn.__name__ for fn in functions}

        # Refresh available_tools if we added local functions this call; this
        # also rebuilds tools._dispatch so call_tool can dispatch them.
        if functions:
            self._ctx.available_tools = _cai_tools.get_all_tools()
        available = self._ctx.available_tools

        # tools append — union of harness tools and any per-call tools + skill tools + new functions.
        all_names = [t["function"]["name"] for t in available]
        added = set(tools or []) | set(call_skill_tool_names) | call_fn_names
        if added:
            eff_names = list(dict.fromkeys(list(self._tools) + [n for n in all_names if n in added]))
        else:
            eff_names = list(self._tools)
        eff_tools = [t for t in available if t["function"]["name"] in set(eff_names)]

        eff_model = model or self._model

        # BLOCK header + BLOCK PROMPT are now emitted by Agent itself from
        # inside its worker thread, so they always land in the same logger
        # that the agent's subsequent records use (Harness's scoped logger
        # via copy_context, or an Agent-level override if log_path is set).
        block_name = name or f"block-{next(_block_seq)}"

        return Agent(
            messages=eff_messages,
            system_prompt=eff_system,
            tools=eff_tools,
            model=eff_model,
            strict_format=strict_format,
            block_name=block_name,
            log_ctx=contextvars.copy_context(),
            hooks=hooks if hooks is not None else self._hooks,
            harness_name=self._name,
        )

    # ─── gate ─────────────────────────────────────────────────────────────────

    def gate(self, options: list, prompt: str,
             *, system_prompt: Optional[str] = None,
             tools: Optional[list] = None,
             skills: Optional[list] = None) -> str:
        """Single-turn strict-format gate: ask a question, get back exactly one
        of ``options``. Wraps the common agent(strict_format=..., ...) idiom
        used for quality checks and routing between harness stages.

        ``system_prompt`` overrides the default "strict quality gate" persona —
        useful when a stage wants a specific reviewer voice (principal engineer,
        security lead, etc.).

        ``tools`` / ``skills`` append to the harness's own (same semantics as
        ``agent()``) — let the gate inspect files, run a search, etc. before
        answering. The strict_format regex still pins the final reply to one
        of ``options``.

        Returns the model's stripped reply, guaranteed to equal one of ``options``.
        """
        import re as _re
        pattern = "|".join(_re.escape(o) for o in options)
        quoted = ", ".join(f"'{o}'" for o in options)
        if system_prompt is None:
            system_prompt = f"You are a strict quality gate. Answer only {quoted}."
        r = self.agent(
            strict_format=f"regex:^({pattern})$",
            system_prompt=system_prompt,
            prompt=prompt,
            tools=tools,
            skills=skills,
            name="gate",
        ).wait()
        return r.text.strip()

    # ─── enrich ───────────────────────────────────────────────────────────────

    def enrich(self, data) -> None:
        """Merge an agent run's output back into harness.messages.

        - ``list[dict]`` → prefix-merge: detect the fork point between ``data``
          and current ``harness.messages`` and append only the diverging tail.
          A leading system message in ``data`` is stripped — the harness owns
          its system prompt and never mutates it here, even when a per-call
          ``skills=`` / ``system_prompt=`` augmented the prompt for that run.
          This lets multiple parallel runs be enriched without clobbering
          each other.
        - ``str``        → append as ``{"role": "assistant", "content": data}``
        """
        if isinstance(data, list):
            incoming = data[1:] if data and data[0].get("role") == "system" else list(data)
            common = 0
            for a, b in zip(self.messages, incoming):
                if a != b:
                    break
                common += 1
            tail = incoming[common:]
            self.messages.extend(tail)
            _cai_logger.log(1, f"ENRICHMENT harness={self._name!r}  "
                            f"kind=messages  fork_at={common}  "
                            f"appended={len(tail)}  total={len(self.messages)}")
            for m in tail:
                _cai_logger.log(2, json.dumps(m, ensure_ascii=False, default=str))
        elif isinstance(data, str):
            self.messages.append({"role": "assistant", "content": data})
            _cai_logger.log(1, f"ENRICHMENT harness={self._name!r}  "
                            f"kind=text  len={len(data)}")
            _cai_logger.log(2, data)
        else:
            raise TypeError(
                f"enrich expects list[dict] or str, got {type(data).__name__}"
            )

    # ─── usage ────────────────────────────────────────────────────────────────

    def usage_percent(self) -> float:
        """Estimated percentage (0–100) of the model's context window currently
        consumed by ``self.messages``. Computed at call time using the same
        chars/4 estimate as ``compact()``.
        """
        from cai.llm import get_model_profile

        total_chars = sum(len(str(m.get('content', ''))) for m in self.messages)
        estimated_tokens = total_chars // 4
        context_limit = get_model_profile(self._model).get('context', 16000) or 16000
        return 100.0 * estimated_tokens / context_limit

    # ─── compact ──────────────────────────────────────────────────────────────

    def compact(self, *, threshold_pct: Optional[float] = None) -> bool:
        """Compact ``self.messages`` using the same summarisation logic as
        the ``compact-if-more-than`` harness directive.

        Middle turns (everything after the first exchange and before the last
        four messages) are replaced by a single ``[memory]`` system message
        generated by the LLM. The first/last turns are preserved verbatim.

        :param threshold_pct: if given, only compact when the estimated token
            usage exceeds this percentage of the configured model's context
            window (same semantics as ``compact-if-more-than``). If ``None``
            (default), compact unconditionally.
        :returns: True if compaction ran, False if it was skipped (threshold
            not met, or not enough messages to compact).
        """
        from cai.llm import _compact_messages, get_model_profile

        if not self.messages:
            return False

        if threshold_pct is not None:
            total_chars = sum(len(str(m.get('content', ''))) for m in self.messages)
            estimated_tokens = total_chars // 4
            profile = get_model_profile(self._model)
            context_limit = profile.get('context', 16000)
            if estimated_tokens < context_limit * (threshold_pct / 100.0):
                _cai_logger.log(1, (
                    f"COMPACT skipped  harness={self._name!r}  "
                    f"~{estimated_tokens} tokens = "
                    f"{estimated_tokens/context_limit:.0%} of {context_limit}  "
                    f"threshold={threshold_pct}%"
                ))
                return False

        n_before = len(self.messages)
        _cai_logger.log(1, f"COMPACT  harness={self._name!r}  messages={n_before}")
        _compact_messages(self.messages, self._model)
        _cai_logger.log(1, f"COMPACT DONE  harness={self._name!r}  "
                        f"{n_before} \u2192 {len(self.messages)} messages")
        return len(self.messages) != n_before

    # ─── save / load ─────────────────────────────────────────────────────────

    FLOW_VERSION = 2

    def save(self, path: str) -> None:
        """Persist the harness's conversation + settings to a flow file.

        The payload is JSON with the CLI-compatible schema v2:

        - ``messages`` with the current system prompt prepended at index 0
          (matches the shape CLI ``:load`` expects).
        - ``settings.system_prompt_base`` — the *user-supplied* base (what
          was passed to ``Harness(system_prompt=...)``). The fully composed
          prompt is never stored; it's always re-derived from base + skills
          on load or on skill mutation, so there is one source of truth.
        - ``settings.skills``, ``settings.selected_tools``, ``settings.model``.

        Tools registered via ``functions=`` can't be serialised (Python
        callables). Their *names* are saved as part of ``selected_tools``,
        but the caller must re-register the same functions before
        ``Harness.load`` can dispatch them.
        """
        payload = {
            "version": Harness.FLOW_VERSION,
            "messages": (
                [{"role": "system", "content": self._system_prompt}] + list(self.messages)
                if self._system_prompt
                else list(self.messages)
            ),
            "settings": {
                "system_prompt_base": self._system_prompt_base,
                "skills": list(self._skills),
                "selected_tools": list(self._tools),
                "model": self._model,
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        _cai_logger.log(1, f"FLOW SAVED  harness={self._name!r}  path={path!r}  "
                        f"messages={len(self.messages)}  tools={len(self._tools)}  "
                        f"skills={self._skills}")

    @classmethod
    def load(cls, path: str, *,
             functions: Optional[list] = None,
             mcp_servers: Optional[list] = None,
             name: Optional[str] = None,
             log_path: Optional[str] = None) -> "Harness":
        """Construct a new Harness from a flow file.

        Accepts v1 files (CLI-saved without ``system_prompt_base``) and v2
        files. Missing fields fall back to constructor defaults.

        ``functions=`` / ``mcp_servers=`` / ``name=`` / ``log_path=`` are
        construction-time concerns that can't be serialised — pass them
        here if the flow used local functions or non-default MCP servers.
        Tools listed in the flow but not present in the current registry
        are logged as a warning and dropped.
        """
        with open(path) as f:
            payload = json.load(f)

        settings = payload.get("settings", {}) or {}
        messages = list(payload.get("messages", []) or [])
        # Drop a leading system message — harness owns the composed prompt.
        if messages and messages[0].get("role") == "system":
            messages = messages[1:]

        harness = cls(
            system_prompt=settings.get("system_prompt_base"),
            skills=settings.get("skills") or [],
            tools=settings.get("selected_tools") or [],
            functions=functions,
            model=settings.get("model"),
            mcp_servers=mcp_servers,
            name=name,
            log_path=log_path,
        )
        harness.messages = messages

        # Warn on tools the flow names but the current registry doesn't expose.
        requested = set(settings.get("selected_tools") or [])
        missing = requested - set(harness._tools)
        if missing:
            log.warning("Harness.load: tools not available in current registry: %s",
                        sorted(missing))
            _cai_logger.log(1, f"FLOW LOAD WARN  harness={harness._name!r}  "
                            f"missing_tools={sorted(missing)}")

        _cai_logger.log(1, f"FLOW LOADED  harness={harness._name!r}  path={path!r}  "
                        f"version={payload.get('version', 1)}  "
                        f"messages={len(harness.messages)}  tools={len(harness._tools)}  "
                        f"skills={harness._skills}")
        return harness

    # ─── clone ────────────────────────────────────────────────────────────────

    def clone(self, *, name: Optional[str] = None) -> "Harness":
        """Return an independent copy of this harness.

        Shares the underlying bootstrap context (config, tool registry, MCP
        servers — so no re-registration cost) but owns its own ``messages``
        list. ``agent()`` / ``enrich()`` / ``compact()`` on the clone do not
        touch the original's conversation state.

        Useful for speculative branches: clone, try something, drop the clone
        if the path didn't pan out — or enrich the original from it if it did.

        Note: per-call ``functions=`` registers globally in ``cai.tools`` (same
        behaviour as a single harness), so new functions registered via the
        clone's ``agent()`` become visible to the original's subsequent calls.
        """
        new = object.__new__(Harness)
        new._ctx = self._ctx
        new._name = name or f"{self._name}-clone-{next(_harness_seq)}"
        new._functions = list(self._functions)
        new._skills = list(self._skills)
        new._system_prompt_base = self._system_prompt_base
        new._system_prompt = self._system_prompt
        new._tools = list(self._tools)
        new._model = self._model
        new._hooks = self._hooks
        new.messages = list(self.messages)
        # Clone inherits the parent's logger via the active contextvar (does
        # not bind its own). No token to reset on close().
        new._logger_token = None

        _cai_logger.log(1, f"HARNESS {new._name}  (clone of {self._name!r})  "
                        f"model={new._model}  tools={len(new._tools)}  "
                        f"messages={len(new.messages)}")
        _cai_logger.push_nest(1)
        new._nest_active = True
        return new

    # ─── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Finalise the harness: emit HARNESS DONE and restore the log
        nesting level pushed in ``__init__``. Idempotent.

        Call this (or use the harness as a context manager) when the script
        is about to create another Harness or do unrelated work — otherwise
        subsequent log records will stay nested under this harness.
        """
        if not getattr(self, "_nest_active", False):
            return
        _cai_logger.pop_nest(1)
        self._nest_active = False
        # Restore the scoped-logger contextvar to whatever it was before
        # this Harness bound itself. No-op if no log_path was provided.
        if getattr(self, "_logger_token", None) is not None:
            _cai_logger._current.reset(self._logger_token)
            self._logger_token = None

    def __enter__(self) -> "Harness":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup for scripts that neither call close() nor use
        # the context manager. May fire at arbitrary points during process
        # shutdown; any failure is silently ignored.
        try:
            self.close()
        except Exception:
            pass
