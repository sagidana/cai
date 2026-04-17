"""
sdk.py — programmatic SDK surface for cai.

Three public classes: Harness, Result, Event. Import via::

    import cai
    h = cai.Harness(system_prompt="...")
    result = h.run_agent(prompt="...")
    for event in result:
        ...
    h.enrich(result.messages)

The SDK is additive: importing cai.sdk does not touch cli or the TUI,
and `call_llm` in llm.py is invoked through its public callback API.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Optional

from cai import core

log = logging.getLogger("cai.sdk")


# ─── Event ────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """One item yielded while streaming a Result.

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


# ─── Result ───────────────────────────────────────────────────────────────────

_DONE = object()  # sentinel pushed onto the queue to signal iterator end


class Result:
    """Lazy, single-consumption handle for a run_agent call.

    Iterating yields Event objects as the agent runs. Reading any final-state
    attribute implicitly drains the iterator first. ``wait()`` drains without
    iterating. ``stop()`` aborts the run.
    """

    def __init__(self,
                 messages: list,
                 system_prompt: str,
                 tool_dicts: list,
                 model: str,
                 config: dict,
                 max_turns: Optional[int] = None,
                 strict_format: Optional[str] = None):
        self._input_messages = messages
        self._system_prompt = system_prompt
        self._tool_dicts = tool_dicts     # OpenAI-format schema list for call_llm
        self._model = model
        self._config = config
        self._max_turns = max_turns
        self._strict_format = strict_format

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

    def wait(self) -> "Result":
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
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Worker thread body: drive llm.call_llm, translate its callbacks
        into Event objects pushed onto the queue, populate final fields."""
        from cai import llm

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

        try:
            # call_llm mutates `messages` in place and returns the final
            # assistant content string on success.
            content = llm.call_llm(
                messages=messages,
                model=self._model,
                tools=self._tool_dicts,
                max_turns=self._max_turns,
                strict_format=self._strict_format,
                stream_callback=stream_cb,
                reasoning_callback=reasoning_cb,
                event_callback=event_cb,
                interrupt_event=self._interrupt,
            )
            # content is the last assistant turn's text; self._text already
            # accumulated it via stream_cb, but override with the returned
            # value as the authoritative final text.
            self._text = content or self._text
            self._finish_reason = "stop"
        except llm.LLMError as e:
            self._error = str(e)
            self._finish_reason = "error"
        except Exception as e:
            log.exception("sdk.Result worker thread failed")
            self._error = f"{type(e).__name__}: {e}"
            self._finish_reason = "error"
        finally:
            self._queue.put(_DONE)


# ─── Harness ──────────────────────────────────────────────────────────────────

class Harness:
    """A configured agent session.

    Constructor-immutable (except for ``.messages``). Per-call overrides on
    ``run_agent`` layer on top of harness state: ``system_prompt``/``skills``/
    ``tools``/``functions`` append (union); ``model``/``task_mode`` override.
    """

    def __init__(self, *,
                 system_prompt: Optional[str] = None,
                 skills: Optional[list] = None,
                 tools: Optional[list] = None,
                 functions: Optional[list] = None,
                 model: Optional[str] = None,
                 task_mode: Optional[str] = None,
                 mcp_servers: Optional[list] = None):
        import cai.tools as _cai_tools

        # 1. Bootstrap — loads config, registers internal MCP, builds APIs,
        #    wires up llm module state.
        ctx = core.bootstrap()
        self._ctx = ctx

        # 2. Extra MCP servers
        for cmd in (mcp_servers or []):
            _cai_tools.register_server(cmd)

        # 3. User functions — registered via in-process local-function path.
        self._functions: list = list(functions or [])
        if self._functions:
            _cai_tools.register_local_functions(self._functions)

        # 4. Skills: pulls in skill tool names + skill prompts
        skill_tool_names, skill_prompts = core.load_skills(skills or [])

        # 5. Assembled system prompt per tri-state:
        #    None → default + mode + skills
        #    ""   → truly empty (nothing, even mode and skills are dropped)
        #    str  → custom base + mode + skills
        if system_prompt == "":
            self._system_prompt = ""
        elif system_prompt is None:
            self._system_prompt = core.assemble_system_prompt(
                ctx.config, task_mode, skill_prompts)
        else:
            parts = [system_prompt]
            if task_mode and task_mode in core.MODE_BLOCKS:
                parts.append(core.MODE_BLOCKS[task_mode])
            parts.extend(skill_prompts)
            self._system_prompt = "\n\n".join(parts)

        # 6. Tool allowlist. Refresh available_tools after any registrations.
        ctx.available_tools = _cai_tools.get_all_tools()
        all_names = [t["function"]["name"] for t in ctx.available_tools]
        allowlist = set(tools or []) | set(skill_tool_names)
        if allowlist:
            self._tools = [n for n in all_names if n in allowlist]
        else:
            self._tools = list(all_names)

        # 7. Model + task_mode
        self._model = model or ctx.config.get("model")
        self._task_mode = task_mode

        # Caller-owned conversation state
        self.messages: list = []

    # ─── read-only introspection ──────────────────────────────────────────────

    @property
    def tools(self) -> list:
        return list(self._tools)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def model(self) -> str:
        return self._model

    @property
    def functions(self) -> list:
        return list(self._functions)

    # ─── run_agent ────────────────────────────────────────────────────────────

    def run_agent(self, *,
                  prompt: Optional[str] = None,
                  messages: Optional[list] = None,
                  system_prompt: Optional[str] = None,
                  skills: Optional[list] = None,
                  tools: Optional[list] = None,
                  functions: Optional[list] = None,
                  model: Optional[str] = None,
                  task_mode: Optional[str] = None,
                  strict_format: Optional[str] = None) -> Result:
        import cai.tools as _cai_tools

        if prompt is None and messages is None:
            raise ValueError("run_agent requires prompt or messages")

        # Build effective messages: defensive copy of caller's list, then
        # append the prompt as a user turn if supplied.
        eff_messages: list = list(messages) if messages else []
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

        # task_mode override: if supplied, replace harness's mode block.
        # (Since the block was already baked into self._system_prompt we
        # simply append the new one — recency bias gives it precedence.)
        if task_mode and task_mode in core.MODE_BLOCKS:
            block = core.MODE_BLOCKS[task_mode]
            eff_system = f"{eff_system}\n\n{block}" if eff_system else block

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
        eff_tool_dicts = [t for t in available if t["function"]["name"] in set(eff_names)]

        eff_model = model or self._model

        return Result(
            messages=eff_messages,
            system_prompt=eff_system,
            tool_dicts=eff_tool_dicts,
            model=eff_model,
            config=self._ctx.config,
            strict_format=strict_format,
        )

    # ─── gate ─────────────────────────────────────────────────────────────────

    def gate(self, options: list, prompt: str,
             *, system_prompt: Optional[str] = None) -> str:
        """Single-turn strict-format gate: ask a question, get back exactly one
        of ``options``. Wraps the common run_agent(strict_format=..., ...) idiom
        used for quality checks and routing between harness stages.

        ``system_prompt`` overrides the default "strict quality gate" persona —
        useful when a stage wants a specific reviewer voice (principal engineer,
        security lead, etc.).

        Returns the model's stripped reply, guaranteed to equal one of ``options``.
        """
        import re as _re
        pattern = "|".join(_re.escape(o) for o in options)
        quoted = ", ".join(f"'{o}'" for o in options)
        if system_prompt is None:
            system_prompt = f"You are a strict quality gate. Answer only {quoted}."
        r = self.run_agent(
            strict_format=f"regex:^({pattern})$",
            system_prompt=system_prompt,
            prompt=prompt,
        ).wait()
        return r.text.strip()

    # ─── enrich ───────────────────────────────────────────────────────────────

    def enrich(self, data) -> None:
        """Merge a Result's output back into harness.messages.

        - ``list[dict]`` → replace harness.messages entirely (full adoption)
        - ``str``        → append as ``{"role": "assistant", "content": data}``
        """
        if isinstance(data, list):
            self.messages = list(data)
        elif isinstance(data, str):
            self.messages.append({"role": "assistant", "content": data})
        else:
            raise TypeError(
                f"enrich expects list[dict] or str, got {type(data).__name__}"
            )
