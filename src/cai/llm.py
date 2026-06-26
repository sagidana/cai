"""llm: the core agentic loop.

call_llm drives the cycle: ask the model -> if it requests tools, run them and
feed the results back -> repeat until the model answers with no tool calls (or
max_steps). It is a *generator*: it yields cai.events.Event objects as output
streams in, and returns the final answer string as the generator's return value
(StopIteration.value). Hooks fire at the documented points so a caller can veto
a tool, rewrite the final answer, or react to a finished turn.

  gen = call_llm(messages, model, api, tools=..., tools_dispatch=...)
  try:
      for event in gen:
          ...                       # stream content / render tool calls
  except StopIteration as stop:
      answer = stop.value           # the final assistant text

This is the smallest loop still faithful to cai's design: streaming, tool
dispatch, and the hook events. Steering, interrupts, strict-format, context
trimming, and stuck detection are deliberately left to later layers.

Layering note: unlike cai's call_llm (which leaves the final assistant message
for a higher `enrich` layer to append), this version appends it to `messages`
itself, so `messages` is the complete transcript once the call returns."""
from __future__ import annotations

import json
import logging
import threading

from cai.events import Event, EventType
from cai.hooks import HookContext, HookEvent, HooksRegistry, ToolCall
from cai.ui import NULL_UI


log = logging.getLogger("cai")


class LLMError(Exception):
    pass


class MaxStepsReached(LLMError):
    def __init__(self, max_steps):
        super().__init__(f"reached max_steps={max_steps} without a final answer")
        self.max_steps = max_steps


def _as_registry(hooks):
    """call_llm takes a HooksRegistry, or None for no hooks. (the list ->
    registry translation lives in Run, not here.)"""
    if hooks is None:
        return HooksRegistry()
    return hooks


def _interrupted(interrupt):
    """True when a kill was requested. interrupt is a threading.Event (or None,
    meaning the run can't be interrupted)."""
    return interrupt is not None and interrupt.is_set()


class SteerQueue:
    """thread-safe queue of steering messages: push() from any thread, drain()
    returns and clears them. call_llm drains it at each turn boundary (via the
    steer= callable) and folds the messages in as user turns."""

    def __init__(self):
        self._lock = threading.Lock()
        self._messages = []

    def push(self, text):
        with self._lock:
            self._messages.append(text)

    def drain(self):
        with self._lock:
            messages = self._messages
            self._messages = []
        return messages


def _parse_args(arguments):
    """parse a tool call's raw argument blob into a dict. returns (ok, args): ok
    is False when the blob was non-empty but not a valid JSON object, so the
    caller can hand the model a clear error instead of running the tool with no
    args (which surfaces later as a confusing 'missing argument'). an empty blob
    is a valid no-arg call."""
    if not arguments:
        return True, {}
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return False, {}
    if not isinstance(parsed, dict):
        return False, {}
    return True, parsed


def _normalize_tool_calls(tool_calls):
    """coerce the provider's raw tool_calls into call records cai can trust: drop
    entries that aren't well-formed function calls, and validate each survivor's
    arguments exactly once here, so _handle_tool_calls never re-parses. each record
    keeps the provider's own id verbatim (the provider matches tool replies on it,
    so we must never rewrite it), the model's original argument string, the parsed
    args, and whether they parsed. dropping is logged, never raised, so one
    malformed call can't abort the whole run."""
    calls = []
    for call in tool_calls:
        if not isinstance(call, dict):
            log.warning("dropping tool call that is not an object: %r", call)
            continue
        if call.get('type') != 'function':
            log.warning("dropping tool call with non-function type: %r", call.get('type'))
            continue
        function = call.get('function')
        if not isinstance(function, dict):
            log.warning("dropping tool call with no function object: %r", call)
            continue
        name = function.get('name')
        if not isinstance(name, str) or not name:
            log.warning("dropping tool call with no function name: %r", call)
            continue
        arguments = function.get('arguments')
        if not isinstance(arguments, str):
            arguments = ''
        valid, args = _parse_args(arguments)
        clean = {}
        clean['id'] = call.get('id')
        clean['name'] = name
        clean['arguments'] = arguments
        clean['args'] = args
        clean['valid'] = valid
        calls.append(clean)
    return calls


def _dispatch_tool(tools_dispatch, name, args):
    """run one tool through the caller's dispatcher, always returning a string.
    failures (no dispatcher, an exception) become an 'Error:' result the model
    reads, rather than blowing up the loop."""
    if tools_dispatch is None:
        return f"Error: no tool dispatcher configured for tool '{name}'"
    try:
        result = tools_dispatch(name, args)
    except Exception as e:
        log.exception("tool %s raised", name)
        return f"Error: tool '{name}' raised: {e}"
    if result is None:
        return ""
    return str(result)


def _turn(api,
          call_messages,
          model,
          tools,
          tool_choice,
          reasoning_effort,
          temperature,
          stream,
          interrupt):
    """Run one model call, yielding content/reasoning events as they arrive.
    Returns (content, reasoning, tool_calls, usage). A set interrupt stops
    reading a long stream early (caught again at the call_llm loop)."""
    if not stream:
        out = api.chat(call_messages,
                       model,
                       tools=tools,
                       tool_choice=tool_choice,
                       reasoning_effort=reasoning_effort,
                       temperature=temperature)
        if out is None:
            return "", "", None, {}
        content, reasoning, tool_calls, usage = out
        if reasoning:
            yield Event(type=EventType.REASONING, text=reasoning)
        if content:
            yield Event(type=EventType.CONTENT, text=content)
        return content or "", reasoning or "", tool_calls, usage or {}

    content_parts = []
    reasoning_parts = []
    tool_calls = None
    usage = {}
    stream_gen = api.chat(call_messages,
                          model,
                          tools=tools,
                          tool_choice=tool_choice,
                          reasoning_effort=reasoning_effort,
                          temperature=temperature,
                          stream=True)
    for delta_content, delta_reasoning, finished_tool_calls, chunk_usage in stream_gen:
        if _interrupted(interrupt): break
        if delta_content:
            content_parts.append(delta_content)
            yield Event(type=EventType.CONTENT, text=delta_content)
        if delta_reasoning:
            reasoning_parts.append(delta_reasoning)
            yield Event(type=EventType.REASONING, text=delta_reasoning)
        if finished_tool_calls is not None:
            tool_calls = finished_tool_calls
        if chunk_usage:
            usage = chunk_usage
    return "".join(content_parts), "".join(reasoning_parts), tool_calls, usage


def _merge_data(hooks_data, **event_keys):
    """the HookContext.data for one fire: the caller's hooks_data with this
    event's own keys layered on top, so caller-supplied data reaches every hook
    while each event can still attach what it needs."""
    data = dict(hooks_data or {})
    data.update(event_keys)
    return data


def _handle_tool_calls(calls,
                       messages,
                       content,
                       reasoning,
                       tools_dispatch,
                       hooks,
                       model,
                       config,
                       ui,
                       hooks_data,
                       usage):
    """Append the assistant turn carrying every call, then run each tool:
    emit tool_call, fire before_tool_call (veto), dispatch, emit tool_result,
    append the tool message, fire messages_mutated + after_tool_call. Mutates
    `messages` in place and yields the tool events. `calls` is the normalized
    list from _normalize_tool_calls, so every entry has an id, name, original
    argument blob, parsed args, and a valid flag; every call gets a matching tool
    message, even one whose arguments don't parse, so the next request's tool_call
    ids all line up."""
    # one assistant message carrying every call (the canonical OpenAI shape:
    # one assistant turn with N tool_calls, then N tool messages). arguments that
    # didn't parse are echoed as '{}' rather than the model's broken blob: a strict
    # provider re-parses the history and 400s on invalid JSON. the original blob is
    # still shown to the model in the tool result below, so it can correct itself.
    wire_calls = []
    for call in calls:
        arguments = call['arguments']
        if not call['valid']:
            arguments = '{}'
        function = {}
        function['name'] = call['name']
        function['arguments'] = arguments
        wire_call = {}
        wire_call['id'] = call['id']
        wire_call['type'] = 'function'
        wire_call['function'] = function
        wire_calls.append(wire_call)

    assistant_msg = {}
    assistant_msg['role'] = 'assistant'
    assistant_msg['content'] = content or ''
    assistant_msg['tool_calls'] = wire_calls
    if reasoning:
        assistant_msg['_reasoning'] = reasoning
    messages.append(assistant_msg)

    for call in calls:
        call_id = call['id']
        name = call['name']
        arguments = call['arguments']
        args = call['args']

        yield Event(type=EventType.TOOL_CALL, tool_name=name, tool_args=args, tool_call_id=call_id)

        tool_call = ToolCall(name=name, arguments=arguments, args=args, id=call_id)
        hook_ctx = HookContext(event=HookEvent.BEFORE_TOOL_CALL,
                               messages=messages,
                               model=model,
                               config=config,
                               ui=ui,
                               usage=usage,
                               tool_call=tool_call,
                               data=_merge_data(hooks_data))
        vetoed = False
        for response in hooks.fire(HookEvent.BEFORE_TOOL_CALL, hook_ctx):
            if response is False:
                vetoed = True

        if vetoed:
            result = f"Error: tool '{name}' was aborted by a before_tool_call hook"
            log.info("tool call: %s vetoed by hook", name)
        elif not call['valid']:
            result = f"Error: arguments for tool '{name}' were not valid JSON: {arguments}"
            log.info("tool call: %s had unparseable arguments", name)
        else:
            result = _dispatch_tool(tools_dispatch, name, args)

        is_error = result.startswith('Error:')
        yield Event(type=EventType.TOOL_RESULT,
                    tool_name=name,
                    tool_result=result,
                    tool_call_id=call_id,
                    is_error=is_error)

        tool_msg = {}
        tool_msg['role'] = 'tool'
        tool_msg['tool_call_id'] = call_id
        tool_msg['content'] = result
        messages.append(tool_msg)

        mutated_ctx = HookContext(event=HookEvent.MESSAGES_MUTATED,
                                  messages=messages,
                                  model=model,
                                  config=config,
                                  ui=ui,
                                  data=_merge_data(hooks_data, name=name, id=call_id))
        hooks.fire(HookEvent.MESSAGES_MUTATED, mutated_ctx)

        after_ctx = HookContext(event=HookEvent.AFTER_TOOL_CALL,
                                messages=messages,
                                model=model,
                                config=config,
                                ui=ui,
                                usage=usage,
                                tool_call=tool_call,
                                content=result,
                                data=_merge_data(hooks_data))
        hooks.fire(HookEvent.AFTER_TOOL_CALL, after_ctx)


def call_llm(messages,
             model,
             api,
             *,
             tools=None,
             tools_dispatch=None,
             hooks=None,
             ui=None,
             interrupt=None,
             steer=None,
             system_prompt=None,
             max_steps=None,
             reasoning_effort=None,
             temperature=None,
             stream=True,
             config=None,
             hooks_data=None):
    """The agentic loop. See the module docstring for the consumer contract.

    messages   - the live conversation; mutated in place as the loop runs.
    api        - a cai.api.OpenAiApi (or anything with the same .chat).
    tools      - JSON tool schemas sent to the model (None disables tools).
    tools_dispatch - callable(name, args_dict) -> result, runs one tool.
    hooks      - a HooksRegistry, or None for no hooks.
    ui         - a UI for hooks to prompt the human, or None for NULL_UI.
    interrupt  - a threading.Event; when set the loop winds down at the next
                 safe boundary and returns the partial text. None = no kill.
    steer      - callable() -> list of pending steering texts; drained at each
                 turn boundary and folded in as user turns. None = no steering.
    Returns the final assistant text (as the generator's return value)."""
    hooks = _as_registry(hooks)
    if ui is None:
        ui = NULL_UI
    if not tools:
        tools = None  # falsy -> the api omits the tools field entirely

    content = ""
    turn = 0
    while True:
        turn += 1
        if max_steps is not None and turn > max_steps:
            raise MaxStepsReached(max_steps)
        if _interrupted(interrupt):
            # killed between turns (or before the first one): stop before the
            # next model call and hand back whatever we have so far.
            return content

        if steer is not None:
            # fold any steering messages into the conversation as user turns, so
            # the next model call sees them (after this turn's tool results).
            for text in steer():
                messages.append({"role": "user", "content": text})

        # the model call wants the system prompt at index 0, but the caller's
        # `messages` must stay system-free and be the live append target for
        # tool turns. prepend the system into a throwaway per-call list.
        call_messages = messages
        if system_prompt:
            call_messages = [{"role": "system", "content": system_prompt}]
            call_messages.extend(messages)

        content, reasoning, tool_calls, usage = yield from _turn(api,
                                                                 call_messages,
                                                                 model,
                                                                 tools,
                                                                 "auto",
                                                                 reasoning_effort,
                                                                 temperature,
                                                                 stream,
                                                                 interrupt)

        if usage:
            yield Event(type=EventType.USAGE, usage=dict(usage))

        if _interrupted(interrupt):
            # killed during the turn (a partial stream, or right after it):
            # return the partial text without the final-answer / tool handling.
            return content

        calls = _normalize_tool_calls(tool_calls or [])

        if not calls:
            # no usable tool call: either a plain answer, or the model sent only
            # malformed calls that normalization dropped. treat the turn as the
            # final answer so a bad batch ends the run cleanly instead of
            # re-looping on identical state until max_steps.
            # let on_final_response hooks rewrite it, append it to the transcript,
            # fire after_run, and hand it back.
            final_ctx = HookContext(event=HookEvent.ON_FINAL_RESPONSE,
                                    messages=messages,
                                    model=model,
                                    config=config,
                                    ui=ui,
                                    usage=usage,
                                    content=content,
                                    data=_merge_data(hooks_data))
            for response in hooks.fire(HookEvent.ON_FINAL_RESPONSE, final_ctx):
                if isinstance(response, str):
                    content = response

            assistant_msg = {}
            assistant_msg['role'] = 'assistant'
            assistant_msg['content'] = content
            if reasoning:
                assistant_msg['_reasoning'] = reasoning
            messages.append(assistant_msg)

            run_ctx = HookContext(event=HookEvent.AFTER_RUN,
                                  messages=messages,
                                  model=model,
                                  config=config,
                                  ui=ui,
                                  usage=usage,
                                  content=content,
                                  data=_merge_data(hooks_data))
            hooks.fire(HookEvent.AFTER_RUN, run_ctx)
            return content

        yield from _handle_tool_calls(calls,
                                      messages,
                                      content,
                                      reasoning,
                                      tools_dispatch,
                                      hooks,
                                      model,
                                      config,
                                      ui,
                                      hooks_data,
                                      usage)

        turn_ctx = HookContext(event=HookEvent.AFTER_TURN,
                               messages=messages,
                               model=model,
                               config=config,
                               ui=ui,
                               usage=usage,
                               data=_merge_data(hooks_data))
        hooks.fire(HookEvent.AFTER_TURN, turn_ctx)
