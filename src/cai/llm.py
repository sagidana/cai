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

from cai.events import Event, EventType
from cai.hooks import HookContext, HookEvent, HookRegistry, ToolCall


log = logging.getLogger("cai")


class LLMError(Exception):
    pass


class MaxStepsReached(LLMError):
    def __init__(self, max_steps):
        super().__init__(f"reached max_steps={max_steps} without a final answer")
        self.max_steps = max_steps


def _as_registry(hooks):
    """coerce call_llm's hooks argument to a HookRegistry: a registry is used
    as-is; None is an empty one; a list of (event, fn) pairs is wrapped."""
    if isinstance(hooks, HookRegistry):
        return hooks
    if hooks is None:
        return HookRegistry()
    return HookRegistry(inherit=hooks)


def _parse_args(arguments):
    """best-effort parse of a tool call's raw argument string into a dict."""
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


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
          stream):
    """Run one model call, yielding content/reasoning events as they arrive.
    Returns (content, reasoning, tool_calls, usage)."""
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


def _handle_tool_calls(tool_calls,
                       messages,
                       content,
                       reasoning,
                       tools_dispatch,
                       hooks,
                       model,
                       config,
                       extra,
                       usage):
    """Append the assistant turn carrying every call, then run each tool:
    emit tool_call, fire before_tool_call (veto), dispatch, emit tool_result,
    append the tool message, fire messages_mutated + after_tool_call. Mutates
    `messages` in place and yields the tool events."""
    calls = []
    for call in tool_calls:
        if call.get('type') != 'function':
            log.warning("tool call with invalid type: %s", call.get('type'))
            continue
        calls.append(call)
    if not calls:
        return

    # one assistant message carrying every call (the canonical OpenAI shape:
    # one assistant turn with N tool_calls, then N tool messages).
    wire_calls = []
    for call in calls:
        function = {}
        function['name'] = call.get('function', {}).get('name')
        function['arguments'] = call.get('function', {}).get('arguments') or ''
        wire_call = {}
        wire_call['id'] = call.get('id')
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
        call_id = call.get('id')
        function = call.get('function', {})
        name = function.get('name')
        arguments = function.get('arguments') or ''
        args = _parse_args(arguments)

        yield Event(type=EventType.TOOL_CALL, tool_name=name, tool_args=args, tool_call_id=call_id)

        tool_call = ToolCall(name=name, arguments=arguments, args=args, id=call_id)
        hook_ctx = HookContext(event=HookEvent.BEFORE_TOOL_CALL,
                               messages=messages,
                               model=model,
                               config=config,
                               usage=usage,
                               tool_call=tool_call,
                               extra=extra)
        vetoed = False
        for response in hooks.fire(HookEvent.BEFORE_TOOL_CALL, hook_ctx):
            if response is False:
                vetoed = True

        if vetoed:
            result = f"Error: tool '{name}' was aborted by a before_tool_call hook"
            log.info("tool call: %s vetoed by hook", name)
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
                                  meta={'name': name, 'id': call_id},
                                  extra=extra)
        hooks.fire(HookEvent.MESSAGES_MUTATED, mutated_ctx)

        after_ctx = HookContext(event=HookEvent.AFTER_TOOL_CALL,
                                messages=messages,
                                model=model,
                                config=config,
                                usage=usage,
                                tool_call=tool_call,
                                content=result,
                                extra=extra)
        hooks.fire(HookEvent.AFTER_TOOL_CALL, after_ctx)


def call_llm(messages,
             model,
             api,
             *,
             tools=None,
             tools_dispatch=None,
             hooks=None,
             system_prompt=None,
             max_steps=None,
             reasoning_effort=None,
             temperature=None,
             stream=True,
             config=None,
             extra=None):
    """The agentic loop. See the module docstring for the consumer contract.

    messages   - the live conversation; mutated in place as the loop runs.
    api        - a cai.api.OpenAiApi (or anything with the same .chat).
    tools      - JSON tool schemas sent to the model (None disables tools).
    tools_dispatch - callable(name, args_dict) -> result, runs one tool.
    hooks      - a HookRegistry or a list of (event, fn) pairs.
    Returns the final assistant text (as the generator's return value)."""
    hooks = _as_registry(hooks)
    if not tools:
        tools = None  # falsy -> the api omits the tools field entirely

    content = ""
    turn = 0
    while True:
        turn += 1
        if max_steps is not None and turn > max_steps:
            raise MaxStepsReached(max_steps)

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
                                                                 stream)

        if usage:
            yield Event(type=EventType.USAGE, usage=dict(usage))

        if not tool_calls:
            # final answer. let on_final_response hooks rewrite it, append it
            # to the transcript, fire after_run, and hand it back.
            final_ctx = HookContext(event=HookEvent.ON_FINAL_RESPONSE,
                                    messages=messages,
                                    model=model,
                                    config=config,
                                    usage=usage,
                                    content=content,
                                    extra=extra)
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
                                  usage=usage,
                                  content=content,
                                  extra=extra)
            hooks.fire(HookEvent.AFTER_RUN, run_ctx)
            return content

        yield from _handle_tool_calls(tool_calls,
                                      messages,
                                      content,
                                      reasoning,
                                      tools_dispatch,
                                      hooks,
                                      model,
                                      config,
                                      extra,
                                      usage)

        turn_ctx = HookContext(event=HookEvent.AFTER_TURN,
                               messages=messages,
                               model=model,
                               config=config,
                               usage=usage,
                               extra=extra)
        hooks.fire(HookEvent.AFTER_TURN, turn_ctx)
