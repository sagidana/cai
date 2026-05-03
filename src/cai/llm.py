import datetime
import json
import logging
import re

from cai import logger as _cai_logger

log = logging.getLogger("cai")

# ---------------------------------------------------------------------------
# Module-level state — initialised by setup()
# ---------------------------------------------------------------------------
config = {}
openai_api = None
call_tool = None

def _noop_diag(*args, **kwargs):
    pass

_diag = _noop_diag


def setup(cfg, api, ct, diag_fn=None):
    """Initialise llm module state. Called once from cli.init()."""
    global config, openai_api, call_tool, _diag
    config = cfg
    openai_api = api
    call_tool = ct
    if diag_fn is not None:
        _diag = diag_fn


# ---------------------------------------------------------------------------
# call_llm error hierarchy
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Base class for errors raised by call_llm."""

class MaxTurnsReached(LLMError):
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        super().__init__(f"Reached maximum turns ({max_turns}) without a final response.")


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------
# Known model capability profiles. Matched by prefix against the model ID.
# tier:    "small" | "mid" | "large" — drives prompt verbosity, max_turns defaults
# context: context window in tokens
# tool_calling: whether the model reliably supports tool/function calling
MODEL_PROFILES = {
    "arcee-ai/trinity-mini":          {"tier": "small", "context": 8192,   "tool_calling": True},
    "openai/gpt-4o-mini":             {"tier": "mid",   "context": 128000, "tool_calling": True},
    "openai/gpt-4o":                  {"tier": "large", "context": 128000, "tool_calling": True},
    "openai/o1":                      {"tier": "large", "context": 128000, "tool_calling": True},
    "openai/o3":                      {"tier": "large", "context": 200000, "tool_calling": True},
    "anthropic/claude-3-haiku":       {"tier": "small", "context": 200000, "tool_calling": True},
    "anthropic/claude-3-5-haiku":     {"tier": "mid",   "context": 200000, "tool_calling": True},
    "anthropic/claude-3-5-sonnet":    {"tier": "large", "context": 200000, "tool_calling": True},
    "anthropic/claude-3-7-sonnet":    {"tier": "large", "context": 200000, "tool_calling": True},
    "anthropic/claude-opus-4":        {"tier": "large", "context": 200000, "tool_calling": True},
    "anthropic/claude-sonnet-4":      {"tier": "large", "context": 200000, "tool_calling": True},
    "google/gemini-2.0-flash":        {"tier": "mid",   "context": 1000000, "tool_calling": True},
    "google/gemini-2.5-pro":          {"tier": "large", "context": 1000000, "tool_calling": True},
    "meta-llama/llama-3.3-70b":       {"tier": "mid",   "context": 128000, "tool_calling": True},
    "meta-llama/llama-3.1-8b":        {"tier": "small", "context": 128000, "tool_calling": True},
    "mistralai/mistral-small":        {"tier": "small", "context": 32000,  "tool_calling": True},
    "mistralai/mistral-large":        {"tier": "large", "context": 128000, "tool_calling": True},
    "deepseek/deepseek-r1":           {"tier": "large", "context": 64000,  "tool_calling": False},
    "deepseek/deepseek-chat":         {"tier": "mid",   "context": 64000,  "tool_calling": True},
    # Conservative fallback for any unrecognised model
    "_default":                       {"tier": "mid",   "context": 128000,  "tool_calling": True},
}

AGENTIC_SYSTEM_PROMPTS = {
    'small': (
        "You are a CLI assistant with access to tools. "
        "You MUST use the available tools to answer questions — do not guess or make up information. "
        "Follow this process: 1) Call the appropriate tool. 2) Read the result carefully. "
        "3) Answer based only on what the tool returned. "
        "Do not ask clarifying questions. Output only the final answer, nothing else."
    ),
    'mid': (
        "You are a CLI assistant with access to tools. "
        "Use tools to gather information before answering. "
        "Do not ask clarifying questions — make reasonable assumptions. "
        "Be concise in your final response."
    ),
    'large': (
        "You are a CLI assistant. Use available tools to answer accurately. "
        "Do not ask clarifying questions. Be concise."
    ),
}

def get_model_profile(model_id):
    """Return capability profile for model_id.

    Tries exact match first, then prefix match (longest prefix wins),
    then applies any per-model overrides from config.json under 'model_profiles'.
    Falls back to '_default' if nothing matches.
    """
    # Exact match
    profile = MODEL_PROFILES.get(model_id)

    # Prefix match — pick the longest matching prefix
    if profile is None:
        best_prefix = ""
        for prefix, p in MODEL_PROFILES.items():
            if prefix == "_default":
                continue
            if model_id.startswith(prefix) and len(prefix) > len(best_prefix):
                best_prefix = prefix
                profile = p

    if profile is None:
        profile = MODEL_PROFILES["_default"]

    profile = dict(profile)  # shallow copy so overrides don't mutate the source

    # Apply user overrides from config.json: {"model_profiles": {"anthropic/claude-opus-4.6": {"context": 32000}}}
    user_overrides = config.get("model_profiles", {}) if config else {}
    # Exact override
    if model_id in user_overrides:
        profile.update(user_overrides[model_id])
    else:
        # Prefix override
        for prefix, overrides in user_overrides.items():
            if model_id.startswith(prefix):
                profile.update(overrides)
                break

    return profile


# ---------------------------------------------------------------------------
# Tool result trimming
# ---------------------------------------------------------------------------
TOOL_RESULT_MAX_CHARS = 8000
_CHARS_PER_TOKEN = 4  # approximate for mixed code/text content

def _dynamic_max_chars(usage, profile):
    """Return a max_chars limit based on remaining context window, or None if no trimming needed.

    Strategy: when more than 50% of the context window remains, don't trim at all.
    As the window fills, progressively reserve a smaller fraction for each result so
    we don't blow the budget in a single call.
    """
    if not usage or not profile:
        return None
    prompt_tokens = usage.get('prompt_tokens', 0)
    context_limit = profile.get('context', 0)
    if not prompt_tokens or not context_limit:
        return None
    tokens_remaining = max(0, context_limit - prompt_tokens)
    remaining_fraction = tokens_remaining / context_limit
    if remaining_fraction > 0.5:
        return None  # plenty of headroom — skip trimming entirely
    # Shrink the per-result budget as context tightens:
    #   50% remaining → allocate 25% of remaining  (≈ large result still allowed)
    #   25% remaining → allocate 12.5% of remaining
    #   10% remaining → allocate 10% (floor)  — gets aggressive
    budget_fraction = max(0.10, remaining_fraction * 0.5)
    max_chars = int(tokens_remaining * budget_fraction * _CHARS_PER_TOKEN)
    return max(max_chars, 2000)  # always allow at least 2000 chars

def trim_tool_result(result, usage=None, profile=None, max_chars=None):
    if max_chars is None:
        dynamic = _dynamic_max_chars(usage, profile)
        if dynamic is not None:
            max_chars = dynamic
            log.info("trim_tool_result: dynamic limit=%d (prompt_tokens=%s, context=%s)",
                     max_chars,
                     usage.get('prompt_tokens') if usage else None,
                     profile.get('context') if profile else None)
        else:
            max_chars = config.get('tool_result_max_chars', TOOL_RESULT_MAX_CHARS) if config else TOOL_RESULT_MAX_CHARS
    if max_chars and len(result) > max_chars:
        omitted = len(result) - max_chars
        log.info("trim_tool_result: trimmed %d chars (limit=%d)", omitted, max_chars)
        return result[:max_chars] + f"\n[truncated: {omitted} chars omitted]"
    return result


# ---------------------------------------------------------------------------
# Public one-shot LLM call for hook authors / SDK users.
# ---------------------------------------------------------------------------
def chat(messages, model):
    """One-shot LLM call. Returns the assistant text content, or None on failure.

    Intended for use from inside hooks (e.g. an after_tool_call hook that wants
    to LLM-compress the last tool result) without reaching into openai_api.
    """
    result = openai_api.chat(messages, model=model)
    if not result:
        return None
    content, _reasoning, _tool_calls, _usage = result
    return content


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def _execute_tool(call_name, arguments, allowed_tool_names, usage=None, profile=None):
    """Validate and run a single tool call. Always returns a result string.

    allowed_tool_names is the set of tool names actually sent to the LLM for
    this session — validates against it so disabled tools cannot be executed
    even if the LLM hallucinates a call to one.
    """
    if call_name not in allowed_tool_names:
        log.warning("tool call: rejected tool '%s' (not in selected tools)", call_name)
        return f"Error: tool '{call_name}' is not enabled. Enabled tools: {sorted(allowed_tool_names)}"

    try:
        call_args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        log.warning("tool call: bad JSON args for '%s': %s", call_name, e)
        return f"Error: tool '{call_name}' received invalid JSON arguments: {e}. Raw arguments: {arguments!r}"

    try:
        log.info("tool call: %s args=%s", call_name, call_args)
        result = call_tool(call_name, call_args)
    except Exception as e:
        log.error("tool call: %s raised: %s", call_name, e)
        return f"Error: tool '{call_name}' raised an exception: {e}"

    if result is None:
        log.warning("tool call: %s returned None", call_name)
        return f"Error: tool '{call_name}' returned no result"
    log.info("tool call: %s -> result length=%d", call_name, len(result))
    return trim_tool_result(result, usage=usage, profile=profile)


def handle_tool_calls(tool_calls, messages, call_content, allowed_tool_names,
                      tool_callback=None, usage=None, profile=None, reasoning=None,
                      event_callback=None, hooks_by_event=None, model=None):
    log.info("handle_tool_calls: dispatching %d tool call(s)", len(tool_calls))
    hooks_by_event = hooks_by_event if hooks_by_event is not None else _group_hooks(None)
    for call in tool_calls:
        if call.get('type') != 'function':
            log.warning("tool call with invalid type: %s", call.get('type'))
            continue
        call_id = call.get('id')
        call_function = call.get('function', {})
        call_name = call_function.get('name')
        arguments = call_function.get('arguments') or ''

        # Log the dispatch before execution so call/result appear paired
        try:
            parsed_args = json.loads(arguments) if arguments else {}
            args_preview = ', '.join(
                "{}={}".format(k, json.dumps(v)[:80])
                for k, v in parsed_args.items()
            )
        except Exception:
            args_preview = arguments
        _cai_logger.log(2, "TOOL CALL {}({})".format(call_name, args_preview))

        if tool_callback:
            tool_callback(f"-> {call_name}({args_preview})\n")
        if event_callback:
            event_callback({
                'type': 'tool_call',
                'name': call_name,
                'args': parsed_args if isinstance(parsed_args, dict) else {},
                'id': call_id,
            })

        tool_call_dict = {'name': call_name, 'arguments': arguments, 'id': call_id}
        hook_ctx = {
            'messages': messages,
            'model': model,
            'usage': usage or {},
            'tool_call': tool_call_dict,
            'content': None,
        }
        vetoed = _fire_hooks('before_tool_call', hook_ctx, hooks_by_event)

        if vetoed:
            result = f"Error: tool '{call_name}' was aborted by a before_tool_call hook"
            log.info("tool call: %s vetoed by hook", call_name)
        else:
            result = _execute_tool(call_name, arguments, allowed_tool_names,
                                   usage=usage, profile=profile)

        if result.startswith("Error:"):
            _cai_logger.log(2, "TOOL ERROR {} — {}".format(call_name, result))
        else:
            _cai_logger.log(3, "[tool:{}] {:,} chars\n{}".format(call_name, len(result), result))

        if tool_callback:
            if result.startswith("Error:"):
                tool_callback(f"  \u2717 {call_name}: {result}\n", error=True)
            else:
                tool_callback(f"  <- {call_name}: {len(result)} chars\n")
        if event_callback:
            event_callback({
                'type': 'tool_result',
                'name': call_name,
                'result': result,
                'id': call_id,
                'is_error': result.startswith('Error:'),
            })

        assistant_msg = {
            'role': 'assistant',
            'content': call_content or '',
            'tool_calls': [{'id': call_id, 'type': 'function',
                            'function': {'name': call_name, 'arguments': arguments}}],
        }
        if reasoning:
            assistant_msg['_reasoning'] = reasoning
            reasoning = None  # only attach to the first tool call message
        tool_msg = {'role': 'tool', 'tool_call_id': call_id, 'content': result}
        messages.append(assistant_msg)
        messages.append(tool_msg)
        fire_event('messages_mutated', {
            'messages': messages,
            'label': 'tool_call',
            'meta': {'name': call_name, 'id': call_id},
        })

        _fire_hooks('after_tool_call', hook_ctx, hooks_by_event)


# ---------------------------------------------------------------------------
# Format enforcement
# ---------------------------------------------------------------------------
def _retry_until_format(call_fn, system_prompt, check_fn, fail_msg_fn, format_label, messages, max_attempts):
    """Shared retry loop for format enforcement.

    call_fn      -- callable returning (content, reasoning, tool_calls, usage) or falsy
    system_prompt -- injected as a system message before the first attempt (if messages given)
    check_fn     -- callable(content) -> (ok: bool, normalised_content: str)
    fail_msg_fn  -- callable(attempt, max_attempts, content) -> str  (user feedback on failure)
    format_label -- short label used in stderr messages
    messages     -- conversation list mutated between retries; may be None
    max_attempts -- maximum number of LLM calls
    """

    for attempt in range(1, max_attempts + 1):
        # Append strict-format guidance to the existing system prompt rather than
        # inserting a second system message — some providers reject conversations
        # with more than one system message. Restore the original after the call.
        original_system_msg = None
        inserted_system = False
        if messages:
            if messages[0].get('role') == 'system':
                original_system_msg = messages[0]
                existing = original_system_msg.get('content', '') or ''
                sep = "\n\n" if existing else ""
                messages[0] = {**original_system_msg, 'content': existing + sep + system_prompt}
            else:
                messages.insert(0, {'role': 'system', 'content': system_prompt})
                inserted_system = True
        result = call_fn()
        if original_system_msg is not None:
            messages[0] = original_system_msg
        elif inserted_system:
            messages.pop(0)
        if not result:
            return result
        orig_content, reasoning, tool_calls, usage = result

        # do not enforce format when tool calls are present
        if tool_calls:
            return orig_content, reasoning, tool_calls, usage

        ok, normalised = check_fn(orig_content)
        if ok:
            return normalised, reasoning, tool_calls, usage

        log.error(f"failed to get requested format from LLM: {format_label=} -> {orig_content=}, {reasoning=}, {tool_calls=}")
        _diag(f"[enforce_strict_format] attempt {attempt}/{max_attempts}: {fail_msg_fn(attempt, max_attempts, orig_content)}")
        if attempt == max_attempts:
            _diag(f"[enforce_strict_format] giving up after {max_attempts} attempts")
            return None
        if messages is not None:
            messages.append({'role': 'user', 'content': fail_msg_fn(attempt, max_attempts, orig_content)})

def _check_json(content):
    try:
        return True, json.dumps(json.loads(content))
    except Exception:
        return False, content

def _check_regex(pattern, content):
    return (True, content) if re.search(pattern, content) else (False, content)

def _check_regex_each_line(pattern, content):
    for line in content.splitlines():
        if not re.search(pattern, line):
            return (False, content)
    return (True, content)

def enforce_strict_format(call_fn, strict_format, messages=None, max_attempts=4):
    """Retry call_fn() until its content matches strict_format.
    call_fn must return (content, reasoning, tool_calls, usage) or None/falsy.
    messages, if provided, is mutated between retries to inject feedback.

    Supported values for strict_format:
      'json'                    -- response must be a valid JSON object
      'regex:<pattern>'         -- response must match the given regex pattern
      'regex-each-line:<pattern>' -- apply regex to each line; succeeds if any line matches
    """
    if strict_format == 'json':
        return _retry_until_format(
            call_fn,
            system_prompt='Respond only with a valid JSON object. Do not include markdown fences, explanations, or any text outside the JSON.',
            check_fn=_check_json,
            fail_msg_fn=lambda attempt, max_attempts, content: (
                f"Your previous response was not valid JSON (attempt {attempt}/{max_attempts}). "
                "Please respond with only a valid JSON object. "
                "Do not include markdown fences, explanations, or any text outside the JSON."
            ),
            format_label='json',
            messages=messages,
            max_attempts=max_attempts,
        )

    if strict_format and strict_format.startswith('regex-each-line:'):
        pattern = strict_format[len('regex-each-line:'):]
        return _retry_until_format(
            call_fn,
            system_prompt=f'At least one line of your response must match the regular expression pattern: {pattern}',
            check_fn=lambda content: _check_regex_each_line(pattern, content),
            fail_msg_fn=lambda attempt, max_attempts, content: (
                f"Your previous response did not have any line matching the required pattern (attempt {attempt}/{max_attempts}). "
                f"Please ensure at least one line matches the regular expression: {pattern}"
            ),
            format_label=f'regex-each-line:{pattern}',
            messages=messages,
            max_attempts=max_attempts,
        )

    if strict_format and strict_format.startswith('regex:'):
        pattern = strict_format[len('regex:'):]
        return _retry_until_format(
            call_fn,
            system_prompt=f'Your response must match the regular expression pattern: {pattern}',
            check_fn=lambda content: _check_regex(pattern, content),
            fail_msg_fn=lambda attempt, max_attempts, content: (
                f"Your previous response did not match the required pattern (attempt {attempt}/{max_attempts}). "
                f"Please ensure your response matches the regular expression: {pattern}"
            ),
            format_label=f'regex:{pattern}',
            messages=messages,
            max_attempts=max_attempts,
        )

    return call_fn()


# ---------------------------------------------------------------------------
# LLM turn runners
# ---------------------------------------------------------------------------
def _run_nonstreaming_turn(messages, included_tools, model, strict_format=None,
                           reasoning_effort=None, temperature=None,
                           stream_callback=None, tool_choice="auto",
                           interrupt_event=None, reasoning_callback=None):
    """Single non-streaming LLM call. Returns (content, reasoning, tool_calls, usage)."""
    _pre_format_len = len(messages)
    result = enforce_strict_format(lambda: openai_api.chat(messages,
                                                           model=model,
                                                           tools=included_tools,
                                                           tool_choice=tool_choice,
                                                           reasoning_effort=reasoning_effort,
                                                           temperature=temperature),
                                   strict_format,
                                   messages=messages)
    # Strip format-retry feedback messages so they never leak into global context via enrichment.
    # Safe: strict_format only retries on text-only turns (tool_calls bypass the retry loop).
    if strict_format and len(messages) > _pre_format_len:
        del messages[_pre_format_len:]

    if not result: return "", "", None, {}

    content, reasoning, tool_calls, usage = result
    return content or "", reasoning or "", tool_calls, usage

def _run_streaming_turn(messages, included_tools, model, strict_format=None,
                        reasoning_effort=None, temperature=None,
                        stream_callback=None, tool_choice="auto",
                        interrupt_event=None, reasoning_callback=None):
    """Single streaming LLM call. Returns (accumulated_text, reasoning, tool_calls, usage)."""
    accumulated = []
    reasoning_chunks = []
    reasoning_newline_pending = False  # need a newline before content starts
    last_tool_calls = None
    usage = {}
    for chunk, reasoning_chunk, tool_calls, usage in openai_api.chat_stream(messages,
                                                                             model=model,
                                                                             tools=included_tools,
                                                                             tool_choice=tool_choice,
                                                                             reasoning_effort=reasoning_effort,
                                                                             temperature=temperature):
        if interrupt_event and interrupt_event.is_set():
            break
        if reasoning_chunk:
            reasoning_chunks.append(reasoning_chunk)
            if reasoning_callback:
                reasoning_callback(reasoning_chunk)
            else:
                _diag(reasoning_chunk, end="", ensure_newline=not reasoning_newline_pending)
            reasoning_newline_pending = True
        if chunk:
            if reasoning_newline_pending:
                if reasoning_callback:
                    reasoning_callback(None)  # signal end of reasoning
                else:
                    _diag("")   # close the reasoning line before content starts
                reasoning_newline_pending = False
            accumulated.append(chunk)
            if stream_callback:
                stream_callback(chunk)
        if tool_calls:
            last_tool_calls = tool_calls
    if reasoning_newline_pending:
        if reasoning_callback:
            reasoning_callback(None)
        else:
            _diag("")   # ensure we end on a new line even if no content followed
    return "".join(accumulated), "".join(reasoning_chunks), last_tool_calls, usage


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

# Observation masking: replace old tool results with a placeholder (no LLM call).
# Fires first at the lower threshold.
OBSERVATION_MASK_THRESHOLDS = {'small': 0.45, 'mid': 0.60, 'large': 0.65}
OBSERVATION_MASK_KEEP = 3  # number of recent tool results to keep verbatim

# LLM compaction: summarise middle turns into a memory message.
# Fires second at the higher threshold, after masking has already been applied.
CONTEXT_BUDGET_THRESHOLDS = {'small': 0.60, 'mid': 0.75, 'large': 0.80}


def _apply_observation_mask(messages):
    """Replace content of older tool results with a short placeholder.

    Keeps the most recent OBSERVATION_MASK_KEEP tool messages verbatim and
    replaces the content of older ones in-place. No LLM call needed.
    Returns the number of messages masked.
    """
    keep = config.get('observation_mask_keep', OBSERVATION_MASK_KEEP) if config else OBSERVATION_MASK_KEEP

    # Find tool messages that haven't already been masked
    tool_indices = [
        i for i, m in enumerate(messages)
        if m.get('role') == 'tool'
        and not str(m.get('content', '')).startswith('[result omitted')
    ]

    to_mask = tool_indices[:-keep] if len(tool_indices) > keep else []
    if not to_mask:
        log.info("observation_mask: nothing to mask (%d tool msgs, keep=%d)", len(tool_indices), keep)
        return 0

    chars_freed = 0
    for i in to_mask:
        content = messages[i].get('content', '')
        if isinstance(content, str) and len(content) > 100:
            chars_freed += len(content)
            messages[i] = dict(messages[i])  # avoid mutating shared refs
            messages[i]['content'] = f'[result omitted — {len(content)} chars]'

    log.info("observation_mask: masked %d tool results, freed ~%d chars (~%d tokens)",
             len(to_mask), chars_freed, chars_freed // _CHARS_PER_TOKEN)
    _cai_logger.log(2, "MASK replaced {} tool results with placeholders ({} chars freed)".format(
        len(to_mask), chars_freed))
    fire_event('messages_mutated', {
        'messages': messages,
        'label': 'observation_mask',
        'meta': {'masked': len(to_mask), 'chars_freed': chars_freed},
    })
    return len(to_mask)


def _compact_messages(messages, model):
    """Summarize middle turns into a memory message to free up context space."""
    # Determine the slice to compact: after [system?][first_user], before last 4 messages
    start_idx = 0
    if messages and messages[0].get('role') == 'system':
        start_idx = 1
    if start_idx < len(messages) and messages[start_idx].get('role') == 'user':
        start_idx += 1

    end_idx = max(start_idx, len(messages) - 4)
    compactable = messages[start_idx:end_idx]
    if len(compactable) < 2:
        log.info("compaction: not enough messages to compact (%d)", len(compactable))
        return

    _cai_logger.log(2, "TRIM compacting {} messages (indices {}–{}) into a memory entry".format(
        len(compactable), start_idx, end_idx - 1))

    compaction_msgs = [{"role": "user", "content": (
        "Summarize the following conversation turns into a concise memory entry. "
        "Preserve all key facts, tool results, findings, and decisions. "
        "Be specific and concrete. Output only the summary, no preamble.\n\n"
        f"{json.dumps(compactable, indent=2)}"
    )}]

    result = openai_api.chat(compaction_msgs, model=model)
    if not result:
        log.warning("compaction: LLM call failed, skipping")
        _cai_logger.log(2, "TRIM compaction LLM call failed — messages unchanged")
        return

    summary, _, _, _ = result
    memory = {"role": "system", "content": f"[memory from compacted turns]: {summary}"}
    log.info("compaction: replaced %d messages with memory (%d chars)", len(compactable), len(summary))
    messages[start_idx:end_idx] = [memory]
    _cai_logger.log(2, "TRIM done — replaced {} messages with 1 [memory] entry ({} chars)\n{}".format(
        len(compactable), len(summary), summary))
    fire_event('messages_mutated', {
        'messages': messages,
        'label': 'compact',
        'meta': {'compacted': len(compactable)},
    })


def mask_hook(ctx):
    """after_turn hook. Fires at observation_mask_pct — masks old tool results
    with a short placeholder. No LLM call.

    Opt-in: register via Harness(hooks=[("after_turn", mask_hook), ...])."""
    messages = ctx["messages"]
    usage = ctx["usage"]
    profile = get_model_profile(ctx["model"])
    prompt_tokens = usage.get('prompt_tokens', 0)
    context_limit = profile.get('context', 16000)
    if not prompt_tokens or not context_limit:
        return
    budget_pct = prompt_tokens / context_limit
    default_threshold = OBSERVATION_MASK_THRESHOLDS.get(profile['tier'], 0.60)
    threshold = config.get('observation_mask_pct', default_threshold) if config else default_threshold
    if budget_pct < threshold:
        return
    log.warning("context budget: %.0f%% used (%d/%d tokens), applying observation mask",
                budget_pct * 100, prompt_tokens, context_limit)
    _diag(f"\n[context {budget_pct:.0%} >= {threshold:.0%}] masking old tool results...")
    _apply_observation_mask(messages)


def compact_hook(ctx):
    """after_turn hook. Fires at context_budget_pct — LLM-summarises middle
    turns into a memory message.

    Opt-in: register via Harness(hooks=[("after_turn", compact_hook), ...])."""
    messages = ctx["messages"]
    usage = ctx["usage"]
    model = ctx["model"]
    profile = get_model_profile(model)
    prompt_tokens = usage.get('prompt_tokens', 0)
    context_limit = profile.get('context', 16000)
    if not prompt_tokens or not context_limit:
        return
    budget_pct = prompt_tokens / context_limit
    default_threshold = CONTEXT_BUDGET_THRESHOLDS.get(profile['tier'], 0.75)
    threshold = config.get('context_budget_pct', default_threshold) if config else default_threshold
    if budget_pct < threshold:
        return
    log.warning("context budget: %.0f%% used (%d/%d tokens), compacting",
                budget_pct * 100, prompt_tokens, context_limit)
    _diag(f"\n[context {budget_pct:.0%} >= {threshold:.0%}] compacting...")
    _compact_messages(messages, model)


# No hooks run by default. mask_hook and compact_hook are exported from the
# cai package for users who want the built-in context-budget behaviour.
DEFAULT_HOOKS = []


# ---------------------------------------------------------------------------
# Hook dispatch
# ---------------------------------------------------------------------------
VALID_HOOK_EVENTS = ("before_tool_call", "after_tool_call", "after_turn", "on_final_response", "messages_mutated")


def _group_hooks(hooks):
    """Group a list of (event, fn) tuples into {event: [fn, ...]}. Validates event names."""
    grouped = {e: [] for e in VALID_HOOK_EVENTS}
    for event, fn in (hooks if hooks is not None else DEFAULT_HOOKS):
        if event not in grouped:
            raise ValueError(f"unknown hook event: {event!r}. Valid: {VALID_HOOK_EVENTS}")
        grouped[event].append(fn)
    return grouped


def fire_event(event, ctx):
    """Fire a user hook event from outside a call_llm turn.

    call_llm groups hooks once at turn start and uses _fire_hooks with the
    grouped dict. Mutation sites in cli.py and the screen overlays don't
    have that grouping in scope — they use this helper, which reads
    userconfig._user_hooks directly.
    """
    from cai.userconfig import _user_hooks
    for e, fn in _user_hooks:
        if e != event:
            continue
        try:
            fn(ctx)
        except Exception:
            log.exception("hook %r for %s raised", getattr(fn, '__name__', repr(fn)), event)


def _fire_hooks(event, ctx, hooks_by_event):
    """Fire hooks for `event` in registration order.

    Exceptions are logged and do not stop the chain.
    Return value:
      before_tool_call  -> True if any hook vetoed (returned False), else False.
      on_final_response -> the replacement string if the last hook returned one, else None.
      other events      -> None.
    """
    vetoed = False
    replacement = None
    for fn in hooks_by_event.get(event, ()):
        try:
            result = fn(ctx)
        except Exception:
            log.exception("hook %r for %s raised", getattr(fn, '__name__', repr(fn)), event)
            continue
        if event == "before_tool_call" and result is False:
            vetoed = True
        elif event == "on_final_response" and isinstance(result, str):
            replacement = result
    if event == "before_tool_call":
        return vetoed
    if event == "on_final_response":
        return replacement
    return None


# ---------------------------------------------------------------------------
# Stuck detection
# ---------------------------------------------------------------------------
def _warn_if_stuck(tool_calls, call_history, messages):
    """Track repeated identical tool calls and inject a warning into messages when stuck."""
    for call in tool_calls:
        name = call.get('function', {}).get('name', '')
        args_str = call.get('function', {}).get('arguments', '')
        key = (name, args_str)
        call_history[key] = call_history.get(key, 0) + 1
        if call_history[key] >= 2:
            warning = (f"Warning: you have already called tool '{name}' with these exact arguments "
                       f"{call_history[key]} times and received the same result. "
                       f"Try a different approach or tool to make progress.")
            log.warning("stuck detection: '%s' called %d times with same args", name, call_history[key])
            messages.append({"role": "user", "content": warning})
            fire_event('messages_mutated', {
                'messages': messages,
                'label': 'stuck_warning',
                'meta': {'tool': name, 'count': call_history[key]},
            })


def _emit_status(text, status_callback):
    if status_callback:
        status_callback(text)


# ---------------------------------------------------------------------------
# Main LLM loop
# ---------------------------------------------------------------------------
def call_llm(messages,
             model,
             tools,
             strict_format=None,
             force_tools=False,
             max_turns=None,
             reasoning_effort=None,
             temperature=None,
             oneline=False,
             stream_callback=None,
             status_callback=None,
             tool_callback=None,
             ctx_callback=None,
             interrupt_event=None,
             reasoning_callback=None,
             event_callback=None,
             hooks=None):
    # Names the LLM was given — used to gate execution in _execute_tool.
    allowed_tool_names = {t.get('function', {}).get('name') for t in tools}

    profile = get_model_profile(model)

    # Group hooks by event once. hooks=None picks up DEFAULT_HOOKS.
    hooks_by_event = _group_hooks(hooks)

    log.info("call_llm: model=%s tier=%s context=%d messages=%d tools=%d streaming=%s strict_format=%s max_turns=%s",
             model,
             profile['tier'],
             profile['context'],
             len(messages),
             len(tools),
             stream_callback,
             strict_format or "none",
             max_turns if max_turns is not None else "unlimited")

    call_history = {}  # (tool_name, args_str) -> call count, for stuck detection

    # Pre-dispatch: when messages arrive already ending on an assistant
    # with pending tool_calls (typical after the :messages "fork" action
    # leaves the user-edited tool call un-executed), run those calls now
    # before asking the API for more. handle_tool_calls re-appends its
    # own assistant+tool pairs, so pop the source row first.
    if (messages and messages[-1].get('role') == 'assistant'
            and (messages[-1].get('tool_calls') or [])):
        _pending = messages.pop()
        _pending_tc        = _pending.get('tool_calls') or []
        _pending_content   = _pending.get('content', '') or ''
        _pending_reasoning = _pending.get('_reasoning')
        # Suppress the duplicate "-> name(args)" header from handle_tool_calls
        # — the caller's view has already rendered the edited tool call.
        # The result line ("  <- name: N chars") still comes through.
        def _tc_filter(line, error=False):
            if tool_callback is None:
                return
            if line.startswith('-> '):
                return
            tool_callback(line, error=error)
        handle_tool_calls(
            _pending_tc, messages, _pending_content, allowed_tool_names,
            tool_callback=_tc_filter, profile=profile,
            reasoning=_pending_reasoning, event_callback=event_callback,
            hooks_by_event=hooks_by_event, model=model,
        )
        if tool_callback:
            tool_callback("\n")

    # strict_format requires the full response before validation/retry, so it
    # cannot coexist with the streaming path (which emits chunks live and has
    # no format enforcement). Route to non-streaming whenever strict_format
    # is set, regardless of stream_callback.
    if stream_callback and not strict_format:
        run_turn = _run_streaming_turn
    else:
        run_turn = _run_nonstreaming_turn

    turn = 0
    while True:
        turn += 1
        if max_turns is not None and turn > max_turns:
            break

        # Force at least one tool call on turn 1 in agentic mode so the model
        # doesn't skip tools and answer directly from training data.
        # Skip when reasoning is enabled — Anthropic rejects tool_choice=required
        # combined with extended thinking.
        if force_tools and turn == 1 and not reasoning_effort:
            tool_choice = "required"
        else:
            tool_choice = "auto"

        turns_label = f"{turn}/{max_turns}" if max_turns is not None else str(turn)
        _cai_logger.push_nest(1)
        _cai_logger.log(1, "=== TURN {} ===  [{}]{}".format(
            turns_label,
            datetime.datetime.now().strftime("%H:%M:%S"),
            "  tool_choice=required (force_tools)" if tool_choice == "required" else ""))

        content, reasoning, tool_calls, usage = run_turn(messages,
                                                         tools,
                                                         model,
                                                         strict_format=strict_format,
                                                         reasoning_effort=reasoning_effort,
                                                         temperature=temperature,
                                                         stream_callback=stream_callback,
                                                         tool_choice=tool_choice,
                                                         interrupt_event=interrupt_event,
                                                         reasoning_callback=reasoning_callback)
        if reasoning:
            _cai_logger.log(2, "REASONING")
            _cai_logger.log(3, reasoning)

            if not stream_callback:
                # Streaming already emitted reasoning live; for non-streaming
                # emit the full block now, before content appears.
                if reasoning_callback:
                    reasoning_callback("[thinking]\n" + reasoning)
                    reasoning_callback(None)
                else:
                    _diag("[thinking]\n{}".format(reasoning))
        log.info("call_llm: turn=%d tokens prompt=%s completion=%s total=%s",
                 turn,
                 usage.get('prompt_tokens'),
                 usage.get('completion_tokens'),
                 usage.get('total_tokens'))

        prompt_tokens = usage.get('prompt_tokens', 0)
        if prompt_tokens and ctx_callback:
            pct = f"{prompt_tokens / profile['context']:.0%}" if profile['context'] else "?"
            ctx_str = f"ctx {pct} ({prompt_tokens}/{profile['context']})"
            ctx_callback(ctx_str)

        # Surface raw usage to SDK consumers (Result._usage, etc).
        if usage and event_callback:
            event_callback({'type': 'usage', 'usage': dict(usage), 'turn': turn})

        if not tool_calls:
            final_ctx = {
                'messages': messages,
                'model': model,
                'usage': usage or {},
                'tool_call': None,
                'content': content,
            }
            replacement = _fire_hooks('on_final_response', final_ctx, hooks_by_event)
            if replacement is not None:
                content = replacement
            log.info("call_llm: done turn=%d length=%d", turn, len(content))
            _cai_logger.log(2, "[assistant]\n{}".format(content))
            _emit_status("ready", status_callback)
            _cai_logger.pop_nest(1)              # pop before early return
            return content

        if content and stream_callback:
            stream_callback('\n')

        if not oneline:
            def _fmt_call(c):
                name = c.get('function', {}).get('name', '?')
                raw_args = c.get('function', {}).get('arguments', '')
                try:
                    parsed = json.loads(raw_args) if raw_args else {}
                    args_str = ", ".join(
                        f"{k}={json.dumps(v)[:80]}{'...' if len(json.dumps(v)) > 80 else ''}"
                        for k, v in parsed.items()
                    )
                except Exception:
                    args_str = raw_args[:160] + ("..." if len(raw_args) > 160 else "")
                return f"{name}({args_str})"

            tool_calls_fmt = [_fmt_call(c) for c in tool_calls]
            status = f"[turn {turns_label}] {', '.join(tool_calls_fmt)}"
            _emit_status(status, status_callback)

        handle_tool_calls(tool_calls, messages, content, allowed_tool_names,
                          tool_callback=tool_callback, usage=usage, profile=profile,
                          reasoning=reasoning, event_callback=event_callback,
                          hooks_by_event=hooks_by_event, model=model)
        if tool_callback:
            tool_callback("\n")

        # _warn_if_stuck may append a [USER] warning into messages
        if config.get('stuck_detection', False):
            _warn_if_stuck(tool_calls, call_history, messages)

        turn_ctx = {
            'messages': messages,
            'model': model,
            'usage': usage or {},
            'tool_call': None,
            'content': None,
        }
        _fire_hooks('after_turn', turn_ctx, hooks_by_event)
        _cai_logger.pop_nest(1)              # pop at end of loop body

    log.warning("call_llm: reached max_turns=%s", max_turns)
    _cai_logger.log(1, "MAX TURNS REACHED ({})".format(max_turns))
    _emit_status(f"[!] reached max turns ({max_turns})", status_callback)
    raise MaxTurnsReached(max_turns)  # only reached when max_turns is set

