"""strict.py: per-run answer-format enforcement, layered above call_llm.

A run may demand its answer match a shape - valid JSON, or a regex. This drives
call_llm from the outside: it appends format guidance to the system prompt,
validates the final answer, and on a mismatch reissues the turn with feedback
until the text matches or the attempts run out. call_llm itself stays unaware of
any of this - strict-format is purely an agent-layer concern.

Because enforcement wraps a whole call_llm run (not a single model call), each
retry reissues the turn from the pre-run conversation: it is meant for shaping a
text answer, so any tools a strict run uses should be side-effect free.

Supported formats:
  'json'                      -- the answer must parse as JSON (normalised on success)
  'regex:<pattern>'           -- the answer must contain a match for the pattern
  'regex-each-line:<pattern>' -- every line of the answer must match the pattern
"""
from __future__ import annotations

import json
import logging
import re

from cai.llm import LLMError


log = logging.getLogger("cai")


def _check_json(content):
    try:
        return True, json.dumps(json.loads(content))
    except Exception:
        return False, content


def _check_regex(pattern, content):
    if re.search(pattern, content):
        return True, content
    return False, content


def _check_regex_each_line(pattern, content):
    for line in content.splitlines():
        if not re.search(pattern, line):
            return False, content
    return True, content


def resolve_format(strict_format):
    """map a strict_format string to (guidance, check_fn, fail_msg_fn, label), or
    None when no enforcement is requested. check_fn(content) -> (ok, normalised);
    fail_msg_fn(attempt, max_attempts, content) -> the feedback shown on a miss."""
    if not strict_format:
        return None

    if strict_format == "json":
        guidance = ("Respond only with a valid JSON object. Do not include markdown "
                    "fences, explanations, or any text outside the JSON.")
        def check_fn(content):
            return _check_json(content)
        def fail_msg_fn(attempt, max_attempts, content):
            return (f"Your previous response was not valid JSON (attempt {attempt}/{max_attempts}). "
                    "Please respond with only a valid JSON object. "
                    "Do not include markdown fences, explanations, or any text outside the JSON.")
        return guidance, check_fn, fail_msg_fn, "json"

    if strict_format.startswith("regex-each-line:"):
        pattern = strict_format[len("regex-each-line:"):]
        guidance = f"Every line of your response must match the regular expression pattern: {pattern}"
        def check_fn(content):
            return _check_regex_each_line(pattern, content)
        def fail_msg_fn(attempt, max_attempts, content):
            return (f"Your previous response contained at least one line that does not match the "
                    f"required pattern (attempt {attempt}/{max_attempts}). Please ensure every line "
                    f"of your response matches the regular expression: {pattern}")
        return guidance, check_fn, fail_msg_fn, f"regex-each-line:{pattern}"

    if strict_format.startswith("regex:"):
        pattern = strict_format[len("regex:"):]
        guidance = f"Your response must match the regular expression pattern: {pattern}"
        def check_fn(content):
            return _check_regex(pattern, content)
        def fail_msg_fn(attempt, max_attempts, content):
            return (f"Your previous response did not match the required pattern (attempt "
                    f"{attempt}/{max_attempts}). Please ensure your response matches the "
                    f"regular expression: {pattern}")
        return guidance, check_fn, fail_msg_fn, f"regex:{pattern}"

    return None


def _augment(system_prompt, guidance):
    if not system_prompt:
        return guidance
    return system_prompt + "\n\n" + guidance


def _user_msg(content):
    msg = {}
    msg["role"] = "user"
    msg["content"] = content
    return msg


def _final_assistant(assistant_msg, content):
    """the clean answer to keep in history: the text that passed validation,
    carrying over the reasoning call_llm captured on the model's turn."""
    msg = {}
    msg["role"] = "assistant"
    msg["content"] = content
    if assistant_msg is not None and assistant_msg.get("_reasoning"):
        msg["_reasoning"] = assistant_msg["_reasoning"]
    return msg


def _interrupted(interrupt):
    return interrupt is not None and interrupt.is_set()


def enforce_strict_format(make_stream,
                          strict_format,
                          system_prompt,
                          messages, *,
                          interrupt=None,
                          max_attempts=4):
    """drive make_stream(system_prompt) until its answer matches strict_format.

    make_stream(system_prompt) -- returns a fresh call_llm generator for one
        attempt, run over `messages`. enforce yields that generator's events and
        returns its final answer text, exactly like call_llm, so the agent can
        `yield from` it transparently.

    On a mismatch the turn is reissued with feedback. The guidance, reminder and
    feedback scaffolding never survive in `messages` - only the conversation plus
    the final answer. Raises LLMError if no attempt matches in max_attempts."""
    resolved = resolve_format(strict_format)
    if resolved is None:
        text = yield from make_stream(system_prompt)
        return text

    guidance, check_fn, fail_msg_fn, label = resolved
    strict_system_prompt = _augment(system_prompt, guidance)

    pre_len = len(messages)
    feedbacks = []
    for attempt in range(1, max_attempts + 1):
        # rebuild the scaffold from scratch each attempt: prior feedback (so the
        # model sees its earlier misses) then a tail reminder that keeps the format
        # salient past any tool results.
        del messages[pre_len:]
        messages.extend(feedbacks)
        messages.append(_user_msg(guidance))

        # drain the attempt's events into a buffer rather than yielding live: a
        # failed attempt is discarded, so only the winning attempt reaches the UI.
        gen = make_stream(strict_system_prompt)
        buffered = []
        while True:
            try:
                event = next(gen)
            except StopIteration as stop:
                text = stop.value
                break
            buffered.append(event)

        # call_llm appended the model's answer at the tail; lift it out, then drop
        # the whole scaffold so a retry (or the final answer) starts from pre_len.
        assistant_msg = None
        if len(messages) > pre_len and messages[-1].get("role") == "assistant":
            assistant_msg = messages[-1]
        del messages[pre_len:]

        content = text
        if content:
            content = content.strip()
        ok, normalised = check_fn(content)
        if ok:
            messages.append(_final_assistant(assistant_msg, normalised))
            yield from buffered
            return normalised

        log.error("strict-format: answer did not match %s (attempt %d/%d): %r",
                  label, attempt, max_attempts, text)
        if _interrupted(interrupt):
            # a kill landed mid-run: keep whatever we have rather than spending
            # more attempts the user asked to stop.
            messages.append(_final_assistant(assistant_msg, text))
            yield from buffered
            return text
        if attempt == max_attempts:
            raise LLMError(f"response did not match required format ({label}) after {max_attempts} attempts")
        feedbacks.append(_user_msg(fail_msg_fn(attempt, max_attempts, text)))
