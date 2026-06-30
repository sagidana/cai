"""context-window accounting shared by every context-% readout: the live status
line (driven by the host) and the :messages overlay. one estimator and one
formatter, so two surfaces never disagree about the same conversation."""

import json

_CHARS_PER_TOKEN = 4  # cold-start guess, used until a real usage sample exists


def message_chars(messages):
    """total characters of a message list, counting content and tool-call
    payloads. the unit only has to be self-consistent: it is calibrated against
    a real prompt_tokens sample, so absolute accuracy doesn't matter."""
    total = 0
    for m in messages:
        content = m.get('content')
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += len(json.dumps(content))
        elif content is not None:
            total += len(str(content))
        for tc in (m.get('tool_calls') or []):
            fn = tc.get('function') or {}
            total += len(fn.get('name') or '')
            total += len(fn.get('arguments') or '')
    return total


def estimate_tokens(messages, sample_tokens=0, sample_chars=0):
    """best-effort token count for messages. with a real usage sample
    (sample_tokens measured when the conversation held sample_chars chars) it
    scales that ratio to the current size; otherwise it falls back to a flat
    chars-per-token guess."""
    chars = message_chars(messages)
    if sample_tokens and sample_chars:
        return max(0, round(sample_tokens * chars / sample_chars))
    return round(chars / _CHARS_PER_TOKEN)


def fmt_ktok(n):
    """token count as a short 'kb' string, 1024-based (12345 -> '12kb')."""
    if not n:
        return "?"
    if n >= 1024:
        return f"{round(n / 1024)}kb"
    return str(n)


def format_ctx(tokens, context_limit):
    """the status-line context string, e.g. 'ctx 5% (5kb/256kb)'.

    tokens is the exact count from the api's usage channel - it only refreshes
    when a real sample arrives, so it reads '?' until the first one."""
    pct = "?"
    if context_limit and tokens:
        pct = f"{tokens / context_limit:.0%}"
    return f"ctx {pct} ({fmt_ktok(tokens)}/{fmt_ktok(context_limit)})"
