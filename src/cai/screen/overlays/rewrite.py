"""the :messages overlay's ad-hoc LLM rewrite (the ``!`` action).

asks the model to rewrite the selected messages per a free-text instruction and
return a JSON array of replacement messages. it runs a one-shot cai.Run (the
public SDK), so it works wherever the TUI process has a bootstrapped config
(cai -i and local SDK use); a thin remote-attach client has no model and can't
rewrite. this lives in the TUI on purpose - rewriting is a message-editor
action, not core logic, so there is no transform registry behind it."""

import json
import logging


log = logging.getLogger("cai")


def rewrite(messages, instruction="", stream_callback=None):
    """rewrite messages per instruction via a single LLM turn; return a JSON
    array of chat-completion messages. markdown fences, a prose preamble, or a
    single object are tolerated. raises ValueError on unrecoverable parse
    failure so the caller can show the user. stream_callback(content, reasoning,
    tool_calls) is driven per event for the live popup; its exceptions are
    swallowed so a redraw failure can't strand the rewrite mid-response."""
    if not instruction:
        raise ValueError("instruction is required")
    import cai

    prompt = ("Rewrite the following messages per the instruction. Return a JSON "
              "array of OpenAI chat-completion messages (each with role/content and "
              "optional tool_calls). Return ONLY the array — no prose, no markdown "
              "fences.\n\n"
              f"Instruction: {instruction}\n\n"
              f"Messages: {json.dumps(messages, indent=2, default=str)}")
    run = cai.Run(messages=[{"role": "user", "content": prompt}])

    if stream_callback is None:
        run.wait()
    else:
        for event in run:
            if event.type == "content":
                try:
                    stream_callback(event.text, None, None)
                except Exception:
                    log.exception("rewrite: stream_callback raised")
            if event.type == "reasoning":
                try:
                    stream_callback(None, event.text, None)
                except Exception:
                    log.exception("rewrite: stream_callback raised")

    if run.error:
        raise RuntimeError(run.error)
    content = run.text or ""
    if not content:
        raise RuntimeError("no response from model (check API key/model)")
    return _extract_json_messages(content)


def _extract_json_messages(text):
    """best-effort extraction of a JSON messages array from LLM output.
    accepts a bare array, a single object (wrapped as a one-element list),
    content inside triple-backtick fences, or an array embedded in prose.
    raises ValueError on complete failure."""
    s = (text or "").strip()
    if not s:
        raise ValueError("empty response")

    if s.startswith("```"):
        s = s[3:]
        if s.lower().startswith("json"):
            s = s[4:]
        end = s.rfind("```")
        if end != -1:
            s = s[:end]
        s = s.strip()

    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        span = _first_balanced_json(s)
        if span is None:
            raise ValueError(f"no JSON array or object found in response: {s[:80]!r}")
        try:
            parsed = json.loads(span)
        except json.JSONDecodeError as e:
            raise ValueError(f"response was not valid JSON: {e}; head={s[:80]!r}") from e

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array or object, got {type(parsed).__name__}")
    for message in parsed:
        if not isinstance(message, dict) or "role" not in message:
            raise ValueError("array entry missing 'role'")
    return parsed


def _first_balanced_json(s):
    """return the first balanced [...] or {...} substring, or None. walks the
    string respecting quoted strings and escapes."""
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if in_str:
            if ch == "\\":
                escape = True
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if depth == 0 and ch in "[{":
            start = i
            depth = 1
            continue
        if depth == 0: continue

        if ch in "[{":
            depth += 1
        if ch in "]}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None
