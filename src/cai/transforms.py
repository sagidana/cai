"""Registry of transforms usable from the :messages overlay.

A transform is any callable with shape ``(messages, **kwargs) -> list[dict]``:
it takes a slice of the conversation and returns a replacement slice. The
overlay splices the return value back into the live messages[] list.

Built-in transforms are registered at import time. User transforms register
via ``@cai.transform(name)`` in init.py. The ad-hoc LLM transform
(``overlay: !``) is also a registered transform named ``"llm"``.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable


log = logging.getLogger("cai")


TransformFn = Callable[..., list[dict]]


@dataclass
class TransformSpec:
    name: str
    fn: TransformFn
    description: str = ""
    params: list[tuple[str, type, Any]] = field(default_factory=list)


_REGISTRY: dict[str, TransformSpec] = {}


def register_transform(
    name: str,
    fn: TransformFn,
    *,
    description: str = "",
    params: list[tuple[str, type, Any]] | None = None,
) -> None:
    if not name or not callable(fn):
        raise ValueError("register_transform requires a non-empty name and a callable fn")
    _REGISTRY[name] = TransformSpec(
        name=name, fn=fn, description=description, params=list(params or [])
    )


def get_transform(name: str) -> TransformSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown transform: {name!r}")
    return _REGISTRY[name]


def list_transforms() -> list[str]:
    return sorted(_REGISTRY.keys())


# ── Built-ins ──────────────────────────────────────────────────────────────

def _tx_drop(messages, **_):
    """Drop every message in the selection."""
    return []


def _tx_prune_content(messages, replacement: str = "[pruned]", **_):
    """Replace each message's text content with ``replacement``.

    tool_calls and other structured fields are preserved so the sequence
    remains valid for replay.
    """
    out = []
    for m in messages:
        nm = dict(m)
        if isinstance(nm.get("content"), list):
            nm["content"] = replacement
        else:
            nm["content"] = replacement
        out.append(nm)
    return out


def _tx_merge_adjacent(messages, **_):
    """Concatenate runs of same-role messages into one.

    tool_calls are dropped from merged runs — the textual content is the
    sensible anchor when consolidating narrative assistant turns. If the
    caller wants to preserve tool_calls they should keep those messages
    unmerged.
    """
    out: list[dict] = []
    for m in messages:
        if out and out[-1].get("role") == m.get("role") and not m.get("tool_calls") \
                and not out[-1].get("tool_calls"):
            prev = out[-1]
            prev_c = prev.get("content") or ""
            cur_c = m.get("content") or ""
            if not isinstance(prev_c, str):
                prev_c = json.dumps(prev_c)
            if not isinstance(cur_c, str):
                cur_c = json.dumps(cur_c)
            prev["content"] = (prev_c + "\n\n" + cur_c).strip()
        else:
            out.append(dict(m))
    return out


def _tx_dedupe_tool_results(messages, **_):
    """Collapse tool results that share (tool_call_id, content).

    Useful after an agent retries the same tool call and produces the
    same output. Keeps the first instance.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "tool":
            key = (m.get("tool_call_id") or "", str(m.get("content") or ""))
            if key in seen:
                continue
            seen.add(key)
        out.append(dict(m))
    return out


def _tx_compact(messages, model: str | None = None, **_):
    """LLM-summarise middle turns into a single [memory] system message.

    Delegates to cai.llm._compact_messages. Requires a model argument —
    the overlay passes the active session model.
    """
    if not messages:
        return []
    if not model:
        raise ValueError("compact transform requires a 'model' argument")
    from cai.llm import _compact_messages
    working = copy.deepcopy(messages)
    _compact_messages(working, model)
    return working


def _extract_json_messages(text: str) -> list:
    """Best-effort extraction of a JSON messages array from LLM output.

    Accepts:
      - bare JSON array: ``[{...}, {...}]``
      - single object: ``{...}`` (wrapped as a one-element list)
      - content inside triple-backtick fences, optionally prefixed ``json``
      - array embedded in prose: finds the first balanced ``[...]`` or ``{...}``

    Raises ValueError on complete failure.
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("empty response")

    # Strip code fences.
    if s.startswith("```"):
        # Drop opening fence line.
        s = s[3:]
        if s.lower().startswith("json"):
            s = s[4:]
        # Drop trailing fence.
        end = s.rfind("```")
        if end != -1:
            s = s[:end]
        s = s.strip()

    # Fast path: direct parse.
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        # Locate the first balanced [ ... ] or { ... } by walking the string
        # while respecting quoted strings and escapes.
        span = _first_balanced_json(s)
        if span is None:
            raise ValueError(f"no JSON array or object found in response: "
                             f"{s[:80]!r}")
        try:
            parsed = json.loads(span)
        except json.JSONDecodeError as e:
            raise ValueError(f"response was not valid JSON: {e}; head={s[:80]!r}") from e

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array or object, got {type(parsed).__name__}")
    for m in parsed:
        if not isinstance(m, dict) or "role" not in m:
            raise ValueError("array entry missing 'role'")
    return parsed


def _first_balanced_json(s: str) -> str | None:
    """Return the first balanced ``[...]`` or ``{...}`` substring, or None."""
    depth = 0
    start = -1
    in_str = False
    escape = False
    opener = None
    closer_for = {"[": "]", "{": "}"}
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if in_str:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if depth == 0 and ch in "[{":
            start = i
            opener = ch
            depth = 1
            continue
        if depth > 0:
            if ch in "[{":
                depth += 1
            elif ch == closer_for.get(opener) and ch == ("]" if opener == "[" else "}"):
                # Only count matching closer for the top-level opener; nested
                # brackets of any kind are tracked via depth.
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
            elif ch in "]}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None


def _tx_llm(messages, instruction: str = "", model: str | None = None,
            stream_callback=None, **_):
    """Ad-hoc transform powered by a single LLM turn.

    The model is asked to rewrite the selection according to ``instruction``
    and return a JSON array of OpenAI chat-completion messages. Markdown
    fences, prose preambles, or a single object (instead of an array) are
    all tolerated. On unrecoverable parse failure, raises ValueError so the
    caller can show the user.

    When ``stream_callback`` is provided, the response is streamed via
    ``openai_api.chat_stream`` and the callback is invoked per content
    chunk (plain string). UI-side exceptions from the callback are swallowed
    so a redraw failure can't strand the transform mid-response.
    """
    if not instruction:
        raise ValueError("instruction is required")
    if not model:
        raise ValueError("model is required")
    from cai.llm import openai_api
    if openai_api is None:
        raise RuntimeError("no openai_api configured (is the CLI bootstrapped?)")

    prompt = (
        "Rewrite the following messages per the instruction. Return a JSON "
        "array of OpenAI chat-completion messages (each with role/content and "
        "optional tool_calls). Return ONLY the array — no prose, no markdown "
        "fences.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Messages: {json.dumps(messages, indent=2, default=str)}"
    )
    request = [{"role": "user", "content": prompt}]

    if stream_callback is not None:
        chunks: list[str] = []
        for chunk, _reasoning, _tool_calls, _usage in openai_api.chat_stream(
                request, model=model):
            if chunk:
                chunks.append(chunk)
                try:
                    stream_callback(chunk)
                except Exception:
                    log.exception("llm transform: stream_callback raised")
        content = ''.join(chunks)
        if not content:
            raise RuntimeError("no response from model (check API key/model)")
    else:
        result = openai_api.chat(request, model=model)
        if not result:
            raise RuntimeError("no response from model (check API key/model)")
        content, _reasoning, _tool_calls, _usage = result
        content = content or ""

    return _extract_json_messages(content)


register_transform("drop", _tx_drop,
                   description="Remove the selected messages entirely.")
register_transform("prune_content", _tx_prune_content,
                   description="Replace each message's content with a placeholder.",
                   params=[("replacement", str, "[pruned]")])
register_transform("merge_adjacent", _tx_merge_adjacent,
                   description="Concatenate runs of same-role messages.")
register_transform("dedupe_tool_results", _tx_dedupe_tool_results,
                   description="Drop tool results that share id+content with earlier ones.")
register_transform("compact", _tx_compact,
                   description="LLM-summarise middle turns into a [memory] entry.",
                   params=[("model", str, None)])
register_transform("llm", _tx_llm,
                   description="Rewrite the selection per a user instruction via the model.",
                   params=[("instruction", str, ""), ("model", str, None)])
