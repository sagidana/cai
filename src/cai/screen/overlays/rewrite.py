"""the :messages overlay's ad-hoc LLM rewrite (the ``!`` action).

asks the model to rewrite the selected messages per a free-text instruction.
the model's response text is taken as-is and becomes the content of a single
assistant message that replaces the whole selection. it runs a one-shot
cai.Run (the public SDK), so it works wherever the TUI process has a
bootstrapped config (cai -i and local SDK use); a thin remote-attach client
has no model and can't rewrite. this lives in the TUI on purpose - rewriting
is a message-editor action, not core logic, so there is no transform registry
behind it."""

import json
import logging


log = logging.getLogger("cai")


def rewrite(messages, instruction="", stream_callback=None):
    """rewrite messages per instruction via a single LLM turn; return a
    one-element list holding an assistant message whose content is the model's
    response verbatim. raises on error so the caller can show the user.
    stream_callback(content, reasoning, tool_calls) is driven per event for
    the live popup; its exceptions are swallowed so a redraw failure can't
    strand the rewrite mid-response."""
    if not instruction:
        raise ValueError("instruction is required")
    import cai

    prompt = ("Rewrite the following messages per the instruction. Respond with "
              "the rewritten content ONLY — no prose preamble, no markdown fences. "
              "Your response is used verbatim as the replacement text.\n\n"
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

    content = run.text or ""
    if not content:
        raise RuntimeError("no response from model (check API key/model)")
    return [{"role": "assistant", "content": content}]
