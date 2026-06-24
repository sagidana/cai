"""cai - a small LLM agent, built from scratch layer by layer.

Layer 0: cai.api    - the OpenAI-compatible HTTP client (the LLM call).
Layer 1: cai.llm    - the core agentic loop (call_llm).
         cai.events - the Event value the loop yields, and EventType.
         cai.hooks  - the hook registry the loop fires.
Entry:   cai.config - bootstrap settings (API key, OpenRouter endpoint).
         cai.cli    - the `cai` command: prompt in, streamed answer out.
"""
from cai.events import Event, EventType
from cai.hooks import HookContext, HookEvent, HookRegistry, ToolCall
from cai.llm import LLMError, MaxStepsReached, call_llm

__all__ = [
    "Event",
    "EventType",
    "HookContext",
    "HookEvent",
    "HookRegistry",
    "ToolCall",
    "LLMError",
    "MaxStepsReached",
    "call_llm",
]
