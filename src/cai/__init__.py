"""cai - a small LLM agent, built from scratch layer by layer.

Layer 0: cai.api    - the OpenAI-compatible HTTP client (the LLM call).
Layer 1: cai.llm    - the core agentic loop (call_llm).
         cai.events - the Event value the loop yields, and EventType.
         cai.hooks  - the hook registry the loop fires.
Layer 2: cai.agent  - Agent (persistent conversation) + Run (one-shot execution).
Entry:   cai.config - bootstrap settings (API key, OpenRouter endpoint).
         cai.cli    - the `cai` command: prompt in, streamed answer out.
"""
import logging

# every module logs through getLogger("cai"); point that at a file so the
# diagnostics (MCP spawns, tool failures, wired turns) land somewhere readable
# instead of the default stderr-only / dropped-below-WARNING behaviour.
logging.basicConfig(
    filename="/tmp/cai.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

from cai.events import Event, EventType
from cai.hooks import HookContext, HookEvent, HooksRegistry, ToolCall
from cai.llm import LLMError, MaxStepsReached, call_llm

__all__ = [
    "Event",
    "EventType",
    "HookContext",
    "HookEvent",
    "HooksRegistry",
    "ToolCall",
    "LLMError",
    "MaxStepsReached",
    "call_llm",
    "Agent",
    "Run",
]


def __getattr__(name):
    # lazy so `import cai` doesn't pull agent (and its config/api) unless used.
    if name == "Agent":
        from cai.agent import Agent
        return Agent
    if name == "Run":
        from cai.agent import Run
        return Run
    raise AttributeError(f"module 'cai' has no attribute {name!r}")
