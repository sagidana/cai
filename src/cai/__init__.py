"""cai - a small LLM agent, built from scratch layer by layer.

Layer 0: cai.api    - the OpenAI-compatible HTTP client (the LLM call).
Layer 1: cai.llm    - the core agentic loop (call_llm).
         cai.events - the Event value the loop yields, and EventType.
         cai.hooks  - the hook registry the loop fires.
Layer 2: cai.agent  - Agent (persistent conversation) + Run (one-shot execution).
Entry:   cai.config      - bootstrap settings (API key, OpenRouter endpoint).
         cai.environment - the loaded install catalogue (tools/hooks/commands/
                           settings) an Agent resolves against.
         cai.cli         - the `cai` command: prompt in, streamed answer out.
"""
import logging
from typing import TYPE_CHECKING

# every module logs through getLogger("cai"); point that at a file so the
# diagnostics (MCP spawns, tool failures, wired turns) land somewhere readable
# instead of the default stderr-only / dropped-below-WARNING behaviour.
logging.basicConfig(
    filename="/tmp/cai.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

from cai.events import Event, EventType
from cai.hooks import HookContext, HookEvent, HooksRegistry, ToolCall, hook
from cai.commands import Command, CommandContext, command
from cai.llm import LLMError, MaxStepsReached, call_llm

# Agent/Run/Environment stay lazy at runtime (see __getattr__ below) so
# `import cai` doesn't pull agent + its config/api. this block is type-checker
# only - it never runs - so an editor resolves cai.Agent / cai.Run to their real
# definitions (go-to-def) without paying the import.
if TYPE_CHECKING:
    from cai.agent import Agent, Run, RunInFlight
    from cai.api import ApiError
    from cai.environment import Environment
    from cai.tools import ToolsRegistry, tool, mcp_server

__all__ = [
    "Event",
    "EventType",
    "HookContext",
    "HookEvent",
    "HooksRegistry",
    "ToolCall",
    "hook",
    "Command",
    "CommandContext",
    "command",
    "ToolsRegistry",
    "tool",
    "mcp_server",
    "ApiError",
    "LLMError",
    "MaxStepsReached",
    "call_llm",
    "Agent",
    "Run",
    "RunInFlight",
    "Environment",
    "settings",
]


def __getattr__(name):
    # lazy so `import cai` doesn't pull agent (and its config/api) unless used.
    if name == "Agent":
        from cai.agent import Agent
        return Agent
    if name == "Run":
        from cai.agent import Run
        return Run
    if name == "RunInFlight":
        from cai.agent import RunInFlight
        return RunInFlight
    if name == "Environment":
        from cai.environment import Environment
        return Environment
    # api pulls requests; lazy so `import cai` stays light.
    if name == "ApiError":
        from cai.api import ApiError
        return ApiError
    # tools pulls the environment, so keep it lazy too - cai.tool resolves
    # here the first time an extension's tools/*.py decorates a function.
    if name == "tool":
        from cai.tools import tool
        return tool
    if name == "mcp_server":
        from cai.tools import mcp_server
        return mcp_server
    if name == "ToolsRegistry":
        from cai.tools import ToolsRegistry
        return ToolsRegistry
    # cai.settings: the current Environment's live Settings - the env being
    # load()ed when one is (so an extension/init.py tunes its own env), else the
    # process default's (the one the :config overlay edits).
    if name == "settings":
        from cai.environment import Environment
        return Environment.target().settings
    raise AttributeError(f"module 'cai' has no attribute {name!r}")
