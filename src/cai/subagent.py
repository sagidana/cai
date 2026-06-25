"""subagent: launch / wait / kill sub-agents as bound tools on a parent Agent.

STUB. Signatures and tool docs match the cai reference; the bodies are
placeholders. See the gap notes (handed back with this file) for what the engine
still needs before these can run.

Target design, using what this rewrite already has:

  launch_agent spawns a child Agent, serves it with UnixWiredAgent on its own
  unix socket, and drives it as a wire client from a background thread - the
  parent connects, SUBMITs the task, and a reader thread drains the child's
  EVENTs and final RESULT into a SubAgent handle. wait_agent blocks on that
  handle; kill_agent stops a running child.

The tools are bound to their launching agent (make_launch_agent / make_wait_agent
/ make_kill_agent): the child's tools/skills/model/hooks are read from the parent
at call time, reduce-only - a child only ever gets a subset of the parent's tools
and skills. These are in-process bound tools, NOT MCP tools: an MCP server runs
in its own subprocess and cannot reach the parent's live Agent, so the builtin
cai__* MCP stubs cannot implement this - these in-process tools replace them."""
from __future__ import annotations

import threading

# the transport a launched child is served on (see the gap notes / _spawn).
from cai.wired_agent import UnixWiredAgent


PREAMBLE = ("You are a sub-agent spawned by a parent agent to complete a specific "
            "task. Work autonomously - you cannot ask the parent questions. Your "
            "final message is the only thing the parent receives, so make it a "
            "complete, self-contained answer.")

_NOT_IMPLEMENTED = "Error: sub-agents are not implemented yet (subagent.py stub)."


class SubAgent:
    """handle for one launched child (stub): its identity plus the slots the
    real engine will fill - the child Agent, the UnixWiredAgent serving it, the
    worker thread, and the done/result/error a wait_agent reads."""

    def __init__(self, id, prompt):
        self.id = id
        self.prompt = prompt
        self.agent = None        # the child Agent, once spawned
        self.server = None       # the UnixWiredAgent serving it
        self.thread = None       # the worker driving the child over the wire
        self.done = threading.Event()
        self.result = None
        self.error = None


def _spawn(parent, handle, prompt, sock_path):
    """serve handle.agent with UnixWiredAgent on sock_path and drive it as a wire
    client from a background thread, collecting events + the final RESULT into
    `handle`. not implemented - this is where UnixWiredAgent gets used."""
    raise NotImplementedError("subagent execution not implemented yet")


_LAUNCH_DOC = """Launch a background sub-agent; name it with descriptive dash-delimited words (e.g. 'audit-auth-flow').

    The child shares your working directory but not your conversation, so
    ``prompt`` must be self-contained. ``tools`` and ``skills`` are each a LIST
    of names (a subset of your own); nothing is inherited by default. ``model``
    and ``system_prompt`` override the inherited model / replace the prompt. It
    runs in the background - collect its result with ``wait_agent``.
    """


def _launch_agent(parent, prompt, name, tools=None, skills=None, model="", system_prompt=""):
    """real launch logic, bound to the launching agent. stub for now."""
    return _NOT_IMPLEMENTED


def make_launch_agent(parent):
    """build a launch_agent tool bound to its launching agent; the child's
    toolset/skills/model/hooks are read from `parent` at call time."""
    def launch_agent(prompt: str,
                     name: str,
                     tools: list[str] = None,
                     skills: list[str] = None,
                     model: str = "",
                     system_prompt: str = "") -> str:
        return _launch_agent(parent, prompt, name, tools, skills, model, system_prompt)
    launch_agent.__doc__ = _LAUNCH_DOC
    return launch_agent


_WAIT_DOC = """Block until the given sub-agent finishes and return its final answer.

    On timeout the sub-agent keeps running; call wait_agent again to keep
    waiting. Pass kill=True to instead kill it when the timeout expires and
    return whatever partial output it had produced.
    """


def _wait_agent(parent, agent_id, timeout=300, kill=False):
    """real wait logic, bound to the launching agent. stub for now."""
    return _NOT_IMPLEMENTED


def make_wait_agent(parent):
    """build a wait_agent tool bound to its launching agent."""
    def wait_agent(agent_id: str, timeout: int = 300, kill: bool = False) -> str:
        return _wait_agent(parent, agent_id, timeout, kill)
    wait_agent.__doc__ = _WAIT_DOC
    return wait_agent


_KILL_DOC = """Kill a running sub-agent now; it winds down in the background.

    Returns immediately. Collect whatever partial output it produced with
    wait_agent('<name>').
    """


def _kill_agent(parent, agent_id):
    """real kill logic, bound to the launching agent. stub for now."""
    return _NOT_IMPLEMENTED


def make_kill_agent(parent):
    """build a kill_agent tool bound to its launching agent."""
    def kill_agent(agent_id: str) -> str:
        return _kill_agent(parent, agent_id)
    kill_agent.__doc__ = _KILL_DOC
    return kill_agent


def subagent_tools(parent):
    """the three sub-agent tools bound to `parent`, ready for the Agent to
    register on its tool registry so the model can call them. this is how the
    tools become accessible to the Agent class."""
    tools = []
    tools.append(make_launch_agent(parent))
    tools.append(make_wait_agent(parent))
    tools.append(make_kill_agent(parent))
    return tools
