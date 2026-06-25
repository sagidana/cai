"""subagent: launch / wait / kill sub-agents as bound tools on a parent Agent.

A parent activates these by listing the 'subagents' skill: Agent bootstraps the
three tools at construction (subagent_tools), bound to the parent so they read
its live state. They are in-process bound tools, NOT MCP tools - an MCP server
runs in its own subprocess and cannot reach the parent's live Agent.

launch_agent spawns a child Agent and serves it with UnixWiredAgent on its own
unix socket; a background driver thread connects as a wire client, SUBMITs the
task, drains the run, and collects the final RESULT into a SubAgent handle. The
child inherits the parent's api / model / hooks; its tools and skills are
reduce-only - only a subset of the parent's, requested by name. wait_agent polls
that handle for the final answer (the only supported channel for now); kill_agent
stops a running child, which winds it down and frees its socket.

The socket is opened at launch and closed at the child's death, when the driver
tears the child down (closes the wire, the server, and the child Agent)."""
from __future__ import annotations

import os
import shutil
import logging
import tempfile
import threading

from cai.agent import Agent, _tool_name
from cai.channel import connect
from cai.ui import BaseUI
from cai.wire import Wire
from cai.wired_agent import UnixWiredAgent


log = logging.getLogger("cai")

# the skill name a parent lists to get these tools bootstrapped (see Agent).
SUBAGENT_SKILL = "subagents"

PREAMBLE = ("You are a sub-agent spawned by a parent agent to complete a specific "
            "task. Work autonomously - you cannot ask the parent questions. Your "
            "final message is the only thing the parent receives, so make it a "
            "complete, self-contained answer.")


class SubAgent:
    """handle for one launched child: its identity plus the live machinery the
    driver fills - the child Agent, the UnixWiredAgent serving it, the server and
    driver threads, and the done/result/error a wait_agent reads."""

    def __init__(self, id, prompt):
        self.id = id
        self.prompt = prompt
        self.agent = None         # the child Agent
        self.server = None        # the UnixWiredAgent serving it
        self.server_thread = None # thread running server.serve()
        self.thread = None        # the driver thread (wire client)
        self.sock_dir = None      # temp dir holding the unix socket file
        self.done = threading.Event()
        self.result = None
        self.error = None


def _inherit_tools(parent, names):
    """the parent tools whose names appear in `names`, in request order. reduce-
    only: a child only ever gets a subset of the parent's tools, and a name the
    parent does not have is silently dropped."""
    if not names:
        return []
    by_name = {}
    for tool in parent.get_tools():
        by_name[_tool_name(tool)] = tool
    chosen = []
    for name in names:
        if name not in by_name: continue
        chosen.append(by_name[name])
    return chosen


def _inherit_skills(parent, names):
    """the requested skills the parent itself has, in request order. reduce-only,
    like the tools: an unknown skill is dropped rather than granted."""
    if not names:
        return []
    allowed = set(parent.get_skills())
    chosen = []
    for name in names:
        if name not in allowed: continue
        chosen.append(name)
    return chosen


def _child_system_prompt(parent, override):
    """the child's system prompt: the autonomous-sub-agent PREAMBLE, then the
    override if given else the parent's own base prompt."""
    base = override or parent._system_prompt
    parts = [PREAMBLE]
    if base:
        parts.append(base)
    return "\n\n".join(parts)


def _teardown(handle, client):
    """close everything this child owned: the wire client, the server (which
    unlinks the socket and ends the serve loop), the child Agent (its MCP
    servers), and the temp dir holding the socket."""
    if client is not None:
        try:
            client.close()
        except OSError:
            pass
    if handle.server is not None:
        handle.server.close()
    if handle.agent is not None:
        try:
            handle.agent.close()
        except Exception:
            log.exception("sub-agent %r: closing child agent failed", handle.id)
    if handle.sock_dir is not None:
        shutil.rmtree(handle.sock_dir, ignore_errors=True)


def _drive(handle):
    """connect to the served child, submit its task, drain the run dropping the
    streamed events (poll-only for now), and collect the final RESULT into the
    handle - then tear the child down. runs on its own thread so launch_agent
    returns at once. a BaseUI answers any wire PROMPT with its default, so a
    child whose hook asks the human never hangs."""
    ui = BaseUI()
    client = None
    try:
        client = connect(handle.server.path)
        wire = Wire(client)
        wire.send_submit(handle.prompt)
        result = None
        while result is None:
            messages = wire.recv()
            if messages is None: break
            for msg in messages:
                if wire.answer(msg, ui): continue
                if msg.get("type") != Wire.RESULT: continue
                result = msg.get("text") or ""
                break
        if result is None:
            result = ""
        handle.result = result
    except Exception as e:
        log.exception("sub-agent %r driver failed", handle.id)
        handle.error = f"Error: {type(e).__name__}: {e}"
    finally:
        _teardown(handle, client)
        handle.done.set()


def _find(parent, agent_id):
    for handle in parent.children:
        if handle.id == agent_id:
            return handle
    return None


_LAUNCH_DOC = """Launch a background sub-agent; name it with descriptive dash-delimited words (e.g. 'audit-auth-flow').

    The child shares your working directory but not your conversation, so
    ``prompt`` must be self-contained. ``tools`` and ``skills`` are each a LIST
    of names (a subset of your own); nothing is inherited by default. ``model``
    and ``system_prompt`` override the inherited model / replace the prompt. It
    runs in the background - collect its result with ``wait_agent``.
    """


def _launch_agent(parent, prompt, name, tools=None, skills=None, model="", system_prompt=""):
    """build a reduce-only child Agent, serve it over a unix socket, and drive it
    from a background thread; return at once with the child's name."""
    try:
        child_tools = _inherit_tools(parent, tools)
        child_skills = _inherit_skills(parent, skills)
        child_model = model or parent.model
        child_prompt = _child_system_prompt(parent, system_prompt)
        handle = SubAgent(name, prompt)
        handle.agent = Agent(name=name,
                             model=child_model,
                             api=parent.api,
                             system_prompt=child_prompt,
                             tools=child_tools,
                             skills=child_skills,
                             hooks=parent._hooks)
        handle.sock_dir = tempfile.mkdtemp(prefix="cai-sub-")
        sock_path = os.path.join(handle.sock_dir, "sock")
        handle.server = UnixWiredAgent(handle.agent, sock_path)
        handle.server_thread = threading.Thread(target=handle.server.serve,
                                                 daemon=True,
                                                 name=f"cai-sub-serve-{name}")
        handle.thread = threading.Thread(target=_drive,
                                         args=(handle,),
                                         daemon=True,
                                         name=f"cai-sub-{name}")
        parent.children.append(handle)
        handle.server_thread.start()
        handle.thread.start()
        return f"Launched sub-agent '{name}'. Collect its result with wait_agent('{name}')."
    except Exception as e:
        log.exception("launch_agent failed")
        return f"Error: launch_agent failed: {type(e).__name__}: {e}"


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


def _wait_agent(parent, agent_id, timeout=30, kill=False):
    """poll a child handle for its final answer; on timeout report it is still
    running, or (kill=True) kill it and collect whatever partial output it had."""
    handle = _find(parent, agent_id)
    if handle is None:
        return f"Error: no sub-agent named '{agent_id}'."
    finished = handle.done.wait(timeout)
    if not finished:
        if not kill:
            return (f"Sub-agent '{agent_id}' is still running after {timeout}s; "
                    f"call wait_agent('{agent_id}') again to keep waiting.")
        handle.agent.kill()
        handle.done.wait()
    if handle.error is not None:
        return handle.error
    return handle.result or ""


def make_wait_agent(parent):
    """build a wait_agent tool bound to its launching agent."""
    def wait_agent(agent_id: str, timeout: int = 30, kill: bool = False) -> str:
        return _wait_agent(parent, agent_id, timeout, kill)
    wait_agent.__doc__ = _WAIT_DOC
    return wait_agent


_KILL_DOC = """Kill a running sub-agent now; it winds down in the background.

    Returns immediately. Collect whatever partial output it produced with
    wait_agent('<name>').
    """


def _kill_agent(parent, agent_id):
    """retire a running child; it winds down on its own thread, the driver then
    collects its partial output and frees the socket."""
    handle = _find(parent, agent_id)
    if handle is None:
        return f"Error: no sub-agent named '{agent_id}'."
    if handle.done.is_set():
        return f"Sub-agent '{agent_id}' already finished."
    handle.agent.kill()
    return (f"Killing sub-agent '{agent_id}'. Collect any partial output with "
            f"wait_agent('{agent_id}').")


def make_kill_agent(parent):
    """build a kill_agent tool bound to its launching agent."""
    def kill_agent(agent_id: str) -> str:
        return _kill_agent(parent, agent_id)
    kill_agent.__doc__ = _KILL_DOC
    return kill_agent


def subagent_tools(parent):
    """the three sub-agent tools bound to `parent`, for the Agent to register on
    its tool registry so the model can call them. this is how the tools become
    accessible to the Agent class."""
    tools = []
    tools.append(make_launch_agent(parent))
    tools.append(make_wait_agent(parent))
    tools.append(make_kill_agent(parent))
    return tools
