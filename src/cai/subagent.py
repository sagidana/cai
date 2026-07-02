"""subagent: launch / wait / kill sub-agents as bound tools on a parent Agent.

A parent activates these by listing the 'subagents' skill: the Environment's
default agent-tool factory hands the three tools (subagent_tools) to every
Agent at construction, bound to that agent so they read its live state - the
env is the composition root, so the core Agent never imports this module. They
are in-process bound tools, NOT MCP tools - an MCP server runs in its own
subprocess and cannot reach the parent's live Agent.

launch_agent builds a child Agent, serves it with UnixWiredAgent on its own unix
socket, and hands it to one owner thread that drives it to completion and tears
it down. From then on the parent reaches the child only over that socket: it
keeps just the child's id (its socket is AgentsRegistry.sock_path(id)). The child
inherits the parent's api / model / hooks; its tools and skills are reduce-only -
only a subset of the parent's, requested by name.

wait_agent attaches to the child's socket and waits for the run's RESULT;
kill_agent sends the kill control op over the wire. When a child's run finishes,
its owner thread pushes the final result to the parent as a STEER over the
parent's socket just before teardown, so the parent receives it as a user turn
(an idle parent runs on it, a busy one folds it into its current run) even if it
never calls wait_agent. If a wait_agent did collect the result inline, it sets the
child's signal in the shared deliveries map and the owner skips the push, so the
answer is delivered exactly once. A wait_agent that attaches after teardown still
finds the child gone - the live socket result is not separately retained. Ownership
is a plain membership check: an id must be in parent.children for these tools to
act on it."""
from __future__ import annotations

import logging
import os
import socket
import threading
import time

from cai.agent import Agent
from cai.agents_registry import AgentsRegistry
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


def _inherit_tools(parent, names):
    """the parent tools whose names appear in `names`, in request order, in a form
    the child can re-register (the callable for a function tool, the '<mcp>__<tool>'
    name for an MCP tool). reduce-only: a child only ever gets a subset of the
    parent's *active* (selected) tools; any other name is silently dropped."""
    if not names:
        return []
    selected = set(parent.tools)
    chosen = []
    for name in names:
        if name not in selected: continue
        chosen.append(parent.tools_registry.get(name))
    return chosen


def _inherit_skills(parent, names):
    """the requested skills the parent itself has, in request order. reduce-only,
    like the tools: an unknown skill is dropped rather than granted."""
    if not names:
        return []
    allowed = set(parent.skills)
    chosen = []
    for name in names:
        if name not in allowed: continue
        chosen.append(name)
    return chosen


def _child_system_prompt(parent, override):
    """the child's system prompt: the autonomous-sub-agent PREAMBLE, then the
    override if given else the parent's own base prompt."""
    base = override or parent.system_prompt_base
    parts = [PREAMBLE]
    if base:
        parts.append(base)
    return "\n\n".join(parts)


def _unique_name(parent, name):
    """a child name not already taken by a sibling or a live socket; append a
    numeric suffix on collision so each child gets its own socket. checked against
    parent.children (ids never leave the list, so this is the durable guard) and
    against an existing socket file (a live agent elsewhere)."""
    candidate = name
    n = 1
    taken = set(parent.children)
    while candidate in taken or os.path.exists(AgentsRegistry.sock_path(candidate)):
        n += 1
        candidate = f"{name}-{n}"
    return candidate


def _push_result_to_parent(parent_name, child_name, result, delivered=None):
    """deliver the child's final result to the parent as a STEER over the parent's
    socket: an idle parent then runs a turn on it (Agent.steer), and a busy parent
    folds it into its in-flight run. best effort - if the parent isn't served (no
    socket) or has already gone, the push is skipped, the same way the rest of the
    wire path tolerates a missing peer.

    `delivered` is the child's shared signal: a wait_agent that collected the result
    inline sets it, and we must not also push a duplicate. it is checked twice - a
    cheap skip before connecting, and again right before the send, since a wait
    racing on the same RESULT broadcast usually sets it during our connect."""
    if delivered is not None and delivered.is_set():
        return
    try:
        client = connect(AgentsRegistry.sock_path(parent_name))
    except OSError:
        return
    try:
        if delivered is not None and delivered.is_set():
            return
        text = f"Sub-agent '{child_name}' finished. Its result:\n\n{result}"
        Wire(client).send_steer(text)
    except OSError:
        pass
    finally:
        try:
            client.close()
        except OSError:
            pass


def _own_child(parent_name, agent, server, prompt, deliveries=None):
    """own one launched child end to end on a single thread: serve it, submit its
    task over the socket, drain the run (poll-only - streamed events are dropped),
    push its final result to the parent as a steer, then tear it down (the wire,
    the server+socket, the child Agent). a BaseUI answers any wire PROMPT with its
    default, so a child whose hook asks the human never hangs. steering the result
    into the parent lands it as a user turn the parent acts on without polling
    wait_agent; the push is over the parent's socket so the owner thread reaches it
    the same way any other client would. but if a wait_agent collected the result
    inline, it has set this child's signal in `deliveries`, so the push is skipped
    to avoid delivering the same answer twice."""
    delivered = None
    if deliveries is not None:
        delivered = deliveries.get(agent.name)
    server_thread = threading.Thread(target=server.serve,
                                     daemon=True,
                                     name=f"cai-sub-serve-{agent.name}")
    server_thread.start()
    ui = BaseUI()
    client = None
    result = None
    try:
        client = connect(server.path)
        wire = Wire(client)
        wire.send_submit(prompt)
        while True:
            messages = wire.recv()
            if messages is None: break
            done = False
            for msg in messages:
                if wire.answer(msg, ui): continue
                if msg.get("type") != Wire.RESULT: continue
                result = msg.get("text") or ""
                done = True
                break
            if done: break
    except Exception:
        log.exception("sub-agent %r owner failed", agent.name)
    finally:
        if result is not None:
            _push_result_to_parent(parent_name, agent.name, result, delivered)
        if deliveries is not None:
            deliveries.pop(agent.name, None)     # the child is done; drop its signal
        if client is not None:
            try:
                client.close()
            except OSError:
                pass
        server.close()
        try:
            agent.close()
        except Exception:
            log.exception("sub-agent %r: closing child agent failed", agent.name)


_LAUNCH_DOC = """Launch ONE background sub-agent and return its agent_id (its name); call it like launch_agent(prompt="<self-contained task>", name="audit-auth-flow"). Pass arguments as named fields.

    The child shares your working directory but not your conversation, so
    ``prompt`` must be self-contained. ``tools`` and ``skills`` are each a LIST
    of names (a subset of your own); nothing is inherited by default. ``model``
    and ``system_prompt`` override the inherited model / replace the prompt. It
    runs in the background; when it finishes its result is delivered back to you
    automatically as a message, and you can also collect it with
    wait_agent(agent_id=...). The returned name may be de-duplicated, so use the
    one this returns.
    """


def _launch_agent(parent,
                  prompt,
                  name,
                  tools=None,
                  skills=None,
                  model="",
                  system_prompt="",
                  deliveries=None):
    """build a reduce-only child Agent, serve it on its own unix socket, and hand
    it to an owner thread that drives it to completion and tears it down. the
    parent keeps only the child's id. returns at once with the (possibly
    de-duplicated) name. `deliveries` is the shared name->signal map the launch and
    wait tools both close over: this registers the child's signal so a wait_agent
    that collects the result inline can tell the owner thread not to also push it."""
    try:
        name = _unique_name(parent, name)
        child_tools = _inherit_tools(parent, tools)
        child_skills = _inherit_skills(parent, skills)
        child_model = model or parent.model
        child_prompt = _child_system_prompt(parent, system_prompt)
        agent = Agent(name=name,
                      model=child_model,
                      api=parent.api,
                      env=parent.env,
                      system_prompt=child_prompt,
                      tools=child_tools,
                      skills=child_skills,
                      hooks=parent.hooks)
        # no path: the child registers at ~/.config/cai/agents/<name>.sock, the
        # common folder every UnixWiredAgent binds in.
        server = UnixWiredAgent(agent)
        if deliveries is not None:
            deliveries[name] = threading.Event()
        thread = threading.Thread(target=_own_child,
                                  args=(parent.name, agent, server, prompt, deliveries),
                                  daemon=True,
                                  name=f"cai-sub-{name}")
        parent.children.append(name)
        thread.start()
        return (f"Launched sub-agent '{name}'. Collect its result with "
                f"wait_agent('{name}').")
    except Exception as e:
        log.exception("launch_agent failed")
        return f"Error: launch_agent failed: {type(e).__name__}: {e}"


def make_launch_agent(parent, deliveries):
    """build a launch_agent tool bound to its launching agent; the child's
    toolset/skills/model/hooks are read from `parent` at call time. `deliveries` is
    shared with the matching wait_agent so the two can coordinate result delivery."""
    def launch_agent(prompt: str,
                     name: str,
                     tools: list[str] = None,
                     skills: list[str] = None,
                     model: str = "",
                     system_prompt: str = "") -> str:
        return _launch_agent(parent, prompt, name, tools, skills, model, system_prompt, deliveries)
    launch_agent.__doc__ = _LAUNCH_DOC
    return launch_agent


def _send_kill(agent_id):
    """send the kill control op to a child over a short-lived connection. returns
    False when there is no socket to reach (the child already tore down)."""
    try:
        client = connect(AgentsRegistry.sock_path(agent_id))
    except OSError:
        return False
    try:
        Wire(client).control("kill")
        return True
    except OSError:
        return False
    finally:
        try:
            client.close()
        except OSError:
            pass


def _drain(wire, timeout):
    """read a child's stream until its RESULT (returned) or EOF (None), bounded by
    `timeout` seconds total. non-RESULT messages (events, broadcast prompts the
    owner answers) are skipped. returns (result_or_None, timed_out)."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, True
        try:
            wire.channel.settimeout(remaining)
            messages = wire.recv()
        except socket.timeout:
            return None, True
        except OSError:
            return None, False     # socket gone mid-read: the child tore down
        if messages is None:
            return None, False
        for msg in messages:
            if msg.get("type") != Wire.RESULT: continue
            return (msg.get("text") or ""), False


_WAIT_DOC = """Wait for ONE sub-agent and return its answer; call it like wait_agent(agent_id="audit-auth-flow") with the name launch_agent returned. agent_id is a single string, one per call - to wait on several, make several calls.

    Attaches to the running child over its socket. On timeout the sub-agent keeps
    running; call wait_agent again to keep waiting, or pass kill=True to kill it
    when the timeout expires. A child that already finished has torn down, so its
    answer is no longer retrievable here.
    """


def _signal_delivered(deliveries, agent_id):
    """tell the child's owner thread the result has been handed to the parent here,
    as this tool's return value, so it does not also push it as a steer. a no-op
    when there is no shared signal (a direct call outside the bootstrapped tools)."""
    if deliveries is None:
        return
    event = deliveries.get(agent_id)
    if event is not None:
        event.set()


def _wait_agent(parent, agent_id, timeout=30, kill=False, deliveries=None):
    """attach to a child over its socket and wait for the run's RESULT. on timeout
    report it is still running, or (kill=True) kill it and drain what it emits. when
    a result is returned, signal the child's owner thread (via `deliveries`) that it
    has been delivered here, so the owner does not also push it to the parent."""
    if agent_id not in parent.children:
        return f"Error: no sub-agent named '{agent_id}'."
    try:
        client = connect(AgentsRegistry.sock_path(agent_id))
    except OSError:
        return f"Sub-agent '{agent_id}' already finished."
    wire = Wire(client)
    try:
        result, timed_out = _drain(wire, timeout)
        if not timed_out:
            if result is None:
                return f"Sub-agent '{agent_id}' already finished."
            _signal_delivered(deliveries, agent_id)
            return result
        if not kill:
            return (f"Sub-agent '{agent_id}' is still running after {timeout}s; "
                    f"call wait_agent('{agent_id}') again to keep waiting.")
        _send_kill(agent_id)
        result, _timed_out = _drain(wire, timeout)
        if result is None:
            return f"Killed sub-agent '{agent_id}'; it produced no output."
        _signal_delivered(deliveries, agent_id)
        return result
    finally:
        try:
            client.close()
        except OSError:
            pass


def make_wait_agent(parent, deliveries):
    """build a wait_agent tool bound to its launching agent; `deliveries` is shared
    with the matching launch_agent so collecting a result here suppresses the owner
    thread's duplicate push."""
    def wait_agent(agent_id: str, timeout: int = 30, kill: bool = False) -> str:
        return _wait_agent(parent, agent_id, timeout, kill, deliveries)
    wait_agent.__doc__ = _WAIT_DOC
    return wait_agent


_KILL_DOC = """Kill ONE running sub-agent now; call it like kill_agent(agent_id="audit-auth-flow") with the name launch_agent returned. It winds down in the background.

    Returns immediately. A child that already finished is gone.
    """


def _kill_agent(parent, agent_id):
    """retire a running child over the wire; it aborts its run and tears down on
    its own thread."""
    if agent_id not in parent.children:
        return f"Error: no sub-agent named '{agent_id}'."
    if not _send_kill(agent_id):
        return f"Sub-agent '{agent_id}' already finished."
    return f"Killing sub-agent '{agent_id}'."


def make_kill_agent(parent):
    """build a kill_agent tool bound to its launching agent."""
    def kill_agent(agent_id: str) -> str:
        return _kill_agent(parent, agent_id)
    kill_agent.__doc__ = _KILL_DOC
    return kill_agent


def subagent_tools(parent):
    """the three sub-agent tools bound to `parent`, for the Agent to register on
    its tool registry so the model can call them. this is how the tools become
    accessible to the Agent class. launch_agent and wait_agent share one `deliveries`
    map (name -> delivered Event) created here, so a wait that collects a child's
    result inline can tell that child's owner thread not to also push it - exactly
    one delivery either way."""
    deliveries = {}
    tools = []
    tools.append(make_launch_agent(parent, deliveries))
    tools.append(make_wait_agent(parent, deliveries))
    tools.append(make_kill_agent(parent))
    return tools
