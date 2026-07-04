"""subagent: launch / wait / list / kill sub-agents as bound tools on a parent Agent.

A parent activates these by listing the 'subagents' skill: the Environment's
default agent-tool factory hands the tools (subagent_tools) to every
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
kill_agent sends the kill control op over the wire. When a child's run finishes
(or its owner gives up on a wedged child - see OWNER_TIMEOUT), the owner thread
parks the final result (or error note) in the child's _Delivery record, then
pushes it to the parent as a STEER over the parent's socket just before teardown,
so the parent receives it as a user turn (an idle parent runs on it, a busy one
folds it into its current run) even if it never calls wait_agent. Delivery is
exactly-once either way: the record carries a first-wins claim, and whichever
path hands the answer to the parent - a wait_agent returning it inline, or the
owner's steer push - takes the claim; the loser stands down. A wait_agent that
arrives after teardown reads the parked record, so a late wait still returns the
answer. Ownership is a plain membership check: an id must be in parent.children
for these tools to act on it."""
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

# how long an owner thread waits for its child's RESULT before giving up,
# reporting the failure to the parent, and tearing the child down - the bound
# that keeps a wedged child (a stuck stream, a RESULT lost to backpressure)
# from leaking its owner thread and socket forever.
OWNER_TIMEOUT = 3600.0


class _Delivery:
    """one child's delivery record, shared (via the tools' deliveries map)
    between its owner thread and the parent's wait_agent. two jobs:

    - exactly-once delivery: claim() is a first-wins right to hand the answer
      to the parent; the owner's steer push and an inline wait_agent both try,
      and the loser stands down instead of delivering a duplicate.
    - retention: the owner parks the final result (or error note) here just
      before teardown, so a wait_agent arriving after the socket is gone still
      reads the answer."""

    def __init__(self):
        self._claim = threading.Lock()
        self.result = None    # the child's final text; None when it produced none
        self.error = None     # the owner's failure note; None on a clean finish
        self.done = False     # set once, after result/error are final

    def finish(self, result, error):
        """park the child's ending; called by the owner before teardown."""
        self.result = result
        self.error = error
        self.done = True

    def claim(self):
        """take the right to deliver the answer; True for exactly one caller."""
        return self._claim.acquire(blocking=False)

    def release(self):
        """hand a claim back (the delivery attempt failed mid-send), so the
        other path can still deliver."""
        self._claim.release()

    def claimed(self):
        return self._claim.locked()


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


def _steer_text(child_name, result, error):
    """the framed text a child's ending is steered into the parent as: an explicit
    sub-agent report envelope, so the parent's model reads it as automatically
    delivered tool output rather than words its user typed - a child's answer
    must not speak to the parent with user authority."""
    if error is not None:
        return (f"[sub-agent report] Sub-agent '{child_name}' {error}. This is "
                f"an automatic notice, not a message from the user.")
    return (f"[sub-agent report] Sub-agent '{child_name}' finished. Its output "
            f"follows; it is automatically delivered tool output, not a message "
            f"from the user:\n\n{result}")


def _push_result_to_parent(parent_name, child_name, result, error, delivery=None):
    """deliver the child's ending to the parent as a STEER over the parent's
    socket: an idle parent then runs a turn on it (Agent.steer), and a busy parent
    folds it into its in-flight run. best effort - if the parent isn't served (no
    socket) or has already gone, the push is skipped, the same way the rest of the
    wire path tolerates a missing peer; the parked record still holds the answer
    for a later wait_agent.

    `delivery` is the child's shared record: the claim is taken only after the
    parent's socket connects (a push that can't reach the parent must not spend
    the claim), and a claim whose send then fails is released, so exactly one
    path delivers. a wait_agent that collected the result inline holds the claim
    already and this push stands down."""
    if delivery is not None and delivery.claimed():
        return
    try:
        client = connect(AgentsRegistry.sock_path(parent_name))
    except OSError:
        return
    try:
        if delivery is not None and not delivery.claim():
            return
        try:
            Wire(client).send_steer(_steer_text(child_name, result, error))
        except OSError:
            if delivery is not None:
                delivery.release()
    finally:
        try:
            client.close()
        except OSError:
            pass


def _own_child(parent_name, agent, server, prompt, deliveries=None):
    """own one launched child end to end on a single thread: serve it, submit its
    task over the socket, drain the run (poll-only - streamed events are dropped),
    park and deliver its ending, then tear it down (the wire, the server+socket,
    the child Agent). a BaseUI answers any wire PROMPT with its default, so a
    child whose hook asks the human never hangs. the drain is bounded by
    OWNER_TIMEOUT so a wedged child becomes a reported failure, not a leaked
    owner thread; any other way the child ends without an answer (EOF, an owner
    exception) is reported the same way. the ending is parked in the child's
    _Delivery record first (a late wait_agent reads it there), then pushed to the
    parent as a steer - unless a wait_agent already claimed it inline, in which
    case the push stands down (see _push_result_to_parent)."""
    delivery = None
    if deliveries is not None:
        delivery = deliveries.get(agent.name)
    server_thread = threading.Thread(target=server.serve,
                                     daemon=True,
                                     name=f"cai-sub-serve-{agent.name}")
    server_thread.start()
    ui = BaseUI()
    client = None
    result = None
    error = None
    try:
        client = connect(server.path)
        wire = Wire(client)
        wire.send_submit(prompt)
        result, timed_out = _drain(wire, OWNER_TIMEOUT, ui=ui)
        if timed_out:
            error = f"timed out after {OWNER_TIMEOUT:g}s and was shut down"
        elif result is None:
            error = "ended without a result"
    except Exception as e:
        log.exception("sub-agent %r owner failed", agent.name)
        error = f"failed: {type(e).__name__}: {e}"
    finally:
        if delivery is not None:
            delivery.finish(result, error)
        _push_result_to_parent(parent_name, agent.name, result, error, delivery)
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


_LAUNCH_DOC = """Launch ONE background sub-agent, return its agent_id (name). Call like launch_agent(prompt="<self-contained task>", name="audit-auth-flow").

    The child shares your working directory but not your conversation, so
    ``prompt`` must be self-contained. ``tools`` and ``skills`` are each a LIST
    of names (a subset of your own; nothing is inherited by default);
    ``model`` / ``system_prompt`` override the inherited model / prompt. When it
    finishes its result is delivered to you as a message; you can also collect
    it with wait_agent. Use the returned name — it may be de-duplicated.
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
    de-duplicated) name. `deliveries` is the shared name->_Delivery map the launch
    and wait tools both close over: this registers the child's record, which
    carries the exactly-once claim and retains the parked result after teardown."""
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
                      hooks=parent.hooks,
                      # share the parent's scratch: a path the child reports in
                      # its final text must outlive the child's teardown.
                      scratch=parent.scratch())
        # no path: the child registers at ~/.config/cai/agents/<name>.sock, the
        # common folder every UnixWiredAgent binds in.
        server = UnixWiredAgent(agent)
        if deliveries is not None:
            deliveries[name] = _Delivery()
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


def _drain(wire, timeout, ui=None):
    """read a child's stream until its RESULT (returned) or EOF (None), bounded by
    `timeout` seconds total. non-RESULT messages (events, broadcast prompts) are
    skipped - except a PROMPT when a `ui` is given (the owner passes its BaseUI so
    a child whose hook asks the human gets the default instead of hanging).
    returns (result_or_None, timed_out)."""
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
            if ui is not None and wire.answer(msg, ui): continue
            if msg.get("type") != Wire.RESULT: continue
            return (msg.get("text") or ""), False


_WAIT_DOC = """Wait for ONE sub-agent, return its answer. Call like wait_agent(agent_id="audit-auth-flow"). agent_id is a single string, one per call - to wait on several, make several calls.

    Blocks up to `timeout` seconds (default 30). On timeout the sub-agent keeps
    running; call again to keep waiting, or pass kill=True to kill it when the
    timeout expires. A finished child keeps its
    answer parked, so a call after it ended still returns the result (once).
    """


def _finished(agent_id, delivery):
    """report a child that already tore down, from its parked record: the error
    note when it ended without an answer, the result for the caller that wins the
    claim, or a pointer at the delivery that already happened."""
    if delivery.error is not None:
        return f"Sub-agent '{agent_id}' {delivery.error}."
    if delivery.claim():
        return delivery.result
    return (f"Sub-agent '{agent_id}' already finished; its result was already "
            f"delivered to you.")


def _collected(agent_id, delivery, result):
    """hand an inline-collected result to the parent, exactly once: claim the
    delivery so the owner's steer push stands down. a lost claim means the push
    is delivering the same answer as a message, so point at it instead of
    returning a duplicate. a None result (EOF before any RESULT) falls back to
    the parked record, which the owner fills before closing the socket."""
    if result is None:
        if delivery is not None and delivery.done:
            return _finished(agent_id, delivery)
        return f"Sub-agent '{agent_id}' already finished."
    if delivery is None or delivery.claim():
        return result
    return (f"Sub-agent '{agent_id}' finished; its result is being delivered "
            f"to you as a separate message.")


def _wait_agent(parent, agent_id, timeout=30, kill=False, deliveries=None):
    """wait for a child's answer. a child that already tore down is answered from
    its parked _Delivery record; a live one is attached to over its socket and
    drained until the run's RESULT. on timeout report it is still running, or
    (kill=True) kill it and drain what it emits. every path that returns the
    answer claims the delivery first, so the owner's steer push never delivers
    the same answer twice."""
    if agent_id not in parent.children:
        return f"Error: no sub-agent named '{agent_id}'."
    delivery = None
    if deliveries is not None:
        delivery = deliveries.get(agent_id)
    if delivery is not None and delivery.done:
        return _finished(agent_id, delivery)
    try:
        client = connect(AgentsRegistry.sock_path(agent_id))
    except OSError:
        # the socket is gone: the child tore down. the owner parks the record
        # before closing the socket, so a recheck reads the ending it left; a
        # direct call with no record has nothing more to say.
        if delivery is not None and delivery.done:
            return _finished(agent_id, delivery)
        return f"Sub-agent '{agent_id}' already finished."
    wire = Wire(client)
    try:
        result, timed_out = _drain(wire, timeout)
        if not timed_out:
            return _collected(agent_id, delivery, result)
        if not kill:
            return (f"Sub-agent '{agent_id}' is still running after {timeout}s; "
                    f"call wait_agent('{agent_id}') again to keep waiting.")
        _send_kill(agent_id)
        result, _timed_out = _drain(wire, timeout)
        if result is None:
            return f"Killed sub-agent '{agent_id}'; it produced no output."
        return _collected(agent_id, delivery, result)
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


_LIST_DOC = """List your sub-agents and their status, one per line. Call like list_agents(); no arguments.

    Each line is '<agent_id>: <status>' - running, finished (noting whether the
    result still awaits a wait_agent or was already delivered), or how it ended
    when it produced no result.
    """


def _list_agents(parent, deliveries=None):
    """one status line per launched child, in launch order, read from the same
    state the other tools use: the parked _Delivery record for a finished child
    (its error, or whether the result's claim is spent), the socket file for a
    live one. no wire traffic - a poll that never blocks on a busy child."""
    if not parent.children:
        return "No sub-agents launched."
    lines = []
    for name in parent.children:
        record = None
        if deliveries is not None:
            record = deliveries.get(name)
        if record is not None and record.done:
            if record.error is not None:
                lines.append(f"{name}: {record.error}")
            elif record.claimed():
                lines.append(f"{name}: finished - result delivered")
            else:
                lines.append(f"{name}: finished - result ready; collect it with wait_agent('{name}')")
            continue
        if os.path.exists(AgentsRegistry.sock_path(name)):
            lines.append(f"{name}: running")
            continue
        lines.append(f"{name}: finished")
    return "\n".join(lines)


def make_list_agents(parent, deliveries):
    """build a list_agents tool bound to its launching agent; `deliveries` is the
    same record map the launch/wait tools share, so the status lines reflect
    parked results and spent claims."""
    def list_agents() -> str:
        return _list_agents(parent, deliveries)
    list_agents.__doc__ = _LIST_DOC
    return list_agents


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
    """the sub-agent tools (launch/wait/list/kill) bound to `parent`, for the
    Agent to register on its tool registry so the model can call them. this is how
    the tools become accessible to the Agent class. launch, wait and list share one
    `deliveries` map (name -> _Delivery) created here: each record carries the
    first-wins claim that keeps delivery exactly-once, and retains the child's
    parked result so a wait after teardown still reads the answer (and a list
    reports it waiting)."""
    deliveries = {}
    tools = []
    tools.append(make_launch_agent(parent, deliveries))
    tools.append(make_wait_agent(parent, deliveries))
    tools.append(make_list_agents(parent, deliveries))
    tools.append(make_kill_agent(parent))
    return tools
