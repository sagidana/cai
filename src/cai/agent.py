"""agent: a persistent conversation (Agent) and a one-shot execution (Run).

Agent owns everything: a growing conversation, a model, an api client, the tool
and skill registries, the hooks and ui, and the call_llm orchestration itself.
`run(prompt)` folds the prompt in as a user turn and hands back a handle over the
agent's live `messages` - iterating it streams the answer and evolves the
conversation in place, so the next `run` continues where this one left off. Once
the handle drains, `text` holds the final answer.

Run is syntactic sugar for a one-shot execution: it builds a throwaway Agent from
the params you pass and streams a single turn over the messages you give it. It
has the same surface as the run handle (iterate for Events, then read `text`),
and closing it closes the Agent it owns.

Tools are explicit: `tools=` is a list whose items are Python callables or MCP
tool-name strings ('<mcp>__<tool>'). `skills=` is a list of skill names; each
unions its tools into the registry and folds its prompt into the system prompt.
Only the tools you pass (plus those skills pull in) are sent to the model. MCP
servers are spawned lazily on first use. `hooks=` is a list of (event, fn) pairs
fired through the run (turned into the internal HooksRegistry that call_llm
wants). `ui=` is the human-interaction surface (cai.ui.UI) those hooks prompt
through via HookContext.ui; None means no human is reachable (NULL_UI). Serving/
attach, saving, and cloning are later layers."""
from __future__ import annotations

import copy
import logging
import os
import re
import secrets
import string
import threading

from cai import config
from cai.api import OpenAiApi
from cai.environment import Environment
from cai.events import Event, EventType
from cai.llm import call_llm, SteerQueue
from cai.strict import enforce_strict_format
from cai.tools import ToolsRegistry
from cai.skills import SkillsRegistry
from cai.hooks import HookContext, HookEvent, HooksRegistry


log = logging.getLogger("cai")


class RunInFlight(RuntimeError):
    """a second run was started while one is consuming the conversation - the
    agent serializes runs, so iterate one handle to completion (or wait() it)
    before starting the next. raised at iteration time, when the losing run
    would first touch the conversation."""


def _combine_prompts(base, skills_prompt):
    """join the base system prompt with the activated skills' prompt (base
    first, skills after), or None when both are empty."""
    parts = []
    if base:
        parts.append(base)
    if skills_prompt:
        parts.append(skills_prompt)
    if not parts:
        return None
    return "\n\n".join(parts)


_NAME_ALPHABET = string.digits + "abcdef"


def _new_name():
    """a git-commit-style short id: 7 lowercase hex chars, drawn the way the
    reference names its session files. used when an Agent is built without an
    explicit name."""
    name = ""
    for _ in range(7):
        name += secrets.choice(_NAME_ALPHABET)
    return name


def _trim_dispatch(dispatch, limit):
    """wrap a tool dispatcher so any result longer than `limit` chars is trimmed
    to it, with a marker noting how much was dropped, before it reaches the model
    or the conversation. keeps oversized tool output from swamping the context."""
    def trimmed(name, args):
        result = dispatch(name, args)
        text = str(result)
        if len(text) > limit:
            dropped = len(text) - limit
            return text[:limit] + f"\n…[trimmed {dropped} chars]"
        return result
    return trimmed


def _tool_name(tool):
    """the exposed name of a tools= item: a callable's exposed name (the
    namespaced '<extension>__<name>' stamped by cai.tool, else its __name__),
    or the string (an MCP tool ref) itself."""
    if callable(tool):
        return getattr(tool, "_cai_tool_name", tool.__name__)
    return tool


class RunHandle:
    """the handle Agent.run() returns: a lazy, single-consumption stream over one
    turn on the agent's live conversation. iterate it for Event objects; once it
    drains, `text` holds the final answer and the agent's `messages` have grown,
    so the next run() continues from there. it doesn't start until you iterate
    (or call `wait()`), and only once."""

    def __init__(self, agent, stream, prompt=None, strict_format=None):
        self.agent = agent
        self.messages = agent.messages   # the agent's live conversation
        self.interrupt = agent.interrupt  # the agent's interrupt Event
        self.stream = stream
        self.prompt = prompt
        # a per-run flag, not agent state: it shapes this one answer and is gone
        # by the next run. None means no enforcement (the plain call_llm path).
        self.strict_format = strict_format
        self.text = ""
        self._consumed = False

    def __iter__(self):
        if self._consumed:
            raise RuntimeError("Run already consumed")
        self._consumed = True
        # delegate to the agent's generator; `yield from` forwards every Event
        # and captures the generator's return value (the final text) for us.
        self.text = yield from self.agent._stream(self.stream, self.prompt, self.strict_format)

    def wait(self):
        """drain the run without consuming the events yourself; returns self,
        so `run.wait().text` reads the final answer."""
        for _event in self:
            pass
        return self

    def close(self):
        """nothing to close: the agent owns its tool registry for its whole life
        and tears it down in Agent.close()."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Agent:
    def __init__(self,
                 *,
                 name=None,
                 model=None,
                 api=None,
                 env=None,
                 system_prompt=None,
                 tools=None,
                 skills=None,
                 hooks=None,
                 ui=None,
                 interrupt=None,
                 reasoning_effort=None,
                 temperature=None,
                 max_steps=None,
                 tool_result_max_chars=None,
                 stream=True):
        # only read config/key for a default the caller didn't supply - a child
        # agent given both a model and an api never touches the disk.
        cfg = None
        if model is None or api is None:
            cfg = config.load_config()
        if model is None: model = cfg.model
        if name is None: name = _new_name()
        if api is None:
            api = OpenAiApi(cfg.base_url,
                            config.load_api_key(),
                            ssl_verify=config.load_optional("ssl_verify", True))

        self.name = name
        self.model = model
        self.api = api
        # the install catalogue this agent resolves tools/skills/hooks against:
        # the caller's env, else the process default (empty until a frontend
        # loads it). a sub-agent / clone inherits its parent's explicitly.
        self.env = env or Environment.default()
        self._system_prompt = system_prompt   # base; combined with skills on demand
        self._hooks = hooks   # [(event, fn), ...] or None; turned into a HooksRegistry per run
        self._ui = ui         # UI hooks prompt through; None -> NULL_UI in call_llm
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_steps = max_steps
        # cap on a single tool result's length; oversized output is trimmed to it
        # before the model (and the conversation) sees it. None disables the cap.
        # an explicit param, not an ambient read - the CLI sources it from
        # cai.settings, an SDK caller passes its own (or leaves it off).
        self.tool_result_max_chars = tool_result_max_chars
        self.stream = stream
        if interrupt is None: interrupt = threading.Event()
        self.interrupt = interrupt
        self._killed = threading.Event()
        self._steer = SteerQueue()
        # held while a run streams (see _stream): the one conversation is only
        # ever consumed by one run at a time. a second concurrent run raises
        # RunInFlight; steer() reads locked() to tell an in-flight run (which
        # drains the steer queue on its own) from an idle agent (which needs a
        # fresh run started).
        self._run_lock = threading.Lock()
        self.messages = []
        self.children = []   # ids of the sub-agents launched this session
        self.tools_registry = ToolsRegistry(self.env)

        # registers the env's agent-bound tools (the sub-agent trio by default).
        # override=True so these bind to *this* agent even if a tool of the same
        # name was inherited from a parent (a child's launch_agent must drive the
        # child, not the parent). the env is the composition root here: the core
        # Agent never names the feature layers above it.
        for tool in self.env.agent_tools(self):
            self.tools_registry.register(tool, override=True)

        for tool in (tools or []): self.tools_registry.select(tool)

        self.skills_registry = SkillsRegistry.for_skills(skills,
                                                         tools_registry=self.tools_registry,
                                                         env=self.env)

    @property
    def system_prompt(self):
        """the base prompt plus the currently-active skills' prompt, rebuilt on
        each read so it reflects skills added or removed since bootstrap."""
        return _combine_prompts(self._system_prompt, self.skills_registry.system_prompt)

    @property
    def tools(self):
        """the names of the active (selected) tools - the subset sent to the
        model, read straight from the registry (the single source of truth)."""
        return self.tools_registry.selected()

    @property
    def skills(self):
        """the active skill names, read straight from the registry (the single
        source of truth)."""
        return self.skills_registry.names()

    def get_selected_tools(self):
        """the active (selected) tool names - what set_selected_tools controls."""
        return self.tools

    def _hooks_registry(self):
        """the registry a run fires: the env's hooks (cai.hook, extensions)
        first, then this agent's own hooks= list on top, composed fresh per use
        so hooks added since bootstrap are reflected."""
        hooks = list(self.env.hooks())
        hooks.extend(self._hooks or [])
        return HooksRegistry.from_list(hooks)

    def get_available_tools(self):
        """every tool the agent could select: the env's catalogue plus the
        tools already registered on this agent (its own function tools included)."""
        names = set(self.env.available_tools())
        for name in self.tools_registry.names():
            names.add(name)
        return sorted(names)

    def set_selected_tools(self, names):
        """set the active tools to `names`, diffing against the current selection:
        a tool no longer listed is deselected (it stays registered), and a newly
        listed one is selected. selecting is best effort: a name that can't be
        resolved to a tool this agent can reach (a function tool, which can't be
        rebuilt from a name) is skipped with a warning rather than failing."""
        want = set(names or [])
        for name in self.tools_registry.selected():
            if name in want: continue
            self.tools_registry.deselect(name)
        for name in want:
            try:
                self.tools_registry.select(name)
            except ValueError:
                log.warning("tool %r is not available to this agent; skipping", name)

    def get_selected_skills(self):
        """the active skill names - what set_selected_skills controls."""
        return self.skills

    def get_available_skills(self):
        """every skill the agent could activate: the env's catalogue plus the
        ones already active on this agent."""
        names = set(self.env.available_skills())
        for name in self.skills_registry.names():
            names.add(name)
        return sorted(names)

    def set_selected_skills(self, names):
        """set the active skills to `names`, diffing against the registry: a skill
        no longer listed is removed (along with the tools it pulled in) and a
        newly listed one added. the combined system prompt follows on next read."""
        want = set(names or [])
        for name in self.skills_registry.names():
            if name in want: continue
            self.skills_registry.remove(name)
        for name in want:
            self.skills_registry.add(name)

    def set_model(self, model):
        """switch the model used for the next run; an empty value is ignored."""
        if not model: return
        self.model = model

    def get_messages(self):
        """the agent's live conversation list."""
        return self.messages

    def _fire_messages_event(self, event):
        """fire `event` for a conversation change made outside a run, so hooks that
        watch the conversation (the same ones call_llm fires mid-run) see it too.
        set_messages fires MESSAGES_MUTATED (an in-place edit); load fires
        MESSAGES_LOADED instead, so a listener like autosave can tell a fresh load
        - whose messages came straight off disk - apart from an edit and not write
        it back. the agent rides in ctx.data the way call_llm passes it via
        hooks_data, so a hook reaches it uniformly however it was triggered."""
        hooks = self._hooks_registry()
        ctx = HookContext(event=event,
                          messages=self.messages,
                          model=self.model,
                          ui=self._ui,
                          data={"agent": self})
        hooks.fire(event, ctx)

    def set_messages(self, messages):
        """replace the agent's conversation; the next run() continues from it."""
        if messages is None: messages = []
        self.messages = messages
        self._fire_messages_event(HookEvent.MESSAGES_MUTATED)

    def set_system_prompt_base(self, base):
        """replace the base system prompt in memory; the system_prompt property
        recomposes it with the active skills on the next read, so this is all the
        live agent needs. the change is never written to disk."""
        self._system_prompt = base

    @property
    def system_prompt_base(self):
        """the user-supplied base prompt (no skills folded in) - what
        set_system_prompt_base writes and a child/clone inherits."""
        return self._system_prompt

    @property
    def hooks(self):
        """the caller-supplied [(event, fn), ...] hooks (None when none) - what
        a child/clone inherits; the env's hooks ride along via env, not here."""
        return self._hooks

    def set_ui(self, ui):
        """replace the human-interaction surface hooks prompt through (ctx.ui).
        the serving layer routes prompts over the wire by installing a WireUI
        here."""
        self._ui = ui

    def clone(self, name=None, ui=None):
        """build a new Agent carrying this one's state - a deep copy of the
        conversation plus the model, base prompt, active tools and skills, hooks,
        and run parameters - so an SDK caller can fork a conversation and drive
        the branch independently without disturbing this agent.

        the clone is a separate agent: it gets its own name (a fresh id unless
        one is given), its own interrupt Event, and an empty children list, so
        its autosave file and any sub-agents it launches are its own. function
        tools carry over by callable and MCP tools by name (the clone spawns its
        own servers on first use), the way a sub-agent inherits a parent's tools.
        the messages are assigned directly rather than through set_messages so no
        MESSAGES_MUTATED hook fires on the fresh clone - the branch persists on
        its first run, not at the moment it is forked.

        the ui is NOT carried over (it defaults to None - no human reachable):
        the original's ui is mutable interactive state, and on a served agent it
        is a WireUI bound to that agent's own server, so sharing it would cross
        the fork's prompts onto the parent's clients. pass ui= to give the branch
        its own surface (the serving layer sets one of its own when it wraps the
        clone)."""
        tools = []
        for selected in self.tools_registry.selected():
            tool = self.tools_registry.get(selected)
            if tool is None: continue
            tools.append(tool)
        clone = Agent(name=name,
                      model=self.model,
                      api=self.api,
                      env=self.env,
                      system_prompt=self._system_prompt,
                      tools=tools,
                      skills=self.skills_registry.names(),
                      hooks=self._hooks,
                      ui=ui,
                      reasoning_effort=self.reasoning_effort,
                      temperature=self.temperature,
                      max_steps=self.max_steps,
                      stream=self.stream)
        clone.messages = copy.deepcopy(self.messages)
        return clone

    def save(self, path=None):
        """persist the conversation + settings to a .flow file (see session.py
        for the format). path defaults to '<name>.flow' in the sessions dir
        (named by the agent's id, the reference's convention); the path written
        is returned. tool callables can't be serialised - their names are saved,
        so the same callables must be passed to the agent for a loaded session to
        dispatch them again."""
        from cai.session import SessionsRegistry

        if path is None:
            path = SessionsRegistry.session_path(self.name)
        # the selected (active) tool names are saved as-is, function tools
        # included. a name that names no MCP tool (a function tool) simply won't
        # resolve when a different agent loads the flow - set_tools skips it with
        # a warning. registered-but-unselected tools are not persisted.
        # the ids of the sub-agents this agent launched (each saved under its own
        # '<id>.flow'), so the :sessions picker can nest them under this session.
        children = list(self.children)
        payload = SessionsRegistry.flow_payload(list(self.messages),
                                                self.system_prompt,
                                                self._system_prompt,
                                                self.skills_registry.names(),
                                                self.tools_registry.selected(),
                                                self.model,
                                                reasoning_effort=self.reasoning_effort,
                                                temperature=self.temperature,
                                                max_steps=self.max_steps,
                                                children=children)
        SessionsRegistry.write_flow(path, payload)
        return path

    def load(self, path):
        """replace the conversation + settings in place from a .flow file, so a
        served agent keeps its live aliases. the agent also adopts the loaded
        file's stem as its name, so autosave writes back to that file instead of
        clobbering this agent's own '<name>.flow'. the stored leading system
        message is dropped and the prompt is re-derived from base + skills. tools
        restore by name only (function tools must be re-supplied via the
        constructor). returns the path loaded."""
        from cai.session import SessionsRegistry

        payload = SessionsRegistry.read_flow(path)
        # retarget name -> the loaded file, so autosave follows it.
        self.name = os.path.splitext(os.path.basename(path))[0]
        # restore the flow's sub-agent ids so re-saving keeps the nesting.
        self.children = list(payload.get("children") or [])
        settings = payload.get("settings") or {}
        messages = list(payload.get("messages") or [])
        # the agent owns the composed prompt; drop a stored leading system msg.
        if messages and messages[0].get("role") == "system":
            messages = messages[1:]

        self._system_prompt = settings.get("system_prompt_base")
        self.set_selected_skills(settings.get("skills") or [])
        self.set_selected_tools(settings.get("selected_tools") or [])
        self.set_model(settings.get("model"))
        self.reasoning_effort = settings.get("reasoning_effort")
        self.temperature = settings.get("temperature")
        self.max_steps = settings.get("max_steps")
        # mutate in place so any external alias keeps pointing at live state.
        self.messages[:] = messages
        self._fire_messages_event(HookEvent.MESSAGES_LOADED)
        return path

    def stop(self):
        """request the in-flight run abort at the next safe boundary (between
        turns, or between streamed chunks). safe to call from another thread."""
        self.interrupt.set()

    def kill(self):
        """retire the agent: abort the in-flight run and refuse further ones.
        unlike stop(), this is permanent. safe to call from another thread."""
        self._killed.set()
        self.interrupt.set()

    @property
    def killed(self):
        return self._killed.is_set()

    def steer(self, text, run_on_idle=True):
        """queue `text` for the conversation; return whether its delivery is
        arranged here.

        the text always goes onto the steer queue - the one path a steer takes
        into the conversation. call_llm drains the queue at every turn boundary
        (the first included), so:

        - while a run is consuming, it folds the text in at its next boundary;
          returns True.
        - while idle with run_on_idle True, a fresh run(None) is driven here (on
          the calling thread) and drains it as its first turn; returns True. if
          another run wins the race to start, that run delivers it instead -
          the push-then-check order is what closes the window in which two
          idle steers could each start a run of their own.
        - while idle with run_on_idle False, nothing is started; returns False
          so the caller (a serving WiredAgent) can drive the turn its own way -
          the text stays queued for whichever run comes next.

        a text pushed after a run's final drain point stays queued and is
        delivered at the next run's first turn boundary. safe to call from any
        thread."""
        self._steer.push(text)
        if self._run_lock.locked():
            return True
        if not run_on_idle:
            return False
        try:
            self.run(None).wait()
        except RunInFlight:
            # another run won the start race; it drains the queue for us.
            pass
        return True

    def steer_pending(self):
        """whether steered texts are queued with no run consuming them - i.e. a
        run(None) is needed for them to be delivered. a serving WiredAgent reads
        this to skip a queued drain-run another run already covered."""
        if self._run_lock.locked():
            return False
        return self._steer.pending()

    def _stream(self, stream, prompt=None, strict_format=None):
        """the orchestration: stream one call_llm turn over the live conversation.
        yields Event objects and returns the final answer text. combines the
        system prompt and translates the hooks list here (not at construction) so
        skills and hooks added since are reflected. holds the run lock for the
        life of the stream - the conversation is only ever consumed by one run,
        so a concurrent one raises RunInFlight (loudly, where today's silent
        interleaving would corrupt the messages) and a concurrent steer folds in
        rather than starting a second run (released in finally, however the
        stream ends).

        strict_format, when set, wraps the run in cai.strict: the answer is
        validated and the turn reissued until it matches. enforcement pins
        temperature to 0 and forces non-streaming (validation needs the whole
        answer before it can pass judgement)."""
        if not self._run_lock.acquire(blocking=False):
            raise RunInFlight(f"agent {self.name!r}: a run is already consuming the conversation")
        try:
            # the prompt lands only under the lock: a run that loses the race
            # raises above without leaving its user turn in the conversation.
            if prompt is not None:
                self.messages.append({"role": "user", "content": prompt})
                yield Event(type=EventType.USER, text=prompt)
            skills_prompt = self.skills_registry.system_prompt
            system_prompt = _combine_prompts(self._system_prompt, skills_prompt)
            schemas = self.tools_registry.tools
            dispatch = self.tools_registry.dispatch
            if self.tool_result_max_chars:
                dispatch = _trim_dispatch(dispatch, self.tool_result_max_chars)
            # the one place the env hooks and the public [(event, fn), ...] list
            # become the internal HooksRegistry that call_llm wants.
            hooks_registry = self._hooks_registry()
            if strict_format:
                def make_stream(strict_system_prompt):
                    return call_llm(self.messages,
                                    self.model,
                                    self.api,
                                    system_prompt=strict_system_prompt,
                                    tools=schemas,
                                    tools_dispatch=dispatch,
                                    hooks=hooks_registry,
                                    ui=self._ui,
                                    interrupt=self.interrupt,
                                    steer=self._steer.drain,
                                    reasoning_effort=self.reasoning_effort,
                                    temperature=0,
                                    max_steps=self.max_steps,
                                    stream=False,
                                    hooks_data={"agent": self})
                text = yield from enforce_strict_format(make_stream,
                                                        strict_format,
                                                        system_prompt,
                                                        self.messages,
                                                        interrupt=self.interrupt)
                return text
            text = yield from call_llm(self.messages,
                                       self.model,
                                       self.api,
                                       system_prompt=system_prompt,
                                       tools=schemas,
                                       tools_dispatch=dispatch,
                                       hooks=hooks_registry,
                                       ui=self._ui,
                                       interrupt=self.interrupt,
                                       steer=self._steer.drain,
                                       reasoning_effort=self.reasoning_effort,
                                       temperature=self.temperature,
                                       max_steps=self.max_steps,
                                       stream=stream,
                                       hooks_data={"agent": self})
            return text
        finally:
            self._run_lock.release()

    def run(self, prompt=None, *, strict_format=None):
        """return a handle over the agent's live conversation; iterating it
        appends `prompt` as a user turn (if given) and streams events. the
        append happens at iteration, under the run lock (see _stream), so a run
        that loses to a concurrent one raises RunInFlight without touching the
        conversation. `messages` keeps growing, so the next run continues this
        conversation.

        strict_format is a per-run flag (not agent state): it constrains only this
        answer's shape - 'json', 'regex:<pat>' or 'regex-each-line:<pat>'."""
        self.interrupt.clear()
        if self._killed.is_set():
            # a killed agent never runs again: re-arm the interrupt so the run
            # winds down at once instead of calling the model.
            self.interrupt.set()
        return RunHandle(self, self.stream, prompt, strict_format=strict_format)

    def gate(self, options, prompt, *, system_prompt=None):
        """single-turn quality gate: ask `prompt` and get back exactly one of
        `options`. runs in isolation - a throwaway conversation on this agent's
        model and api, so the agent's own history is untouched - under a strict
        gate persona (override with system_prompt). a regex strict_format pins the
        reply to one option, so the stripped result is guaranteed to be one of
        them. built on run(strict_format=...): the regex is just '^(a|b|...)$'."""
        escaped = []
        for option in options:
            escaped.append(re.escape(option))
        pattern = "|".join(escaped)

        if system_prompt is None:
            quoted = []
            for option in options:
                quoted.append(f"'{option}'")
            system_prompt = "You are a strict quality gate. Answer only " + ", ".join(quoted) + "."

        messages = [{"role": "user", "content": prompt}]
        run = Run(messages,
                  self.model,
                  self.api,
                  env=self.env,
                  system_prompt=system_prompt,
                  strict_format=f"regex:^({pattern})$")
        try:
            run.wait()
            return run.text.strip()
        finally:
            run.close()

    def close(self):
        """close the tool registry, terminating any live MCP server connections
        this agent spawned at bootstrap."""
        self.tools_registry.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Run(RunHandle):
    """one-shot sugar: build a throwaway Agent from these params and stream a
    single turn over `messages` (which must already include the prompt turn). the
    surface is the run handle's - iterate for Events, then read `text` - and
    closing it closes the Agent it owns (and the MCP servers that agent spawned).

    model and api default from config when omitted, so `Run(messages=[...])` is a
    valid one-liner (used by the messages-overlay rewrite)."""

    def __init__(self,
                 messages,
                 model=None,
                 api=None,
                 *,
                 env=None,
                 system_prompt=None,
                 tools=None,
                 skills=None,
                 hooks=None,
                 ui=None,
                 interrupt=None,
                 reasoning_effort=None,
                 temperature=None,
                 max_steps=None,
                 tool_result_max_chars=None,
                 stream=True,
                 strict_format=None):
        agent = Agent(model=model,
                      api=api,
                      env=env,
                      system_prompt=system_prompt,
                      tools=tools,
                      skills=skills,
                      hooks=hooks,
                      ui=ui,
                      interrupt=interrupt,
                      reasoning_effort=reasoning_effort,
                      temperature=temperature,
                      max_steps=max_steps,
                      tool_result_max_chars=tool_result_max_chars,
                      stream=stream)
        agent.set_messages(messages)
        # no prompt to fold in: `messages` is already the complete conversation,
        # so stream straight over it (Agent.run's prompt-append is for the
        # persistent path). strict_format rides the run handle, not the Agent -
        # it shapes this one turn, never the agent's persistent settings.
        super().__init__(agent, stream, strict_format=strict_format)

    def close(self):
        """close the Agent this Run built (and owns), tearing down its tool
        registry and any MCP servers it spawned."""
        self.agent.close()
