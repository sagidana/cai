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

import secrets
import string
import threading

from cai import config
from cai.api import OpenAiApi
from cai.llm import call_llm, SteerQueue
from cai.tools import ToolRegistry
from cai.skills import SkillsRegistry
from cai.hooks import HooksRegistry


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


def _tool_name(tool):
    """the exposed name of a tools= item: a callable's __name__, else the string
    (an MCP tool ref) itself."""
    if callable(tool):
        return tool.__name__
    return tool


class RunHandle:
    """the handle Agent.run() returns: a lazy, single-consumption stream over one
    turn on the agent's live conversation. iterate it for Event objects; once it
    drains, `text` holds the final answer and the agent's `messages` have grown,
    so the next run() continues from there. it doesn't start until you iterate
    (or call `wait()`), and only once."""

    def __init__(self, agent, stream):
        self.agent = agent
        self.messages = agent.messages   # the agent's live conversation
        self.interrupt = agent.interrupt  # the agent's interrupt Event
        self.stream = stream
        self.text = ""
        self._consumed = False

    def __iter__(self):
        if self._consumed:
            raise RuntimeError("Run already consumed")
        self._consumed = True
        # delegate to the agent's generator; `yield from` forwards every Event
        # and captures the generator's return value (the final text) for us.
        self.text = yield from self.agent._stream(self.stream)

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
                 system_prompt=None,
                 tools=None,
                 skills=None,
                 hooks=None,
                 ui=None,
                 interrupt=None,
                 reasoning_effort=None,
                 temperature=None,
                 max_steps=None,
                 stream=True):
        # only read config/key for a default the caller didn't supply - a child
        # agent given both a model and an api never touches the disk.
        cfg = None
        if model is None or api is None:
            cfg = config.load_config()
        if model is None: model = cfg.model
        if name is None: name = _new_name()
        if api is None: api = OpenAiApi(cfg.base_url, config.load_api_key())

        self.name = name
        self.model = model
        self.api = api
        self._system_prompt = system_prompt   # base; combined with skills on demand
        self._tools = tools or []
        self._skills = skills or []
        # build the registries once at bootstrap so every run() reuses them (and
        # the MCP servers they spawned); set_tools/set_skills mutate them in place.
        self.tools_registry = ToolRegistry.for_tools(self._tools)
        self.skills_registry = SkillsRegistry.for_skills(self._skills, tools_registry=self.tools_registry)
        self._hooks = hooks   # [(event, fn), ...] or None; turned into a HooksRegistry per run
        self._ui = ui         # UI hooks prompt through; None -> NULL_UI in call_llm
        # per-run defaults forwarded to call_llm; None -> model/library default.
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_steps = max_steps
        self.stream = stream
        # set from another thread (stop()) to abort the in-flight run; each run()
        # clears it so a fresh run starts un-interrupted.
        if interrupt is None: interrupt = threading.Event()
        self.interrupt = interrupt
        # set by kill(): a harder stop that also retires the agent - it aborts
        # the in-flight run and refuses further ones. never cleared.
        self._killed = threading.Event()
        # messages pushed from another thread (steer()) and folded into the
        # in-flight (or next) run as user turns at each turn boundary.
        self._steer = SteerQueue()
        self.messages = []
        # SubAgent handles for children launched via the sub-agent tools - both
        # active and dead; they are never pruned (a wait_agent can still read a
        # finished child's result).
        self.children = []
        # a child agent is an in-process layer that needs a live reference to
        # this Agent, so the 'subagents' skill's tools are bound here at
        # construction rather than spawned as an MCP subprocess (which could not
        # reach us). lazy import: subagent imports Agent in turn.
        if "subagents" in self._skills:
            from cai.subagent import subagent_tools
            for tool in subagent_tools(self):
                self.tools_registry.add(tool)

    @property
    def system_prompt(self):
        """the base prompt plus the currently-active skills' prompt, rebuilt on
        each read so it reflects skills added or removed since bootstrap."""
        return _combine_prompts(self._system_prompt, self.skills_registry.system_prompt)

    def get_tools(self):
        return self._tools

    def set_tools(self, tools):
        """replace the agent's tools, updating the live registry in place."""
        if tools is None: tools = []
        new_names = set()
        for tool in tools:
            new_names.add(_tool_name(tool))
        for tool in self._tools:
            name = _tool_name(tool)
            if name in new_names: continue
            self.tools_registry.remove(name)
        for tool in tools:
            self.tools_registry.add(tool)
        self._tools = tools

    def get_skills(self):
        return self._skills

    def set_skills(self, skills):
        """replace the agent's skills. the skill layer is torn down and rebuilt
        so dependencies and shared tools resolve correctly; base tools and the
        cached MCP servers are left in place. tools and system prompt follow."""
        if skills is None: skills = []
        self.skills_registry.clear()
        for name in skills:
            self.skills_registry.add(name)
        self._skills = skills

    def get_messages(self):
        """the agent's live conversation list."""
        return self.messages

    def set_messages(self, messages):
        """replace the agent's conversation; the next run() continues from it."""
        if messages is None: messages = []
        self.messages = messages

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
        names = []
        for tool in self._tools:
            names.append(_tool_name(tool))
        payload = SessionsRegistry.flow_payload(list(self.messages),
                                                self.system_prompt,
                                                self._system_prompt,
                                                list(self._skills),
                                                names,
                                                self.model,
                                                reasoning_effort=self.reasoning_effort,
                                                temperature=self.temperature,
                                                max_steps=self.max_steps)
        SessionsRegistry.write_flow(path, payload)
        return path

    def load(self, path):
        """replace the conversation + settings in place from a .flow file, so a
        served agent keeps its identity and live aliases. the stored leading
        system message is dropped and the prompt is re-derived from base +
        skills. tools restore by name only (callables must be re-supplied via
        set_tools). returns the path loaded."""
        from cai.session import SessionsRegistry

        payload = SessionsRegistry.read_flow(path)
        settings = payload.get("settings") or {}
        messages = list(payload.get("messages") or [])
        # the agent owns the composed prompt; drop a stored leading system msg.
        if messages and messages[0].get("role") == "system":
            messages = messages[1:]

        self._system_prompt = settings.get("system_prompt_base")
        self.set_skills(settings.get("skills") or [])
        self.set_tools(settings.get("selected_tools") or [])
        if settings.get("model"):
            self.model = settings["model"]
        self.reasoning_effort = settings.get("reasoning_effort")
        self.temperature = settings.get("temperature")
        self.max_steps = settings.get("max_steps")
        # mutate in place so any external alias keeps pointing at live state.
        self.messages[:] = messages
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

    def steer(self, text):
        """queue a steering message; the in-flight (or next) run folds it in as a
        user turn at its next turn boundary. safe to call from another thread."""
        self._steer.push(text)

    def _stream(self, stream):
        """the orchestration: stream one call_llm turn over the live conversation.
        yields Event objects and returns the final answer text. combines the
        system prompt and translates the hooks list here (not at construction) so
        skills and hooks added since are reflected."""
        skills_prompt = self.skills_registry.system_prompt
        system_prompt = _combine_prompts(self._system_prompt, skills_prompt)
        schemas = self.tools_registry.tools
        dispatch = self.tools_registry.dispatch
        # the one place the public [(event, fn), ...] list becomes the internal
        # HooksRegistry that call_llm wants.
        hooks_registry = HooksRegistry.from_list(self._hooks)
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
                                   stream=stream)
        return text

    def run(self, prompt=None):
        """append `prompt` as a user turn (if given) and return a handle over the
        agent's live conversation. Iterate it to stream events; `messages`
        keeps growing, so the next run continues this conversation."""
        self.interrupt.clear()
        if self._killed.is_set():
            # a killed agent never runs again: re-arm the interrupt so the run
            # winds down at once instead of calling the model.
            self.interrupt.set()
        if prompt is not None:
            self.messages.append({"role": "user", "content": prompt})
        return RunHandle(self, self.stream)

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
                 system_prompt=None,
                 tools=None,
                 skills=None,
                 hooks=None,
                 ui=None,
                 interrupt=None,
                 reasoning_effort=None,
                 temperature=None,
                 max_steps=None,
                 stream=True):
        agent = Agent(model=model,
                      api=api,
                      system_prompt=system_prompt,
                      tools=tools,
                      skills=skills,
                      hooks=hooks,
                      ui=ui,
                      interrupt=interrupt,
                      reasoning_effort=reasoning_effort,
                      temperature=temperature,
                      max_steps=max_steps,
                      stream=stream)
        agent.set_messages(messages)
        # no prompt to fold in: `messages` is already the complete conversation,
        # so stream straight over it (Agent.run's prompt-append is for the
        # persistent path).
        super().__init__(agent, stream)

    def close(self):
        """close the Agent this Run built (and owns), tearing down its tool
        registry and any MCP servers it spawned."""
        self.agent.close()
