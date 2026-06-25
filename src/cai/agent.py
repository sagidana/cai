"""agent: a persistent conversation (Agent) and a one-shot execution (Run).

Run wraps a single call_llm invocation and makes it iterable: you loop over it
to stream Event objects, and once it drains, `text` holds the final answer and
`messages` holds the full transcript. It is lazy and single-consumption - the
loop doesn't start until you iterate (or call `wait()`), and only once.

Agent owns a growing conversation, a model, and an api client. `run(prompt)`
folds the prompt in as a user turn and hands back a Run over the agent's live
`messages`, so iterating it streams the answer and evolves the conversation in
place - the next `run` continues where this one left off.

Tools are explicit: `tools=` is a list whose items are Python callables or MCP
tool-name strings ('<mcp>__<tool>'). `skills=` is a list of skill names; each
unions its tools into the registry and folds its prompt into the system prompt.
Only the tools you pass (plus those skills pull in) are sent to the model. MCP
servers are spawned lazily on first use. `hooks=` is a list of (event, fn) pairs
fired through the run (Run turns it into the internal HooksRegistry that call_llm
wants). Serving/attach, saving, and cloning are later layers."""
from __future__ import annotations

from cai import config
from cai.api import OpenAiApi
from cai.llm import call_llm
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


def _tool_name(tool):
    """the exposed name of a tools= item: a callable's __name__, else the string
    (an MCP tool ref) itself."""
    if callable(tool):
        return tool.__name__
    return tool


class Run:
    def __init__(self,
                 messages,
                 model,
                 api,
                 *,
                 system_prompt=None,
                 tools=None,
                 skills=None,
                 hooks=None,
                 tools_registry=None,
                 skills_registry=None,
                 stream=True):
        self.messages = messages
        self.model = model
        self.api = api
        self.system_prompt = system_prompt        # base prompt; skills folded in at run time
        self.tools = tools                        # callables/'<mcp>__<tool>' strings (standalone)
        self.skills = skills                      # skill names to activate (standalone)
        self.hooks = hooks                        # [(event, fn), ...] or None; translated at run time
        self.tools_registry = tools_registry      # a prebuilt ToolRegistry (an Agent passes its own)
        self.skills_registry = skills_registry    # the Agent's SkillsRegistry (for the live prompt)
        # we own (and must close) a registry only if we build it ourselves; one
        # passed in belongs to its creator (an Agent), so we leave it alone.
        self._private_registry = tools_registry is None
        self.stream = stream
        self.text = ""
        self._consumed = False

    def __iter__(self):
        if self._consumed:
            raise RuntimeError("Run already consumed")
        self._consumed = True

        # reuse the Agent's registries when passed; otherwise build our own from
        # the tools/skills handed in. either way, recombine the system prompt now
        # (not at construction) so skills added/removed since are reflected.
        registry = self.tools_registry
        skills_registry = self.skills_registry
        if registry is None:
            registry = ToolRegistry.for_tools(self.tools)
            self.tools_registry = registry
            skills_registry = SkillsRegistry.for_skills(self.skills, tools_registry=registry)
            self.skills_registry = skills_registry

        skills_prompt = None
        if skills_registry is not None:
            skills_prompt = skills_registry.system_prompt
        system_prompt = _combine_prompts(self.system_prompt, skills_prompt)

        schemas = registry.tools
        dispatch = registry.dispatch

        # Run is the one place the public [(event, fn), ...] list becomes the
        # internal HooksRegistry that call_llm wants.
        hooks_registry = HooksRegistry.from_list(self.hooks)

        gen = call_llm(self.messages,
                       self.model,
                       self.api,
                       system_prompt=system_prompt,
                       tools=schemas,
                       tools_dispatch=dispatch,
                       hooks=hooks_registry,
                       stream=self.stream)
        try:
            while True:
                try:
                    event = next(gen)
                except StopIteration as stop:
                    self.text = stop.value
                    return
                yield event
        finally:
            # close our own registry whether we drained or the caller broke out
            # early (a borrowed registry stays open for its owner).
            if self._private_registry:
                registry.close()

    def wait(self):
        """drain the run without consuming the events yourself; returns self,
        so `run.wait().text` reads the final answer."""
        for _event in self:
            pass
        return self

    def close(self):
        """close the registry this Run built itself; a registry passed in (an
        Agent's) is owned by the caller and left untouched."""
        if self._private_registry and self.tools_registry is not None:
            self.tools_registry.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Agent:
    def __init__(self,
                 *,
                 model=None,
                 system_prompt=None,
                 tools=None,
                 skills=None,
                 hooks=None):
        cfg = config.load_config()

        if model is None: model = cfg.model

        self.model = model
        self.api = OpenAiApi(cfg.base_url, config.load_api_key())
        self._system_prompt = system_prompt   # base; combined with skills on demand
        self._tools = tools or []
        self._skills = skills or []
        # build the registries once at bootstrap so every run() reuses them (and
        # the MCP servers they spawned); set_tools/set_skills mutate them in place.
        self.tools_registry = ToolRegistry.for_tools(self._tools)
        self.skills_registry = SkillsRegistry.for_skills(self._skills, tools_registry=self.tools_registry)
        self._hooks = hooks   # [(event, fn), ...] or None; forwarded to each Run as-is
        self.messages = []

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

    def run(self, prompt=None):
        """append `prompt` as a user turn (if given) and return a Run over the
        agent's live conversation. Iterate it to stream events; `messages`
        keeps growing, so the next run continues this conversation."""
        if prompt is not None:
            self.messages.append({"role": "user", "content": prompt})
        return Run(self.messages,
                   self.model,
                   self.api,
                   system_prompt=self._system_prompt,
                   hooks=self._hooks,
                   tools_registry=self.tools_registry,
                   skills_registry=self.skills_registry)

    def close(self):
        """close the tool registry, terminating any live MCP server connections
        this agent spawned at bootstrap."""
        self.tools_registry.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
