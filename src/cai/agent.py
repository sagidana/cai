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
servers are spawned lazily on first use. Serving/attach, saving, and cloning are
later layers."""
from __future__ import annotations

from cai import config
from cai.api import OpenAiApi
from cai.llm import call_llm
from cai.tools import ToolRegistry
from cai.skills import SkillsRegistry


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


class Run:
    def __init__(self,
                 messages,
                 model,
                 api,
                 *,
                 system_prompt=None,
                 tools=None,
                 skills=None,
                 tools_registry=None,
                 stream=True):
        self.messages = messages
        self.model = model
        self.api = api
        self.system_prompt = system_prompt
        self.tools = tools                    # callables and/or '<mcp>__<tool>' strings, or None
        self.skills = skills                  # skill names to activate (standalone Run only)
        self.tools_registry = tools_registry  # a prebuilt ToolRegistry (an Agent passes its own)
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

        # reuse the registry an Agent built once at bootstrap (skills already
        # activated, system prompt already combined); otherwise build our own
        # from the tools handed in, activate our skills into it, and fold the
        # skills' prompt onto ours.
        registry = self.tools_registry
        system_prompt = self.system_prompt
        if registry is None:
            registry = ToolRegistry.for_tools(self.tools)
            self.tools_registry = registry
            skills = SkillsRegistry.for_skills(self.skills, tools_registry=registry)
            system_prompt = _combine_prompts(self.system_prompt, skills.system_prompt)
        schemas = registry.tools
        dispatch = registry.dispatch

        gen = call_llm(self.messages,
                       self.model,
                       self.api,
                       system_prompt=system_prompt,
                       tools=schemas,
                       tools_dispatch=dispatch,
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
    def __init__(self, *, model=None, system_prompt=None, tools=None, skills=None):
        cfg = config.load_config()

        if model is None: model = cfg.model

        self.model = model
        self.api = OpenAiApi(cfg.base_url, config.load_api_key())
        # build the tool registry once, here, so every run() reuses it (and the
        # MCP servers it spawned) instead of rebuilding it per run.
        self.tools_registry = ToolRegistry.for_tools(tools)
        # activate skills once too: they extend the tool registry above and
        # contribute prompt fragments folded into the system prompt.
        self.skills_registry = SkillsRegistry.for_skills(skills, tools_registry=self.tools_registry)
        self.system_prompt = _combine_prompts(system_prompt, self.skills_registry.system_prompt)
        self.messages = []

    def run(self, prompt=None):
        """append `prompt` as a user turn (if given) and return a Run over the
        agent's live conversation. Iterate it to stream events; `messages`
        keeps growing, so the next run continues this conversation."""
        if prompt is not None:
            self.messages.append({"role": "user", "content": prompt})
        return Run(self.messages,
                   self.model,
                   self.api,
                   system_prompt=self.system_prompt,
                   tools_registry=self.tools_registry)

    def close(self):
        """close the tool registry, terminating any live MCP server connections
        this agent spawned at bootstrap."""
        self.tools_registry.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
