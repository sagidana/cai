"""skills: load skill files into a SkillsRegistry that layers prompt fragments
and tools onto an existing ToolRegistry.

A skill is a markdown file ~/.config/cai/skills/<name>.md. Everything before the
first '---' line is a header of 'key: value' lines; everything after is the
prompt body. Header fields:

  name:   <skill name>             the added first field, naming the skill
  tools:  <tool>, <tool>, ...      MCP tool refs to expose ('<mcp>__<tool>')
  skills: <skill>, <skill>, ...    other skills to activate first (recursive)

Activating a skill unions its tools into the ToolRegistry it was built against
and appends its prompt body to the combined system prompt. `skills:` are
activated first (foundation-first), and each skill loads at most once."""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass

from cai import config


log = logging.getLogger("cai")


@dataclass
class Skill:
    name: str
    tools: list
    skills: list
    body: str


def skills_dir():
    return os.path.join(config.config_dir(), "skills")


def _split_csv(value):
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part: continue
        items.append(part)
    return items


def _parse_skill(text):
    """split a skill file into its header fields and prompt body. a file with no
    '---' line is all body, declaring no tools and no uses."""
    lines = text.splitlines()
    sep_index = None
    for i, line in enumerate(lines):
        if line.strip() != "---": continue
        sep_index = i
        break

    header = {}
    body = text
    if sep_index is not None:
        body = "\n".join(lines[sep_index + 1:])
        for line in lines[:sep_index]:
            if ":" not in line: continue
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()

    return Skill(name=header.get("name", ""),
                 tools=_split_csv(header.get("tools", "")),
                 skills=_split_csv(header.get("skills", "")),
                 body=body)


def _read_skill(name):
    path = os.path.join(skills_dir(), name + ".md")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    return _parse_skill(text)


class SkillsRegistry:
    """The skills activated for an Agent/Run: their combined prompt plus the
    tools they pull into the ToolRegistry they were built against. Like
    ToolRegistry, but it always works on a ToolRegistry passed in - activating a
    skill extends that same registry object in place."""

    def __init__(self, tools_registry):
        self.tools_registry = tools_registry
        self.skills = []        # activated Skill objects, foundation-first
        # names visited, marked before recursing so foundation-first ordering
        # (append after recursion) stays cycle-safe.
        self._loaded = set()

    @classmethod
    def for_skills(cls, skills=None, *, tools_registry):
        """activate `skills` (recursively, via each skill's `skills:` header)
        against `tools_registry`, which is extended in place with every tool the
        skills declare. returns the SkillsRegistry holding the activated skills."""
        registry = cls(tools_registry)
        if not skills:
            return registry
        for name in skills:
            registry._activate(name)
        return registry

    def _activate(self, name):
        if name in self._loaded:
            return
        self._loaded.add(name)
        skill = _read_skill(name)
        if skill is None:
            log.warning("skill %r not found in %s", name, skills_dir())
            return
        # foundation first: activate everything this skill builds on, then it.
        for used in skill.skills:
            self._activate(used)
        for tool_name in skill.tools:
            if self.tools_registry.has(tool_name): continue
            try:
                self.tools_registry.register_mcp_tool(tool_name)
            except ValueError as e:
                log.error("skill %r: bad tool %r: %s", name, tool_name, e)
        self.skills.append(skill)

    @property
    def system_prompt(self):
        """the activated skills' prompt bodies, foundation-first, joined - or
        None when no skill contributed a prompt."""
        parts = []
        for skill in self.skills:
            body = skill.body.strip()
            if not body: continue
            parts.append(body)
        if not parts:
            return None
        return "\n\n".join(parts)
