"""skills: load skill files into a SkillsRegistry that layers prompt fragments
and tools onto an existing ToolRegistry.

A skill is a markdown file <name>.md, resolved from each extension's skills/
dir (see cai.userconfig) first, then the builtins shipped with cai
(builtins/skills/ beside this module). Everything before the first '---' line is
a header of 'key: value' lines; everything after is the prompt body. Header
fields:

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
from cai.userconfig import UserConfig


log = logging.getLogger("cai")


@dataclass
class Skill:
    name: str
    tools: list
    skills: list
    body: str


def builtin_skills_dir():
    """the skills shipped with cai by default, in builtins/skills/ beside this
    module - searched after the extension skills dirs."""
    return os.path.join(os.path.dirname(__file__), "builtins", "skills")


def _search_dirs():
    """the skill source dirs in resolution order: each extension's skills/ dir
    (an earlier one shadows a later one) then the bundled builtins."""
    dirs = list(UserConfig.skill_dirs())
    dirs.append(builtin_skills_dir())
    return dirs


def _skill_path(name):
    """resolve <name>.md to a source file, searching the extension skills dirs
    first then the bundled builtins. None when neither has it."""
    filename = name + ".md"
    for directory in _search_dirs():
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return None


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
    path = _skill_path(name)
    if path is None:
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
        # the active skill names, in activation order: this registry is the
        # single source of truth for which skills are on. dependency skills
        # (pulled via a `skills:` header) and unresolved names are kept too, so
        # names() round-trips through set_skills.
        self._names = []
        # name -> Skill for the resolved skills, foundation-first by insertion
        # (a skill is recorded after its dependencies), for the combined prompt
        # and for knowing which tools to drop on remove().
        self._skills = {}

    @classmethod
    def for_skills(cls, skills=None, *, tools_registry):
        """activate `skills` (recursively, via each skill's `skills:` header)
        against `tools_registry`, which is extended in place with every tool the
        skills declare. returns the SkillsRegistry holding the activated skills."""
        registry = cls(tools_registry)
        if not skills:
            return registry
        for name in skills:
            registry.add(name)
        return registry

    @classmethod
    def available_skills(cls):
        """every skill name available to activate: the *.md stems across the
        extension skills dirs and the builtins, deduped and sorted. a filesystem
        scan only - no skill is loaded."""
        names = set()
        for directory in _search_dirs():
            if not os.path.isdir(directory): continue
            for filename in sorted(os.listdir(directory)):
                if not filename.endswith(".md"): continue
                names.add(filename[:-len(".md")])
        return sorted(names)

    def names(self):
        """the active skill names, in activation order. includes dependency
        skills and names that resolved to no file, so the list round-trips
        through set_skills."""
        return list(self._names)

    def add(self, name):
        """activate a skill (and, first, the skills it builds on), unioning its
        tools into the tool registry. an already-active name is a no-op."""
        if name in self._names:
            return
        self._names.append(name)
        self._activate(name)

    def _activate(self, name):
        """register a skill's tools and record its prompt, after first activating
        the skills it builds on. an unresolved name is warned and skipped, but
        stays in names() (mirroring set_skills's input)."""
        skill = _read_skill(name)
        if skill is None:
            log.warning("skill %r not found", name)
            return
        # foundation first: activate everything this skill builds on, then it.
        for used in skill.skills:
            if used in self._names: continue
            self._names.append(used)
            self._activate(used)
        # select the skill's tools, registering an MCP tool on demand. a tool
        # already registered (e.g. an Agent's sub-agent tools) is just selected.
        for tool_name in skill.tools:
            try:
                self.tools_registry.select(tool_name)
            except ValueError as e:
                log.error("skill %r: bad tool %r: %s", name, tool_name, e)
        self._skills[name] = skill

    def remove(self, name):
        """deactivate a skill: drop its name, its prompt, and deselect every tool
        it declared - even one also selected by hand or shared with another skill
        (kept deliberately simple). the tools stay registered; dependency skills
        it pulled in stay active."""
        if name in self._names:
            self._names.remove(name)
        skill = self._skills.pop(name, None)
        if skill is None:
            return
        for tool_name in skill.tools:
            self.tools_registry.deselect(tool_name)

    @property
    def system_prompt(self):
        """the activated skills' prompt bodies, foundation-first, joined - or
        None when no skill contributed a prompt."""
        parts = []
        for skill in self._skills.values():
            body = skill.body.strip()
            if not body: continue
            parts.append(body)
        if not parts:
            return None
        return "\n\n".join(parts)
