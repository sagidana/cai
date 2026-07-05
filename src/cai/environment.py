"""environment: the loaded install catalogue - everything ~/.config/cai offers.

An Environment holds what used to live in process-global class stores: the
function tools registered via cai.tool, the MCP servers declared via
cai.mcp_server, the hooks and commands registered via cai.hook / cai.command,
the discovered extension bundles, and the live Settings (cai.settings). An
Agent is constructed against an Environment (env=) and resolves everything
install-wide through it, so two agents in one process can see two different
installs, and a test builds a private empty one instead of resetting globals.

An extension is a self-contained bundle at ~/.config/cai/extensions/<name>/
that may contribute skills/*.md, tools/*.py (function tools), mcps/*.py (MCP
servers), and Python that registers hooks and commands through cai.hook /
cai.command in init.py / hooks/init.py / commands/init.py.

Construction and loading are separate:

  Environment()                     - empty; touches no disk. tests, embedders.
  Environment(list_extensions())    - the on-disk view without importing any
                                      extension Python (tab completion).
  Environment.default()             - the process-wide instance, created empty
                                      on first touch. Agent's env= default.
  env.load()                        - discover the extensions and import their
                                      Python into this env: while the imports
                                      run, the cai.tool / cai.hook / cai.command
                                      / cai.mcp_server decorators (and
                                      cai.settings) target it via
                                      Environment.target(). the CLI and TUI call
                                      Environment.default().load() at startup.

Function tools registered while an extension loads are namespaced
'<extension>__<name>' (mirroring MCP tools); the user's top-level init.py and
plain SDK registrations keep the bare name. The loader knows which bundle it is
importing, so no path-matching attribution is needed.

One caveat: Python imports each bundle file once per process (sys.modules), so
only the first load() of a given bundle runs its decorators. The isolation an
extra Environment buys is for synthetic/test catalogues and embedders with
different dirs - not for loading the same bundles twice."""
from __future__ import annotations

import os
import sys
import hashlib
import logging
import importlib.util
from dataclasses import dataclass
from dataclasses import field

from cai import config


log = logging.getLogger("cai")


@dataclass
class Extension:
    name: str
    path: str

    @property
    def dir(self):
        return self.path

    @property
    def skills_dir(self):
        return os.path.join(self.path, "skills")

    @property
    def tools_dir(self):
        return os.path.join(self.path, "tools")

    @property
    def mcps_dir(self):
        return os.path.join(self.path, "mcps")

    @property
    def init_file(self):
        return os.path.join(self.path, "init.py")

    @property
    def hooks_file(self):
        return os.path.join(self.path, "hooks", "init.py")

    @property
    def commands_file(self):
        return os.path.join(self.path, "commands", "init.py")


@dataclass
class Settings:
    """the live session settings (cai.settings): an extension/init.py tunes
    them by attribute during load, the :config overlay edits the same object in
    place. the skills / tools lists are auto-activated on every CLI run and
    merged into --skill / --tool by Environment.merge_activations; an SDK run
    builds Agent/Run directly and never passes through that merge."""
    show_reasoning: bool = True
    show_chips: bool = True
    show_chips_skills: bool = True
    show_chips_tools: bool = False
    show_chips_subagents: bool = False
    tool_result_max_chars: int = 40_000
    auto_save_sessions: bool = True
    max_sessions_mb: int = 500
    skills: list = field(default_factory=list)
    tools: list = field(default_factory=list)


def extensions_dir():
    return os.path.join(config.config_dir(), "extensions")


def init_path():
    return os.path.join(config.config_dir(), "init.py")


def builtin_skills_dir():
    """the skills shipped with cai, in builtins/skills/ inside the package -
    searched after the extension skills dirs."""
    return os.path.join(os.path.dirname(__file__), "builtins", "skills")


def builtin_mcp_dir():
    """the MCP servers shipped with cai, in builtins/mcps/ inside the package -
    searched after the extension mcps dirs."""
    return os.path.join(os.path.dirname(__file__), "builtins", "mcps")


def list_extensions():
    """the extension bundles under ~/.config/cai/extensions/, sorted by name.
    a filesystem scan only - no extension Python is run."""
    extensions = []
    try:
        entries = sorted(os.listdir(extensions_dir()))
    except OSError:
        return extensions
    for name in entries:
        path = os.path.join(extensions_dir(), name)
        if not os.path.isdir(path): continue
        extensions.append(Extension(name=name, path=path))
    return extensions


def _import_file(path):
    if not os.path.exists(path):
        return
    _load_module(path)


def _load_module(path):
    """load a bundle file as a package whose search path is its own directory,
    so it can `from . import helper` a sibling file, and cache it in
    sys.modules so its body (and the cai.hook / cai.command registrations)
    runs once per process. None (logged) on failure."""
    name = _module_name(path)
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    try:
        spec = importlib.util.spec_from_file_location(
            name,
            path,
            submodule_search_locations=[os.path.dirname(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(name, None)
        log.exception("environment: failed loading %s", path)
        return None


def _module_name(path):
    """a stable, collision-free package name for a bundle file: its directory
    name for readable tracebacks plus a short hash of the absolute path so two
    files never map to the same name."""
    folder = os.path.basename(os.path.dirname(path))
    digest = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:8]
    return "cai_ext_" + folder + "_" + digest


def _import_tool_files(extension):
    """import each tools/*.py in an extension so its cai.tool decorators
    register their function tools. files prefixed with _ are skipped, matching
    the MCP server discovery."""
    directory = extension.tools_dir
    if not os.path.isdir(directory):
        return
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".py"): continue
        if filename.startswith("_"): continue
        _import_file(os.path.join(directory, filename))


def _subagent_tools(agent):
    """the sub-agent tools (launch/wait/list/kill) bound to `agent`; the import is
    deferred so building an env - or importing cai.agent - never pulls the
    serving stack. wired here, at the composition root, so the core Agent
    stays ignorant of the layers above it."""
    from cai.subagent import subagent_tools
    return subagent_tools(agent)


def _python_tools(agent):
    """the python tool bound to `agent`; deferred import, same reason
    as _subagent_tools - so an env never pulls the tool's subprocess machinery."""
    from cai.pytool import python_tools
    return python_tools(agent)


class Environment:
    """one install catalogue: extensions, function tools, declared MCP servers,
    hooks, commands, and the live Settings. see the module docstring for how
    construction, default() and load() relate."""

    # the process-wide default (created empty on first touch), and the env a
    # load() is currently importing into - the decorators' registration target.
    _default = None
    _loading = None

    def __init__(self, extensions=None):
        self.extensions = list(extensions or [])
        self.settings = Settings()
        self._function_tools = {}    # exposed name -> callable
        self._declared_servers = {}  # server name -> spec dict
        self._hooks = []             # (event, fn) pairs, registration order
        self._commands = {}          # name -> cai.commands.Command
        # factories called with each new Agent to produce tools bound to it -
        # registered on the agent (unselected) until a skill or the user
        # selects them. default: the sub-agent tools.
        self._agent_tools = [_subagent_tools, _python_tools]
        # the bundle load() is importing right now: an extension name, 'user'
        # for the top-level init.py, or None outside a load. register_tool
        # reads it to namespace extension tools.
        self._loading_extension = None

    @classmethod
    def default(cls):
        """the process-wide Environment, created empty on first touch. Agent
        falls back to it when no env= is given; the CLI/TUI load() it at
        startup."""
        if cls._default is None:
            cls._default = cls()
        return cls._default

    @classmethod
    def target(cls):
        """where the cai.tool / cai.hook / cai.command / cai.mcp_server
        decorators (and cai.settings) register right now: the env being
        load()ed when one is, else the process default."""
        if cls._loading is not None:
            return cls._loading
        return cls.default()

    @staticmethod
    def merge_activations(selected, configured):
        """CLI helper: the --skill / --tool names plus the cai.settings ones,
        de-duplicated, the explicit names first. an SDK run builds Agent/Run
        directly and never reaches here, so the settings lists are CLI-only."""
        merged = list(selected or [])
        for name in (configured or []):
            if name not in merged:
                merged.append(name)
        return merged

    def load(self):
        """discover the extensions and import their Python (alphabetical) then
        the user's top-level init.py, all registering into this env: while the
        imports run this env is Environment.target(), so the cai.hook /
        cai.command / cai.tool / cai.mcp_server decorators and cai.settings land
        here. returns self."""
        self.extensions = list_extensions()
        previous = Environment._loading
        Environment._loading = self
        try:
            for extension in self.extensions:
                self._loading_extension = extension.name
                _import_file(extension.init_file)
                _import_file(extension.hooks_file)
                _import_file(extension.commands_file)
                _import_tool_files(extension)
            self._loading_extension = "user"
            _import_file(init_path())
        finally:
            self._loading_extension = None
            Environment._loading = previous
        return self

    # --- registration (the decorators' backing) ---

    def register_tool(self, fn):
        """record a function tool, keyed by its exposed name. a tool registered
        while an extension loads is namespaced '<extension>__<name>' (mirroring
        MCP tools); the user's init.py and plain SDK registrations keep the bare
        __name__. the exposed name is stamped on fn so the schema, dispatch key
        and selection all agree on it. a later registration of the same name
        wins (the user's init runs last)."""
        name = fn.__name__
        extension = self._loading_extension
        if extension and extension != "user":
            name = f"{extension}__{name}"
        fn._cai_tool_name = name
        self._function_tools[name] = fn

    def function_tool(self, name):
        """the callable registered under `name` via cai.tool, or None when no
        function tool by that name is known."""
        return self._function_tools.get(name)

    def function_tools(self):
        """the registered function tools as exposed name -> callable (a copy)."""
        return dict(self._function_tools)

    def register_server(self, name, command=None, url=None, env=None, headers=None, cwd=None):
        """declare a named MCP server (cai.mcp_server's backing), the
        programmatic counterpart to dropping a <name>.py into an mcps/ dir.
        exactly one of `command` (a stdio server: an argv list, with optional
        env/cwd) or `url` (a remote Streamable-HTTP server, with optional
        headers) must be given. its tools surface namespaced '<name>__<tool>'
        like any MCP server's. a later declaration of the same name wins."""
        if command is None and url is None:
            raise ValueError(f"MCP server {name!r}: give command= or url=")
        if command is not None and url is not None:
            raise ValueError(f"MCP server {name!r}: give command= or url=, not both")
        if url is not None and (env is not None or cwd is not None):
            raise ValueError(f"MCP server {name!r}: env/cwd apply to a command server, not a url")
        if command is not None and headers is not None:
            raise ValueError(f"MCP server {name!r}: headers apply to a url server, not a command")
        spec = {}
        if command is not None:
            if isinstance(command, str):
                raise ValueError(f"MCP server {name!r}: command must be an argv list, not a string")
            spec["command"] = list(command)
            if env is not None:
                spec["env"] = dict(env)
            if cwd is not None:
                spec["cwd"] = cwd
        else:
            spec["url"] = url
            if headers is not None:
                spec["headers"] = dict(headers)
        self._declared_servers[name] = spec

    def server_spec(self, name):
        """the spec dict declared for MCP server `name`, or None."""
        return self._declared_servers.get(name)

    def declared_servers(self):
        """the declared MCP servers as name -> spec dict (a copy)."""
        return dict(self._declared_servers)

    def register_hook(self, event, fn):
        """record a hook pair (cai.hook's backing; the event is validated by
        the decorator). every run of an agent on this env fires it."""
        self._hooks.append((event, fn))

    def hooks(self):
        """the registered (event, fn) hook pairs, registration order (a copy)."""
        return list(self._hooks)

    def register_command(self, name, command):
        """record a :command (cai.command's backing). a later registration of
        the same name wins (the user's init.py runs last)."""
        self._commands[name] = command

    def commands(self):
        """the registered commands as name -> Command (a copy)."""
        return dict(self._commands)

    def agent_tools(self, agent):
        """the tools bound to a new Agent, from every registered factory.
        Agent.__init__ registers them (unselected, override=True) so each
        agent - a clone, a child - gets its OWN bindings, never an inherited
        parent's."""
        tools = []
        for factory in self._agent_tools:
            tools.extend(factory(agent))
        return tools

    # --- the filesystem search paths and the install-wide catalogue ---

    def skill_dirs(self):
        """the skill source dirs in resolution order: each extension's skills/
        dir (an earlier one shadows a later one) then the bundled builtins."""
        dirs = []
        for extension in self.extensions:
            dirs.append(extension.skills_dir)
        dirs.append(builtin_skills_dir())
        return dirs

    def mcp_dirs(self):
        """the MCP source dirs in resolution order: each extension's mcps/ dir
        (an earlier one shadows a later one) then the bundled builtins."""
        dirs = []
        for extension in self.extensions:
            dirs.append(extension.mcps_dir)
        dirs.append(builtin_mcp_dir())
        return dirs

    def available_skills(self):
        """every skill name available to activate: the *.md stems across the
        skill dirs, deduped and sorted. a filesystem scan only - no skill is
        loaded."""
        names = set()
        for directory in self.skill_dirs():
            if not os.path.isdir(directory): continue
            for filename in sorted(os.listdir(directory)):
                if not filename.endswith(".md"): continue
                names.add(filename[:-len(".md")])
        return sorted(names)

    def available_tools(self):
        """every tool this env offers, as a flat list of names in the form
        `tools=` accepts: the registered function tools (their exposed names)
        followed by every MCP tool ('<mcp_name>__<tool_name>') the servers
        expose - declared via cai.mcp_server first, then across the mcp dirs.
        each MCP server is spawned (or, when remote, contacted) briefly to list
        its tools and then closed - nothing is left running. a declared server
        shadows an on-disk one of the same name, and an extension file shadows a
        builtin; a server that fails to start is logged and skipped."""
        from cai.tools import LocalMCPServer, server_from_spec

        names = list(self._function_tools.keys())
        seen_labels = set()
        for label in sorted(self._declared_servers.keys()):
            seen_labels.add(label)
            server = None
            try:
                server = server_from_spec(label, self._declared_servers[label])
                for tool in server.list_tools():
                    names.append(f"{label}__{tool['name']}")
            except Exception as e:
                log.error("available_tools: declared server %r failed: %s", label, e)
            finally:
                if server is not None:
                    server.close()
        for directory in self.mcp_dirs():
            if not os.path.isdir(directory): continue
            for filename in sorted(os.listdir(directory)):
                if not filename.endswith(".py"): continue
                if filename.startswith("_"): continue
                label = filename[:-len(".py")]
                if label in seen_labels: continue
                seen_labels.add(label)
                path = os.path.join(directory, filename)
                server = None
                try:
                    server = LocalMCPServer([sys.executable, path], label)
                    for tool in server.list_tools():
                        names.append(f"{label}__{tool['name']}")
                except Exception as e:
                    log.error("available_tools: %r failed: %s", path, e)
                finally:
                    if server is not None:
                        server.close()
        return names
