"""userconfig: the per-user layout under ~/.config/cai and its extensions.

An extension is a self-contained bundle at ~/.config/cai/extensions/<name>/ that
may contribute skills/*.md, tools/*.py (MCP servers), and Python registering
hooks and commands through init.py / hooks/init.py / commands/init.py. Each such
module exposes register(reg), where reg is the Extension itself: reg.add_hook /
reg.add_command record what the bundle adds, and reg.dir / reg.config give it the
bundle's directory and the loaded cai Config.

Every Extension carries the full set of properties - skills_dir, tools_dir,
hooks, commands - whether or not it defines them; the unowned ones are just
empty. load() builds an Extension per bundle (alphabetical), runs each one's
register modules into it, then the user's top-level init.py as a final 'user'
extension, and returns a UserConfig whose extensions is an ExtensionsRegistry
aggregating them: the hooks and commands the cli and tui fold into the agent and
the `:`-command dispatch.

skill_dirs/tool_dirs are the per-extension resource paths skills.py and tools.py
search before the builtins; they are a filesystem scan only and never run any
extension Python."""
from __future__ import annotations

import os
import sys
import hashlib
import logging
import importlib.util
from dataclasses import dataclass, field

from cai import config
from cai.commands import Command


log = logging.getLogger("cai")


def extensions_dir():
    return os.path.join(config.config_dir(), "extensions")


def init_path():
    return os.path.join(config.config_dir(), "init.py")


@dataclass
class Extension:
    """one extension bundle and the hooks and commands it registers. it is the
    reg handed to each of its register(reg) modules; a bundle that registers
    nothing keeps the empty hooks/commands it starts with."""
    name: str
    path: str
    config: object = None
    hooks: list = field(default_factory=list)
    commands: dict = field(default_factory=dict)

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
    def init_file(self):
        return os.path.join(self.path, "init.py")

    @property
    def hooks_file(self):
        return os.path.join(self.path, "hooks", "init.py")

    @property
    def commands_file(self):
        return os.path.join(self.path, "commands", "init.py")

    def add_hook(self, event, fn):
        self.hooks.append((event, fn))

    def add_command(self, name, fn, help=""):
        self.commands[name] = Command(fn=fn, help=help)

    def load(self):
        """run this bundle's register modules into self, in order."""
        _run_register(self.init_file, self)
        _run_register(self.hooks_file, self)
        _run_register(self.commands_file, self)
        return self


class ExtensionsRegistry:
    """the loaded extensions and the hooks and commands they aggregate. hooks
    concatenate in extension order; commands merge into one table, so a later
    extension (the user's top-level init.py runs last) overrides an earlier one
    sharing a name."""

    def __init__(self, extensions):
        self.extensions = extensions

    @property
    def hooks(self):
        out = []
        for extension in self.extensions:
            out.extend(extension.hooks)
        return out

    @property
    def commands(self):
        out = {}
        for extension in self.extensions:
            out.update(extension.commands)
        return out


@dataclass
class UserConfig:
    """the loaded ~/.config/cai: the place per-user state hangs off. holds the
    extensions and the user-level settings the init scripts set."""
    extensions: ExtensionsRegistry
    show_reasoning: bool = True


def list_extensions():
    """the extension bundles under ~/.config/cai/extensions/, sorted by name. a
    filesystem scan only - no extension Python is run."""
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


def skill_dirs():
    """each extension's skills/ dir, in extension order."""
    dirs = []
    for extension in list_extensions():
        dirs.append(extension.skills_dir)
    return dirs


def tool_dirs():
    """each extension's tools/ dir, in extension order."""
    dirs = []
    for extension in list_extensions():
        dirs.append(extension.tools_dir)
    return dirs


def load():
    """build and load every extension, then the user's top-level init.py as a
    final 'user' extension, returning a UserConfig whose extensions is an
    ExtensionsRegistry over them all."""
    cfg = _safe_config()
    extensions = []
    for extension in list_extensions():
        extension.config = cfg
        extension.load()
        extensions.append(extension)
    user = Extension(name="user", path=config.config_dir(), config=cfg)
    _run_register(user.init_file, user)
    extensions.append(user)
    return UserConfig(extensions=ExtensionsRegistry(extensions))


def _safe_config():
    try:
        return config.load_config()
    except (FileNotFoundError, ValueError):
        return None


def _run_register(path, extension):
    if not os.path.exists(path): return
    module = _load_module(path)
    if module is None: return
    register = getattr(module, "register", None)
    if register is None:
        log.warning("userconfig: %s exposes no register(reg); skipping", path)
        return
    try:
        register(extension)
    except Exception:
        log.exception("userconfig: register() in %s raised", path)


def _load_module(path):
    """load a register module as a package whose search path is its own
    directory, so it can `from . import helper` a sibling file, and cache it in
    sys.modules so its body runs once per process. None (logged) on failure."""
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
        log.exception("userconfig: failed loading %s", path)
        return None


def _module_name(path):
    """a stable, collision-free package name for a bundle file: its directory
    name for readable tracebacks plus a short hash of the absolute path so two
    files never map to the same name."""
    folder = os.path.basename(os.path.dirname(path))
    digest = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:8]
    return "cai_ext_" + folder + "_" + digest
