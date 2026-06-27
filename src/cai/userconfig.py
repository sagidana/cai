"""userconfig: the per-user layout under ~/.config/cai and its extensions.

An extension is a self-contained bundle at ~/.config/cai/extensions/<name>/ that
may contribute skills/*.md, tools/*.py (function tools), mcps/*.py (MCP servers),
and Python that registers hooks and commands through cai.hook / cai.command in
init.py / hooks/init.py / commands/init.py.

UserConfig is the namespace for the operations as well as the loaded value:
UserConfig.load() imports every extension's Python once - init/hooks/commands so
the cai.hook / cai.command decorators bake what they register into the global
HooksRegistry / CommandsRegistry, and each tools/*.py so the cai.tool decorator
bakes its function tools into the global ToolsRegistry. UserConfig.skill_dirs() /
mcp_dirs() are the filesystem search paths skills.py and tools.py consult before
the builtins. UserConfig.extension_for() maps a registered hook/command/tool (by
the file it was defined in) back to its extension."""
from __future__ import annotations

import os
import sys
import hashlib
import logging
import importlib.util
from dataclasses import dataclass

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


class ExtensionsRegistry:
    """the discovered extension bundles. hooks and commands no longer live here -
    they are baked into the process-global HooksRegistry / CommandsRegistry by the
    cai.hook / cai.command decorators when UserConfig.load() imports each
    extension."""

    def __init__(self, extensions):
        self.extensions = extensions


@dataclass
class UserConfig:
    """the loaded ~/.config/cai (the discovered extensions plus the user-level
    settings) and the namespace for the operations that produce it - load(), the
    dir helpers, and extension_for, all static methods on this class."""

    extensions: ExtensionsRegistry
    show_reasoning: bool = True

    @staticmethod
    def extensions_dir():
        return os.path.join(config.config_dir(), "extensions")

    @staticmethod
    def init_path():
        return os.path.join(config.config_dir(), "init.py")

    @staticmethod
    def list_extensions():
        """the extension bundles under ~/.config/cai/extensions/, sorted by name.
        a filesystem scan only - no extension Python is run."""
        extensions = []
        try:
            entries = sorted(os.listdir(UserConfig.extensions_dir()))
        except OSError:
            return extensions
        for name in entries:
            path = os.path.join(UserConfig.extensions_dir(), name)
            if not os.path.isdir(path): continue
            extensions.append(Extension(name=name, path=path))
        return extensions

    @staticmethod
    def skill_dirs():
        """each extension's skills/ dir, in extension order."""
        dirs = []
        for extension in UserConfig.list_extensions():
            dirs.append(extension.skills_dir)
        return dirs

    @staticmethod
    def tool_dirs():
        """each extension's tools/ dir, in extension order."""
        dirs = []
        for extension in UserConfig.list_extensions():
            dirs.append(extension.tools_dir)
        return dirs

    @staticmethod
    def mcp_dirs():
        """each extension's mcps/ dir, in extension order."""
        dirs = []
        for extension in UserConfig.list_extensions():
            dirs.append(extension.mcps_dir)
        return dirs

    @staticmethod
    def extension_for(path):
        """the extension a file belongs to: the name of the extension whose dir
        contains `path`, 'user' for the top-level ~/.config/cai, or None for code
        outside the cai config (e.g. a plain SDK script). resolves the origin of a
        globally-registered hook or command back to its extension."""
        if not path:
            return None
        target = os.path.abspath(path)
        for extension in UserConfig.list_extensions():
            root = os.path.abspath(extension.path) + os.sep
            if target.startswith(root):
                return extension.name
        root = os.path.abspath(config.config_dir()) + os.sep
        if target.startswith(root):
            return "user"
        return None

    @staticmethod
    def load():
        """import every extension's Python (alphabetical) then the user's
        top-level init.py, so their cai.hook / cai.command / cai.tool decorators
        register into the global HooksRegistry / CommandsRegistry / ToolsRegistry.
        returns the discovered extensions and user settings; the hooks, commands
        and function tools live in those registries."""
        extensions = UserConfig.list_extensions()
        for extension in extensions:
            UserConfig._import_file(extension.init_file)
            UserConfig._import_file(extension.hooks_file)
            UserConfig._import_file(extension.commands_file)
            UserConfig._import_tools(extension)
        UserConfig._import_file(UserConfig.init_path())
        return UserConfig(extensions=ExtensionsRegistry(extensions))

    @staticmethod
    def _import_tools(extension):
        """import each tools/*.py in an extension so its cai.tool decorators
        register their function tools globally. files prefixed with _ are
        skipped, matching the MCP server discovery in tools.py."""
        directory = extension.tools_dir
        if not os.path.isdir(directory):
            return
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".py"): continue
            if filename.startswith("_"): continue
            UserConfig._import_file(os.path.join(directory, filename))

    @staticmethod
    def _import_file(path):
        if not os.path.exists(path):
            return
        UserConfig._load_module(path)

    @staticmethod
    def _load_module(path):
        """load a bundle file as a package whose search path is its own directory,
        so it can `from . import helper` a sibling file, and cache it in
        sys.modules so its body (and the cai.hook / cai.command registrations)
        runs once per process. None (logged) on failure."""
        name = UserConfig._module_name(path)
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

    @staticmethod
    def _module_name(path):
        """a stable, collision-free package name for a bundle file: its directory
        name for readable tracebacks plus a short hash of the absolute path so two
        files never map to the same name."""
        folder = os.path.basename(os.path.dirname(path))
        digest = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:8]
        return "cai_ext_" + folder + "_" + digest
