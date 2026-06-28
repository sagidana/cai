"""Tests for cai.userconfig - extension discovery, resource dirs, and load()
importing each extension so its cai.hook / cai.command decorators register into
the global registries (with attribution back to the extension). Fully offline:
each test gets its own ~/.config/cai via HOME, and the conftest fixture resets
the global registries between tests."""
import os
import textwrap

import pytest

from cai.userconfig import UserConfig
from cai.hooks import HooksRegistry
from cai.commands import CommandsRegistry
from cai.tools import ToolsRegistry


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))


def _ext(name):
    path = os.path.join(UserConfig.extensions_dir(), name)
    os.makedirs(path, exist_ok=True)
    return path


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(text))


def test_no_extensions_dir_registers_nothing():
    assert UserConfig.list_extensions() == []
    assert UserConfig.skill_dirs() == []
    assert UserConfig.tool_dirs() == []
    assert UserConfig.mcp_dirs() == []
    UserConfig.load()
    assert HooksRegistry.registered() == []
    assert CommandsRegistry.commands() == {}
    assert ToolsRegistry.registered() == {}


def test_extensions_sorted_and_resource_dirs():
    _ext("bbb")
    _ext("aaa")
    names = []
    for extension in UserConfig.list_extensions():
        names.append(extension.name)
    assert names == ["aaa", "bbb"]

    root = UserConfig.extensions_dir()
    assert UserConfig.skill_dirs() == [os.path.join(root, "aaa", "skills"),
                                       os.path.join(root, "bbb", "skills")]
    assert UserConfig.tool_dirs() == [os.path.join(root, "aaa", "tools"),
                                      os.path.join(root, "bbb", "tools")]
    assert UserConfig.mcp_dirs() == [os.path.join(root, "aaa", "mcps"),
                                     os.path.join(root, "bbb", "mcps")]


def test_files_that_are_not_dirs_are_ignored():
    os.makedirs(UserConfig.extensions_dir(), exist_ok=True)
    _write(os.path.join(UserConfig.extensions_dir(), "README.md"), "not an ext")
    assert UserConfig.list_extensions() == []


def test_load_registers_hooks_and_commands_globally():
    path = _ext("fs")
    _write(os.path.join(path, "hooks", "init.py"), """
        import cai
        @cai.hook("after_turn")
        def fold(ctx):
            return "h"
    """)
    _write(os.path.join(path, "commands", "init.py"), """
        import cai
        @cai.command(help="files")
        def fs(ctx):
            return None
    """)
    UserConfig.load()

    registered = HooksRegistry.registered()
    assert len(registered) == 1
    event, fn, _origin = registered[0]
    assert event == "after_turn"
    assert fn.__name__ == "fold"

    assert "fs" in CommandsRegistry.commands()
    assert CommandsRegistry.commands()["fs"].help == "files"


def test_load_registers_function_tools_globally_with_attribution():
    path = _ext("calc")
    _write(os.path.join(path, "tools", "math.py"), """
        import cai
        @cai.tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b
    """)
    UserConfig.load()

    fn = ToolsRegistry.global_function("calc__add")
    assert fn is not None
    assert fn(2, 3) == 5
    _other, origin = ToolsRegistry.registered()["calc__add"]
    assert UserConfig.extension_for(origin) == "calc"


def test_extension_function_tool_is_namespaced_by_extension():
    path = _ext("web")
    _write(os.path.join(path, "tools", "net.py"), """
        import cai
        @cai.tool
        def fetch_url(url: str) -> str:
            \"\"\"Fetch a URL.\"\"\"
            return url
    """)
    UserConfig.load()

    # the tool surfaces as '<extension>__<name>', mirroring MCP tools, in the
    # global store, the available list, and the schema sent to the model.
    assert "web__fetch_url" in ToolsRegistry.available_tools()
    assert "fetch_url" not in ToolsRegistry.available_tools()
    registry = ToolsRegistry.for_tools(["web__fetch_url"])
    assert registry.selected() == ["web__fetch_url"]
    schema = registry.tools[0]
    assert schema["function"]["name"] == "web__fetch_url"


def test_top_level_user_function_tool_keeps_its_bare_name():
    # a tool defined in the top-level ~/.config/cai/init.py is attributed to
    # "user", which is not an extension, so it is not namespaced.
    _write(UserConfig.init_path(), """
        import cai
        @cai.tool
        def now() -> str:
            \"\"\"The time.\"\"\"
            return "now"
    """)
    UserConfig.load()

    assert "now" in ToolsRegistry.available_tools()
    _fn, origin = ToolsRegistry.registered()["now"]
    assert UserConfig.extension_for(origin) == "user"


def test_tools_underscore_files_are_not_imported():
    path = _ext("calc")
    _write(os.path.join(path, "tools", "_helper.py"), """
        import cai
        @cai.tool
        def hidden() -> str:
            \"\"\"Hidden.\"\"\"
            return "x"
    """)
    UserConfig.load()
    assert ToolsRegistry.global_function("hidden") is None


def test_registered_hook_is_baked_into_new_registries():
    path = _ext("fs")
    _write(os.path.join(path, "hooks", "init.py"), """
        import cai
        @cai.hook("after_turn")
        def fold(ctx):
            return "h"
    """)
    UserConfig.load()
    registry = HooksRegistry()
    assert len(registry.pairs()) == 1
    assert registry.pairs()[0][0] == "after_turn"


def test_attribution_resolves_origin_to_extension():
    path = _ext("fs")
    _write(os.path.join(path, "hooks", "init.py"), """
        import cai
        @cai.hook("after_turn")
        def fold(ctx):
            return "h"
    """)
    UserConfig.load()
    _event, _fn, origin = HooksRegistry.registered()[0]
    assert UserConfig.extension_for(origin) == "fs"


def test_top_level_init_is_user_attributed_and_overrides():
    path = _ext("fs")
    _write(os.path.join(path, "commands", "init.py"), """
        import cai
        @cai.command(name="x", help="from ext")
        def x_ext(ctx): pass
    """)
    _write(UserConfig.init_path(), """
        import cai
        @cai.command(name="x", help="from user")
        def x_user(ctx): pass
    """)
    UserConfig.load()
    command = CommandsRegistry.commands()["x"]
    assert command.help == "from user"
    assert UserConfig.extension_for(command.origin) == "user"


def test_command_can_import_sibling_relatively():
    path = _ext("sib")
    _write(os.path.join(path, "helper.py"), "LABEL = 'from-sibling'\n")
    _write(os.path.join(path, "init.py"), """
        import cai
        from . import helper
        @cai.command(name="sib", help=helper.LABEL)
        def sib(ctx): pass
    """)
    UserConfig.load()
    assert CommandsRegistry.commands()["sib"].help == "from-sibling"


def test_bare_sibling_import_still_fails():
    path = _ext("bare")
    _write(os.path.join(path, "helper.py"), "LABEL = 'x'\n")
    _write(os.path.join(path, "init.py"), """
        import helper
        import cai
        @cai.command(name="bare")
        def bare(ctx): pass
    """)
    UserConfig.load()
    assert "bare" not in CommandsRegistry.commands()


def test_module_body_runs_once_across_loads():
    path = _ext("once")
    log = os.path.join(path, "execs.log")
    _write(os.path.join(path, "init.py"), f"""
        import cai
        with open({log!r}, "a") as f:
            f.write("x")
        @cai.command(name="once")
        def once(ctx): pass
    """)
    UserConfig.load()
    UserConfig.load()
    with open(log) as f:
        assert f.read() == "x"


def test_module_that_raises_is_isolated():
    good = _ext("aaa_good")
    bad = _ext("zzz_bad")
    _write(os.path.join(good, "init.py"), """
        import cai
        @cai.command(name="good")
        def good(ctx): pass
    """)
    _write(os.path.join(bad, "init.py"), """
        raise RuntimeError("boom")
    """)
    UserConfig.load()
    assert "good" in CommandsRegistry.commands()
