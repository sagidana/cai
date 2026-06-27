"""Tests for cai.userconfig - extension discovery, resource dirs, and load()
importing each extension so its cai.hook / cai.command decorators register into
the global registries (with attribution back to the extension). Fully offline:
each test gets its own ~/.config/cai via HOME, and the conftest fixture resets
the global registries between tests."""
import os
import textwrap

import pytest

from cai import userconfig
from cai.hooks import HooksRegistry
from cai.commands import CommandsRegistry


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))


def _ext(name):
    path = os.path.join(userconfig.extensions_dir(), name)
    os.makedirs(path, exist_ok=True)
    return path


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(text))


def test_no_extensions_dir_registers_nothing():
    assert userconfig.list_extensions() == []
    assert userconfig.skill_dirs() == []
    assert userconfig.tool_dirs() == []
    userconfig.load()
    assert HooksRegistry.registered() == []
    assert CommandsRegistry.commands() == {}


def test_extensions_sorted_and_resource_dirs():
    _ext("bbb")
    _ext("aaa")
    names = []
    for extension in userconfig.list_extensions():
        names.append(extension.name)
    assert names == ["aaa", "bbb"]

    root = userconfig.extensions_dir()
    assert userconfig.skill_dirs() == [os.path.join(root, "aaa", "skills"),
                                       os.path.join(root, "bbb", "skills")]
    assert userconfig.tool_dirs() == [os.path.join(root, "aaa", "tools"),
                                      os.path.join(root, "bbb", "tools")]


def test_files_that_are_not_dirs_are_ignored():
    os.makedirs(userconfig.extensions_dir(), exist_ok=True)
    _write(os.path.join(userconfig.extensions_dir(), "README.md"), "not an ext")
    assert userconfig.list_extensions() == []


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
    userconfig.load()

    registered = HooksRegistry.registered()
    assert len(registered) == 1
    event, fn, _origin = registered[0]
    assert event == "after_turn"
    assert fn.__name__ == "fold"

    assert "fs" in CommandsRegistry.commands()
    assert CommandsRegistry.commands()["fs"].help == "files"


def test_registered_hook_is_baked_into_new_registries():
    path = _ext("fs")
    _write(os.path.join(path, "hooks", "init.py"), """
        import cai
        @cai.hook("after_turn")
        def fold(ctx):
            return "h"
    """)
    userconfig.load()
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
    userconfig.load()
    _event, _fn, origin = HooksRegistry.registered()[0]
    assert userconfig.extension_for(origin) == "fs"


def test_top_level_init_is_user_attributed_and_overrides():
    path = _ext("fs")
    _write(os.path.join(path, "commands", "init.py"), """
        import cai
        @cai.command(name="x", help="from ext")
        def x_ext(ctx): pass
    """)
    _write(userconfig.init_path(), """
        import cai
        @cai.command(name="x", help="from user")
        def x_user(ctx): pass
    """)
    userconfig.load()
    command = CommandsRegistry.commands()["x"]
    assert command.help == "from user"
    assert userconfig.extension_for(command.origin) == "user"


def test_command_can_import_sibling_relatively():
    path = _ext("sib")
    _write(os.path.join(path, "helper.py"), "LABEL = 'from-sibling'\n")
    _write(os.path.join(path, "init.py"), """
        import cai
        from . import helper
        @cai.command(name="sib", help=helper.LABEL)
        def sib(ctx): pass
    """)
    userconfig.load()
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
    userconfig.load()
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
    userconfig.load()
    userconfig.load()
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
    userconfig.load()
    assert "good" in CommandsRegistry.commands()
