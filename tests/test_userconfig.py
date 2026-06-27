"""Tests for cai.userconfig - extension discovery, resource dirs, and the hooks
and commands collected by load(). Fully offline: each test gets its own
~/.config/cai via HOME, and extensions are plain files written into it."""
import os
import textwrap

import pytest

from cai import config
from cai import userconfig


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


def test_no_extensions_dir_is_empty():
    assert userconfig.list_extensions() == []
    assert userconfig.skill_dirs() == []
    assert userconfig.tool_dirs() == []
    uc = userconfig.load()
    assert uc.extensions.hooks == []
    assert uc.extensions.commands == {}


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


def test_load_collects_hooks_and_commands():
    path = _ext("fs")
    _write(os.path.join(path, "hooks", "init.py"), """
        def register(reg):
            reg.add_hook("after_turn", lambda ctx: "h")
    """)
    _write(os.path.join(path, "commands", "init.py"), """
        def register(reg):
            reg.add_command("fs", lambda ctx: None, help="files")
    """)
    uc = userconfig.load()
    assert len(uc.extensions.hooks) == 1
    assert uc.extensions.hooks[0][0] == "after_turn"
    assert "fs" in uc.extensions.commands
    assert uc.extensions.commands["fs"].help == "files"


def test_top_level_init_runs_last_and_overrides():
    path = _ext("fs")
    _write(os.path.join(path, "init.py"), """
        def register(reg):
            reg.add_command("x", lambda ctx: "ext", help="from ext")
    """)
    _write(userconfig.init_path(), """
        def register(reg):
            reg.add_command("x", lambda ctx: "user", help="from user")
    """)
    uc = userconfig.load()
    assert uc.extensions.commands["x"].help == "from user"


def test_register_gets_dir_and_config():
    path = _ext("probe")
    _write(os.path.join(config.config_dir(), "config.json"),
           '{"base_url": "http://x", "model": "m"}')
    _write(os.path.join(path, "init.py"), """
        seen = {}
        def register(reg):
            seen["dir"] = reg.dir
            seen["model"] = reg.config.model
            reg.add_command(reg.dir.split("/")[-1], lambda ctx: None)
    """)
    uc = userconfig.load()
    assert "probe" in uc.extensions.commands


def test_module_without_register_is_skipped():
    path = _ext("bad")
    _write(os.path.join(path, "init.py"), "value = 1\n")
    uc = userconfig.load()
    assert uc.extensions.commands == {}


def test_extension_has_empty_properties_when_it_registers_nothing():
    _ext("plain")
    uc = userconfig.load()
    plain = None
    for extension in uc.extensions.extensions:
        if extension.name == "plain":
            plain = extension
    assert plain is not None
    assert plain.hooks == []
    assert plain.commands == {}
    assert plain.skills_dir.endswith(os.path.join("plain", "skills"))
    assert plain.tools_dir.endswith(os.path.join("plain", "tools"))


def test_registry_owns_per_extension_attribution():
    a = _ext("aaa")
    b = _ext("bbb")
    _write(os.path.join(a, "init.py"), """
        def register(reg):
            reg.add_command("a", lambda ctx: None)
    """)
    _write(os.path.join(b, "init.py"), """
        def register(reg):
            reg.add_command("b", lambda ctx: None)
    """)
    uc = userconfig.load()
    owned = {}
    for extension in uc.extensions.extensions:
        owned[extension.name] = sorted(extension.commands)
    assert owned["aaa"] == ["a"]
    assert owned["bbb"] == ["b"]
    assert sorted(uc.extensions.commands) == ["a", "b"]


def test_module_that_raises_is_isolated():
    good = _ext("aaa_good")
    bad = _ext("zzz_bad")
    _write(os.path.join(good, "init.py"), """
        def register(reg):
            reg.add_command("good", lambda ctx: None)
    """)
    _write(os.path.join(bad, "init.py"), """
        def register(reg):
            raise RuntimeError("boom")
    """)
    uc = userconfig.load()
    assert "good" in uc.extensions.commands
