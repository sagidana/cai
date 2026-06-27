"""Tests for cai.extend - the `cai extend` subcommand: installing bundles from a
folder / zip (with single-wrapper-folder descent), --replace, validation, --list
and --remove. Fully offline (no http source is exercised): each test gets its own
~/.config/cai via HOME, and bundles are built on disk in tmp_path."""
import os
import zipfile

import pytest

from cai import cli
from cai import extend
from cai.userconfig import UserConfig


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _make_bundle(root, name, skill="hi"):
    """a minimal valid bundle folder at root/name with one skill."""
    path = os.path.join(root, name)
    _write(os.path.join(path, "skills", "demo.md"), skill)
    return path


def _installed_dir(name):
    return os.path.join(UserConfig.extensions_dir(), name)


def _run(argv):
    """drive `cai extend ...` through the real CLI parser and dispatch."""
    return cli.main(["extend"] + argv)


def test_install_folder(tmp_path):
    bundle = _make_bundle(str(tmp_path), "demo")
    assert _run([bundle]) == 0

    installed = os.path.join(_installed_dir("demo"), "skills", "demo.md")
    assert os.path.isfile(installed)
    names = []
    for extension in UserConfig.list_extensions():
        names.append(extension.name)
    assert names == ["demo"]


def test_install_skips_pycache(tmp_path):
    bundle = _make_bundle(str(tmp_path), "demo")
    _write(os.path.join(bundle, "tools", "__pycache__", "x.pyc"), "junk")
    _write(os.path.join(bundle, "tools", "srv.py"), "# server")
    assert _run([bundle]) == 0
    assert os.path.isfile(os.path.join(_installed_dir("demo"), "tools", "srv.py"))
    assert not os.path.exists(os.path.join(_installed_dir("demo"), "tools", "__pycache__"))


def test_install_zip_with_wrapper_folder(tmp_path):
    bundle = _make_bundle(str(tmp_path), "wrapped")
    zip_path = os.path.join(str(tmp_path), "archive.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(os.path.join(bundle, "skills", "demo.md"), "wrapped/skills/demo.md")

    assert _run([zip_path]) == 0
    # the name comes from the wrapper folder, not the archive filename.
    assert os.path.isfile(os.path.join(_installed_dir("wrapped"), "skills", "demo.md"))


def test_install_flat_zip_named_from_archive(tmp_path):
    bundle = _make_bundle(str(tmp_path), "demo")
    zip_path = os.path.join(str(tmp_path), "myext.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(os.path.join(bundle, "skills", "demo.md"), "skills/demo.md")

    assert _run([zip_path]) == 0
    assert os.path.isfile(os.path.join(_installed_dir("myext"), "skills", "demo.md"))


def test_install_aborts_when_already_installed(tmp_path, capsys):
    bundle = _make_bundle(str(tmp_path), "demo", skill="first")
    assert _run([bundle]) == 0

    again = _make_bundle(str(tmp_path / "v2"), "demo", skill="second")
    assert _run([again]) == 1
    out = capsys.readouterr().out
    assert "already installed" in out
    with open(os.path.join(_installed_dir("demo"), "skills", "demo.md")) as f:
        assert f.read() == "first"


def test_install_replace_overwrites(tmp_path):
    bundle = _make_bundle(str(tmp_path), "demo", skill="first")
    assert _run([bundle]) == 0
    again = _make_bundle(str(tmp_path / "v2"), "demo", skill="second")
    assert _run([again, "--replace"]) == 0
    with open(os.path.join(_installed_dir("demo"), "skills", "demo.md")) as f:
        assert f.read() == "second"


def test_replace_drops_removed_files(tmp_path):
    bundle = _make_bundle(str(tmp_path), "demo")
    _write(os.path.join(bundle, "tools", "old.py"), "# old")
    assert _run([bundle]) == 0

    leaner = _make_bundle(str(tmp_path / "v2"), "demo")
    assert _run([leaner, "--replace"]) == 0
    assert not os.path.exists(os.path.join(_installed_dir("demo"), "tools", "old.py"))


def test_install_rejects_non_bundle(tmp_path, capsys):
    plain = os.path.join(str(tmp_path), "plain")
    _write(os.path.join(plain, "notes.txt"), "nothing here")
    assert _run([plain]) == 1
    assert "not a bundle" in capsys.readouterr().out


def test_install_rejects_missing_path(tmp_path, capsys):
    assert _run([os.path.join(str(tmp_path), "nope")]) == 1
    assert "no such path" in capsys.readouterr().out


def test_install_prints_readme(tmp_path, capsys):
    bundle = _make_bundle(str(tmp_path), "demo")
    _write(os.path.join(bundle, "README.md"), "HELLO FROM README")
    assert _run([bundle]) == 0
    assert "HELLO FROM README" in capsys.readouterr().out


def test_list_empty(capsys):
    assert _run(["--list"]) == 0
    assert "no extensions installed" in capsys.readouterr().out


def test_list_shows_components(tmp_path, capsys):
    bundle = _make_bundle(str(tmp_path), "demo")
    _write(os.path.join(bundle, "tools", "srv.py"), "# server")
    _write(os.path.join(bundle, "init.py"), "# init")
    assert _run([bundle]) == 0
    capsys.readouterr()

    assert _run(["--list"]) == 0
    out = capsys.readouterr().out
    assert "demo" in out
    assert "skills" in out
    assert "tools" in out
    assert "init" in out


def test_remove(tmp_path, capsys):
    bundle = _make_bundle(str(tmp_path), "demo")
    assert _run([bundle]) == 0
    capsys.readouterr()

    assert _run(["--remove", "demo"]) == 0
    assert "removed extension 'demo'" in capsys.readouterr().out
    assert not os.path.exists(_installed_dir("demo"))


def test_remove_absent(capsys):
    assert _run(["--remove", "ghost"]) == 1
    assert "no extension named 'ghost'" in capsys.readouterr().out


def test_remove_completer_lists_installed(tmp_path):
    _make_bundle(str(tmp_path), "demo")
    assert _run([_make_bundle(str(tmp_path), "alpha")]) == 0
    assert _run([os.path.join(str(tmp_path), "demo")]) == 0

    assert extend._extension_completer("") == ["alpha", "demo"]
    assert extend._extension_completer("al") == ["alpha"]


def test_no_mode_is_an_error(capsys):
    with pytest.raises(SystemExit):
        _run([])
    assert "give a source" in capsys.readouterr().err


def test_mutually_exclusive_modes_error(tmp_path, capsys):
    bundle = _make_bundle(str(tmp_path), "demo")
    with pytest.raises(SystemExit):
        _run([bundle, "--list"])
    assert "mutually exclusive" in capsys.readouterr().err
