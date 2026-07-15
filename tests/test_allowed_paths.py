"""Tests for --allowed-paths: the CLI publishes the extra roots as
CAI_ALLOWED_PATHS, cai.safe_path admits them alongside the cwd jail, spawned
tool processes inherit the var through os.environ, and the python-tool kernel
jail binds them as read roots."""
import os

import pytest

import cai
from cai import cli
from cai import pytool_bootstrap
from cai.tools import ToolsRegistry


def test_safe_path_admits_allowed_paths(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("CAI_SCRATCH", raising=False)
    monkeypatch.setenv("CAI_ALLOWED_PATHS", str(extra))
    assert cai.safe_path(str(extra)) == str(extra)
    assert cai.safe_path(str(extra / "data.txt")) == str(extra / "data.txt")
    # a sibling that merely shares the allowed dir's name prefix stays jailed
    with pytest.raises(ValueError):
        cai.safe_path(str(tmp_path / "extra-evil" / "x"))
    with pytest.raises(ValueError):
        cai.safe_path("/etc/passwd")


def test_safe_path_admits_each_of_several_roots(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    one = tmp_path / "one"
    one.mkdir()
    two = tmp_path / "two"
    two.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("CAI_SCRATCH", raising=False)
    monkeypatch.setenv("CAI_ALLOWED_PATHS", os.pathsep.join([str(one), str(two)]))
    assert cai.safe_path(str(one / "a")) == str(one / "a")
    assert cai.safe_path(str(two / "b")) == str(two / "b")
    with pytest.raises(ValueError):
        cai.safe_path(str(tmp_path / "three" / "c"))


def test_a_file_grant_admits_just_that_file(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    granted = shared / "granted.txt"
    granted.write_text("x")
    (shared / "sibling.txt").write_text("y")
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("CAI_SCRATCH", raising=False)
    monkeypatch.setenv("CAI_ALLOWED_PATHS", str(granted))
    assert cai.safe_path(str(granted)) == str(granted)
    # neither the sibling nor the containing directory comes along
    with pytest.raises(ValueError):
        cai.safe_path(str(shared / "sibling.txt"))
    with pytest.raises(ValueError):
        cai.safe_path(str(shared))


def test_unset_var_grants_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAI_SCRATCH", raising=False)
    monkeypatch.delenv("CAI_ALLOWED_PATHS", raising=False)
    with pytest.raises(ValueError):
        cai.safe_path("/etc/passwd")


def test_publish_sets_env_from_comma_list(tmp_path, monkeypatch):
    one = tmp_path / "one"
    one.mkdir()
    two = tmp_path / "two.txt"
    two.write_text("a file entry works too")
    monkeypatch.delenv("CAI_ALLOWED_PATHS", raising=False)
    assert cli._publish_allowed_paths(f"{one}, {two}") is True
    expected = os.pathsep.join([str(one), str(two)])
    assert os.environ["CAI_ALLOWED_PATHS"] == expected


def test_publish_resolves_relative_entries_against_cwd(tmp_path, monkeypatch):
    (tmp_path / "rel").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAI_ALLOWED_PATHS", raising=False)
    assert cli._publish_allowed_paths("rel") is True
    assert os.environ["CAI_ALLOWED_PATHS"] == str(tmp_path / "rel")


def test_publish_rejects_a_missing_entry(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CAI_ALLOWED_PATHS", raising=False)
    assert cli._publish_allowed_paths(str(tmp_path / "nope")) is False
    assert "CAI_ALLOWED_PATHS" not in os.environ
    assert "does not exist" in capsys.readouterr().err


def test_main_exits_on_bad_allowed_paths(tmp_path):
    assert cli.main(["-p", "x", "--allowed-paths", str(tmp_path / "nope")]) == 1


def test_spawned_fs_server_inherits_the_grant(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "data.txt").write_text("granted bytes")
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("still jailed")
    monkeypatch.setenv("CAI_ALLOWED_PATHS", str(extra))

    registry = ToolsRegistry()
    registry.select("fs__read_file")
    try:
        out = registry.dispatch("fs__read_file", {"file_path": str(extra / "data.txt")})
        assert "granted bytes" in out
        out = registry.dispatch("fs__read_file", {"file_path": str(elsewhere)})
        assert "outside working directory" in out
    finally:
        registry.close()


def test_jail_read_roots_include_the_grant(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.setenv("CAI_ALLOWED_PATHS", str(extra))
    roots = pytool_bootstrap.compute_read_roots()
    assert str(extra) in roots
    # write roots stay scratch-only: the grant is read-only inside the jail
    assert str(extra) not in pytool_bootstrap.compute_write_roots()
