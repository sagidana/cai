"""Tests for the session scratch directory: the Agent owns (or inherits) it,
the ToolsRegistry hands it to every local MCP spawn as CAI_SCRATCH, and the
builtin fs server admits it alongside the cwd jail. The fs server is really
spawned - no network, everything under tmp_path."""
import os
import sys

import cai
from cai.agent import Agent
from cai.environment import Environment, builtin_mcp_dir
from cai.tools import ToolsRegistry


def _fs_path():
    return os.path.join(builtin_mcp_dir(), "fs.py")


def test_scratch_injected_into_declared_server(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "artifact.txt").write_text("intermediate bytes")
    cai.mcp_server("myfs", command=[sys.executable, _fs_path()])

    registry = ToolsRegistry(scratch=lambda: str(scratch))
    registry.select("myfs__read_file")
    try:
        out = registry.dispatch("myfs__read_file",
                                {"file_path": str(scratch / "artifact.txt")})
        assert "intermediate bytes" in out
    finally:
        registry.close()


def test_scratch_injected_into_file_discovered_server(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "artifact.txt").write_text("found via builtins")

    registry = ToolsRegistry(scratch=lambda: str(scratch))
    registry.select("fs__read_file")
    try:
        out = registry.dispatch("fs__read_file",
                                {"file_path": str(scratch / "artifact.txt")})
        assert "found via builtins" in out
    finally:
        registry.close()


def test_declared_env_wins_over_injected_scratch(tmp_path):
    declared = tmp_path / "declared"
    declared.mkdir()
    (declared / "a.txt").write_text("declared wins")
    injected = tmp_path / "injected"
    injected.mkdir()
    (injected / "b.txt").write_text("never reachable")
    cai.mcp_server("myfs",
                   command=[sys.executable, _fs_path()],
                   env={"CAI_SCRATCH": str(declared)})

    registry = ToolsRegistry(scratch=lambda: str(injected))
    registry.select("myfs__read_file")
    try:
        out = registry.dispatch("myfs__read_file", {"file_path": str(declared / "a.txt")})
        assert "declared wins" in out
        out = registry.dispatch("myfs__read_file", {"file_path": str(injected / "b.txt")})
        assert "outside working directory" in out
    finally:
        registry.close()


def test_no_scratch_provider_means_no_injection(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("jailed")

    registry = ToolsRegistry()
    registry.select("fs__read_file")
    try:
        out = registry.dispatch("fs__read_file", {"file_path": str(outside)})
        assert "outside working directory" in out
    finally:
        registry.close()


def test_scratch_does_not_unlock_other_paths(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("still jailed")

    registry = ToolsRegistry(scratch=lambda: str(scratch))
    registry.select("fs__read_file")
    try:
        out = registry.dispatch("fs__read_file", {"file_path": str(elsewhere)})
        assert "outside working directory" in out
    finally:
        registry.close()


def test_agent_creates_scratch_lazily_and_deletes_on_close():
    agent = Agent(model="m", api=object())
    assert agent._scratch is None
    path = agent.scratch()
    assert os.path.isdir(path)
    assert agent.scratch() == path
    assert agent.tools_registry.scratch() == path
    agent.close()
    assert not os.path.exists(path)


def test_agent_inherited_scratch_is_never_deleted(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    agent = Agent(model="m", api=object(), scratch=str(shared))
    assert agent.scratch() == str(shared)
    agent.close()
    assert os.path.isdir(str(shared))
