"""Tests for the builtin `python` tool: the audit-hook sandbox, the managed
venv, and the call() tool-proxy that dispatches the agent's own tools in-process
through the run's gates. The child is really spawned; for speed most tests point
the venv at the current interpreter (the sandbox blocks cffi regardless), and one
test exercises real venv creation."""
import os
import sys

import pytest

import cai
from cai import config, hooks, pytool
from cai.agent import Agent
from cai.hooks import HooksRegistry, RunGate


def _fast_venv(monkeypatch):
    """skip real venv creation - run the child under this interpreter."""
    monkeypatch.setattr(pytool, "ensure_venv", lambda: sys.executable)


def _run(agent, code, timeout=20):
    return agent.tools_registry.dispatch("python", {"code": code, "timeout": timeout})


def _agent_with_echo():
    @cai.tool
    def echo(x: int) -> str:
        """echo x back."""
        return f"echo:{x}"
    return Agent(model="m", api=object(), tools=["echo"])


# --- wiring ---------------------------------------------------------------

def test_python_registered_unselected_until_the_skill():
    agent = Agent(model="m", api=object())
    try:
        assert "python" in agent.tools_registry.names()
        assert "python" not in agent.tools
    finally:
        agent.close()


def test_python_skill_selects_the_tool():
    agent = Agent(model="m", api=object(), skills=["python"])
    try:
        assert "python" in agent.tools
    finally:
        agent.close()


# --- venv -----------------------------------------------------------------

def test_ensure_venv_creates_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: str(tmp_path))
    python = pytool.ensure_venv()
    assert os.path.exists(python)
    assert pytool.ensure_venv() == python


def test_ensure_venv_rebuilds_a_broken_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: str(tmp_path))
    python = pytool.ensure_venv()
    os.remove(python)
    assert pytool.ensure_venv() == python
    assert os.path.exists(python)


# --- sandbox --------------------------------------------------------------

def test_output_and_exit_code(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "print(2 ** 10)").strip() == "1024"
        assert _run(agent, "x = 1") == "(no output)"
        assert "[exit code 3]" in _run(agent, "import sys; sys.exit(3)")
    finally:
        agent.close()


def test_reads_allowed_but_all_writes_denied(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (cwd / "existing.txt").write_text("data")
    monkeypatch.chdir(cwd)

    @cai.tool
    def noop() -> str:
        """noop."""
        return ""
    agent = Agent(model="m", api=object(), tools=["noop"], scratch=str(scratch))
    try:
        # reading inside the jail is fine
        assert _run(agent, "print(open('existing.txt').read())").strip() == "data"
        # the tool is read-only: no write, even under cwd or scratch
        assert "read-only" in _run(agent, "open('new.txt','w').write('x')")
        assert not (cwd / "new.txt").exists()
        scratch_write = ("import os; p=os.path.join(os.environ['CAI_SCRATCH'],'s.txt');"
                         " open(p,'w').write('y')")
        assert "read-only" in _run(agent, scratch_write)
        assert not (scratch / "s.txt").exists()
        # and no delete / rename of the existing file
        assert "read-only" in _run(agent, "import os; os.remove('existing.txt')")
        assert (cwd / "existing.txt").exists()
        assert "read-only" in _run(agent, "import os; os.rename('existing.txt','r.txt')")
        assert "read-only" in _run(agent, "import shutil; shutil.rmtree('.')")
    finally:
        agent.close()


@pytest.mark.parametrize("snippet", [
    "open('/etc/hostname','w')",
    "open('/etc/hostname')",
    "import ctypes",
    "import cffi",
    "import subprocess; subprocess.Popen(['ls'])",
    "import os; os.system('ls')",
    "import socket; socket.socket().connect(('example.com', 80))",
    # directory enumeration outside the cwd is a read too - must not leak the tree
    "import os; os.listdir('/')",
    "import os; os.scandir('/etc')",
    "import pathlib; list(pathlib.Path('/').iterdir())",
])
def test_sandbox_denials(snippet, tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        assert "PermissionError" in _run(agent, snippet)
    finally:
        agent.close()


def test_masked_traversal_leaks_nothing(tmp_path, monkeypatch):
    # os.walk / glob swallow the scandir denial (onerror ignores it), so they
    # don't raise - but they must come back EMPTY, never the outside listing.
    _fast_venv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "import os; print(list(os.walk('/etc')))").strip() == "[]"
        assert _run(agent, "import glob; print(glob.glob('/etc/*'))").strip() == "[]"
    finally:
        agent.close()


def test_reading_and_listing_allowed_inside_the_jail(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("x")
    agent = _agent_with_echo()
    try:
        out = _run(agent, "import os; print(sorted(os.listdir('.')))")
        assert "a.txt" in out
        out = _run(agent, "import glob; print(glob.glob('*.txt'))")
        assert "a.txt" in out
    finally:
        agent.close()


def test_timeout(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "while True: pass", timeout=1) == "Error: run timed out after 1s"
    finally:
        agent.close()


# --- tool proxy -----------------------------------------------------------

def test_call_proxies_a_tool_and_only_print_reaches_output(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        # the raw result is computed but NOT printed; only the derived value is
        out = _run(agent, "r = call('echo', x=7)\nprint(r.split(':')[1])")
        assert out.strip() == "7"
        assert "echo:7" not in out
    finally:
        agent.close()


def test_call_recursion_and_unavailable_guards(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        assert "cannot call itself" in _run(agent, "print(call('python', code='x'))")
        assert "not available" in _run(agent, "print(call('secret'))")
    finally:
        agent.close()


def test_inner_call_is_gated_by_before_tool_call(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()

    def veto(ctx):
        return False
    registry = HooksRegistry()
    registry.register("before_tool_call", veto)
    gate = RunGate(hooks=registry,
                   dispatch=agent.tools_registry.dispatch,
                   model="m",
                   config=None,
                   ui=hooks.NULL_UI,
                   messages=[],
                   usage=None,
                   hooks_data={})
    token = hooks.set_gate(gate)
    try:
        out = _run(agent, "print(call('echo', x=1))")
        assert "aborted by a before_tool_call hook" in out
    finally:
        hooks.reset_gate(token)
        agent.close()


def test_inner_call_without_a_gate_still_dispatches(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        # no run gate published (SDK-style direct use): confined dispatch, ungated
        assert hooks.current_gate() is None
        assert _run(agent, "print(call('echo', x=9))").strip() == "echo:9"
    finally:
        agent.close()
