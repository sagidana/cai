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


def _force_sandbox(monkeypatch, mode):
    """pin the python_sandbox mode - keeps the test independent of the
    developer's real config.json."""
    def fake_optional(key, default=None):
        if key == "python_sandbox":
            return mode
        return default
    monkeypatch.setattr(config, "load_optional", fake_optional)


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


def test_reads_allowed_writes_confined_to_scratch(tmp_path, monkeypatch):
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
        # writes outside scratch are denied - cwd included
        assert "scratch dir only" in _run(agent, "open('new.txt','w').write('x')")
        assert not (cwd / "new.txt").exists()
        # scratch is the one writable island - the write really lands
        scratch_write = ("import os; p=os.path.join(os.environ['CAI_SCRATCH'],'s.txt');"
                         " open(p,'w').write('y')")
        _run(agent, scratch_write)
        assert (scratch / "s.txt").read_text() == "y"
        # deleting / renaming under scratch works too
        scratch_remove = ("import os; os.remove(os.path.join(os.environ['CAI_SCRATCH'],'s.txt'))")
        _run(agent, scratch_remove)
        assert not (scratch / "s.txt").exists()
        # but no delete / rename / mkdir of anything outside it
        assert "scratch dir only" in _run(agent, "import os; os.remove('existing.txt')")
        assert (cwd / "existing.txt").exists()
        assert "scratch dir only" in _run(agent, "import os; os.rename('existing.txt','r.txt')")
        assert "scratch dir only" in _run(agent, "import shutil; shutil.rmtree('.')")
        # a rename may not smuggle a file across the boundary either way
        cross = ("import os; os.rename('existing.txt',"
                 " os.path.join(os.environ['CAI_SCRATCH'],'stolen.txt'))")
        assert "scratch dir only" in _run(agent, cross)
    finally:
        agent.close()


@pytest.mark.parametrize("snippet", [
    "open('/etc/hostname','w')",
    "open('/etc/hostname')",
    "import ctypes",
    "import cffi",
    "import subprocess; subprocess.Popen(['ls'])",
    "import os; os.system('ls')",
    "import socket; socket.socket().connect(('127.0.0.1', 9))",
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


# --- kernel jail ------------------------------------------------------------

STAT_PROBE = """import os
try:
    os.stat('/etc/passwd')
    print('visible')
except FileNotFoundError:
    print('hidden')
"""


def test_kernel_jail_outside_paths_do_not_exist(tmp_path, monkeypatch):
    # the stat family emits no audit event - only the kernel jail hides this
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        assert _run(agent, STAT_PROBE).strip() == "hidden"
        assert _run(agent, "import os; print(os.path.exists('/etc'))").strip() == "False"
    finally:
        agent.close()


NET_PROBE = """import socket
s = socket.socket()
try:
    s.connect(('example.com', 80))
    print('connected')
except socket.gaierror:
    print('no-network')
"""


def test_kernel_jail_has_no_network(tmp_path, monkeypatch):
    # hostname resolution runs in libc, below the audit hook - inside the empty
    # network namespace it always fails, however the snippet reaches for the net
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        assert _run(agent, NET_PROBE).strip() == "no-network"
    finally:
        agent.close()


def test_kernel_jail_still_reads_cwd_and_scratch(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (cwd / "in.txt").write_text("cwd-data")
    (scratch / "s.txt").write_text("scratch-data")
    monkeypatch.chdir(cwd)

    @cai.tool
    def noop2() -> str:
        """noop."""
        return ""
    agent = Agent(model="m", api=object(), tools=["noop2"], scratch=str(scratch))
    try:
        assert _run(agent, "print(open('in.txt').read())").strip() == "cwd-data"
        scratch_read = ("import os; p=os.path.join(os.environ['CAI_SCRATCH'],'s.txt');"
                        " print(open(p).read())")
        assert _run(agent, scratch_read).strip() == "scratch-data"
    finally:
        agent.close()


def test_kernel_jail_scratch_is_writable_but_cwd_is_not(tmp_path, monkeypatch):
    # the write must land on the real disk through the read-write scratch bind,
    # while a cwd write still fails (the hook answers first with PermissionError,
    # an OSError; had it been evaded, the read-only mount answers with EROFS -
    # another OSError - so the probe holds for both layers).
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(cwd)

    @cai.tool
    def noop3() -> str:
        """noop."""
        return ""
    agent = Agent(model="m", api=object(), tools=["noop3"], scratch=str(scratch))
    try:
        scratch_write = ("import os; p=os.path.join(os.environ['CAI_SCRATCH'],'w.txt');"
                         " open(p,'w').write('landed'); print('ok')")
        assert _run(agent, scratch_write).strip() == "ok"
        assert (scratch / "w.txt").read_text() == "landed"
        cwd_probe = """import os
try:
    os.open('raw.txt', os.O_WRONLY | os.O_CREAT)
    print('writable')
except OSError:
    print('read-only')
"""
        assert _run(agent, cwd_probe).strip() == "read-only"
        assert not (cwd / "raw.txt").exists()
    finally:
        agent.close()


def test_kernel_jail_scratch_nested_under_cwd_still_writable(tmp_path, monkeypatch):
    # scratch inside cwd: the recursive cwd bind already carries it, so it gets
    # no read bind of its own - the read-write bind must still stack on top.
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(tmp_path)

    @cai.tool
    def noop4() -> str:
        """noop."""
        return ""
    agent = Agent(model="m", api=object(), tools=["noop4"], scratch=str(scratch))
    try:
        scratch_write = ("import os; p=os.path.join(os.environ['CAI_SCRATCH'],'n.txt');"
                         " open(p,'w').write('nested'); print('ok')")
        assert _run(agent, scratch_write).strip() == "ok"
        assert (scratch / "n.txt").read_text() == "nested"
    finally:
        agent.close()


def test_kernel_jail_staging_dir_is_cleaned_up(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    made = []
    real_mkdtemp = pytool.tempfile.mkdtemp

    def spy_mkdtemp(prefix=None):
        path = real_mkdtemp(prefix=prefix)
        if prefix == "cai-py-":
            made.append(path)
        return path
    monkeypatch.setattr(pytool.tempfile, "mkdtemp", spy_mkdtemp)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "print('ok')").strip() == "ok"
        assert len(made) == 1
        assert not os.path.exists(made[0])
    finally:
        agent.close()


def test_hook_mode_skips_the_kernel_jail(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "hook")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        # the host filesystem is visible again (the documented stat residual)...
        assert _run(agent, STAT_PROBE).strip() == "visible"
        # ...but the audit hook still confines opens
        assert "PermissionError" in _run(agent, "open('/etc/hostname')")
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
