"""pytool: the builtin `python` tool - run Python in cai's sandbox, with a
callback into the agent's own tools.

`python(code, timeout=60)` runs a snippet in a subprocess of a cai-MANAGED
virtualenv (config.venv_dir(), created lazily, empty by default - stdlib only,
so its package surface is well-defined and no compiled escape hatch ships in
it). The sandbox is two layers, both set up by the child script
(pytool_bootstrap.py, fed to the interpreter as `python -c` source text):

KERNEL layer (the boundary; default, `python_sandbox: "kernel"`). Before the
snippet runs, the child enters fresh user + mount + network namespaces and
pivot_roots onto a tmpfs holding bind mounts of ONLY the interpreter prefixes,
the working directory, the session scratch dir and the system library dirs the
dynamic loader needs (so stdlib C extensions like zlib can dlopen libz & co
even when the interpreter is not /usr-based) - every other path does not
exist, at the kernel level, no matter how the snippet issues the syscall (this
also closes the stat-probe leak the audit hook alone had). The empty network
namespace has no interfaces (not even loopback), so no network, and abstract
unix sockets die with it too. The whole mount tree is flipped READ-ONLY
(mount_setattr, kernel >= 5.12), then the scratch dir alone is re-bound
read-write on top - the one writable island - before capabilities are
dropped, so the snippet can neither write outside scratch nor rearrange its
own jail: confinement AND the write policy are both kernel-enforced.

HOOK layer (UX + defense in depth). A sys.addaudithook jail enforces the same
policy in userspace: reads and directory listings are confined to cwd +
scratch + the interpreter prefixes, every write - create, modify, delete,
rename - is denied outside the scratch dir, as are subprocess/exec/fork,
sockets, ctypes and cffi. A denied op raises PermissionError, whose traceback
is the result (a mount-level EROFS would be raw and confusing). Audit hooks
are python-level and evadable by determined code - but anything that slips
past them is still inside the kernel jail, so the blast radius is the scratch
dir.

Hosts that forbid unprivileged user namespaces (e.g. default-hardened Docker
seccomp/AppArmor) fail closed with a clear message; there the operator - whose
container is then the boundary - may set `python_sandbox: "hook"` in
config.json to run with the hook layer only.

The snippet is also handed a `tool_call(name, **kwargs) -> str` builtin: it names one
of the agent's OWN selected tools, and the call is dispatched IN the cai process
(so every tool keeps its own confinement and reaches live agent state), through
the run's before/after_tool_call hooks (so a gate still vetoes it), and the
result string comes back into the snippet. Only what the snippet print()s becomes
the tool result - so a script can read a big tool result, reduce it in Python,
and return just the answer, the intermediate data never entering model context.
The RPC pipes are plain inherited fds, which namespaces do not sever - the
tool_call() channel is the one deliberate hole in the jail.

The tool is bound to its Agent (like the sub-agent tools) - that is what gives
tool_call() a live dispatch. It is registered on every agent but only offered to the
model when the `python` skill selects it."""

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cai
from cai import config
from cai import hooks


PY_TOOL_NAME = "python"


# the child bootstrap lives in pytool_bootstrap.py - a real module so it
# reads and edits like code - but it is executed as SOURCE TEXT via
# `python -c`, never imported: -c keeps sys.path[0] = cwd (so a snippet can
# import the project's own modules), and the file's side effects sit behind
# a __main__ guard, which -c satisfies.
def _bootstrap_source():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytool_bootstrap.py")
    with open(path) as f:
        return f.read()


def _truncate(text):
    """keep a long run's head and tail - tracebacks live at the tail."""
    limit = 20000
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return text[:half] + f"\n[... {omitted} chars omitted ...]\n" + text[-half:]


_venv_lock = threading.Lock()


def _venv_usable(python):
    """True when `python` exists and answers a trivial probe - guards against a
    half-built or stale venv."""
    if not os.path.exists(python):
        return False
    try:
        probe = subprocess.run([python, "-c", "import sys"],
                               capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def ensure_venv():
    """the venv's python, materialized on first use. lazy and idempotent -
    created under a lock so two agents first-using at once don't collide.

    a user-supplied `python_venv` is never built, rebuilt or deleted here: it
    is the user's own env (e.g. a pyenv one), so a broken one raises rather
    than getting silently wiped. only the cai-managed ~/.config/cai/venv is
    created."""
    python = config.venv_python()
    if _venv_usable(python):
        return python
    if config.load_optional("python_venv"):
        raise RuntimeError(
            f"configured python_venv interpreter is missing or unusable: {python}\n"
            "cai does not build or repair a user-supplied env - check the path, or "
            "drop the python_venv setting to use the managed venv.")
    with _venv_lock:
        if _venv_usable(python):
            return python
        base = config.load_optional("python_base", sys.executable)
        shutil.rmtree(config.venv_dir(), ignore_errors=True)
        subprocess.run([base, "-m", "venv", config.venv_dir()], check=True)
        return config.venv_python()


def install(packages):
    """`cai python install` - pip-install packages into the managed venv, from
    outside the sandbox (the jail has no network, so a snippet never can)."""
    python = ensure_venv()
    result = subprocess.run([python, "-m", "pip", "install"] + list(packages))
    return result.returncode


def uninstall(packages):
    """`cai python uninstall` - pip-uninstall packages from the managed venv.
    -y because there is no terminal to confirm on when scripted."""
    python = ensure_venv()
    result = subprocess.run([python, "-m", "pip", "uninstall", "-y"] + list(packages))
    return result.returncode


def list_packages():
    """`cai python list-packages` - the packages installed in the managed venv."""
    python = ensure_venv()
    result = subprocess.run([python, "-m", "pip", "list"])
    return result.returncode


def _dispatch_gated(agent, name, kwargs):
    """dispatch a tool the snippet asked for, on the agent's behalf. refuses the
    python tool itself (recursion) and any tool not currently offered to the
    model, then routes through the run gate so a before_tool_call gate can veto -
    falling back to a plain confined dispatch outside a run (SDK use)."""
    if not name:
        return "Error: tool request had no name"
    if name == PY_TOOL_NAME:
        return f"Error: {PY_TOOL_NAME} cannot call itself"
    if name not in agent.tools:
        return f"Error: tool {name!r} is not available to tool_call() - callable tools are the agent's selected ones"
    gate = hooks.current_gate()
    if gate is None:
        return agent.tools_registry.dispatch(name, kwargs)
    return hooks.gated_dispatch(gate, name, kwargs, call_id=PY_TOOL_NAME)


def _handle_request(agent, line, rep_w):
    """serve one child tool_call() request: dispatch it and write the reply back."""
    try:
        request = json.loads(line.decode("utf-8"))
    except ValueError:
        result = "Error: malformed tool request"
    else:
        result = _dispatch_gated(agent, request.get("name"), request.get("kwargs") or {})
    reply = {}
    reply["result"] = result
    os.write(rep_w, json.dumps(reply).encode("utf-8") + b"\n")


def _serve(agent, proc, req_r, rep_w, timeout):
    """single-threaded driver: while the child runs, serve its tool_call() requests on
    this (the run-loop) thread - so inner calls are dispatched exactly like
    top-level ones. returns True if the child was killed on timeout. the child's
    stdout/stderr go to a temp file, not a pipe, so this loop never blocks on
    output and only ever reads the request channel."""
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.kill()
            return True
        ready, _, _ = select.select([req_r], [], [], min(remaining, 0.5))
        if req_r not in ready:
            if proc.poll() is not None:
                return False
            continue
        chunk = os.read(req_r, 65536)
        if not chunk:
            return False
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(buf[:nl])
            del buf[:nl + 1]
            _handle_request(agent, line, rep_w)


def _run_python(agent, code, timeout):
    """spawn the sandboxed child, feed it the snippet, serve its tool calls, and
    return its combined stdout/stderr (truncated), with an exit-code footer."""
    python = ensure_venv()
    scratch = cai.scratch_dir()
    sandbox = config.load_optional("python_sandbox", "kernel")

    req_r, req_w = os.pipe()   # child -> parent: tool_call() requests
    rep_r, rep_w = os.pipe()   # parent -> child: results
    child_env = dict(os.environ)
    if scratch:
        child_env["CAI_SCRATCH"] = scratch
    child_env["CAI_PY_RPC_READ"] = str(rep_r)
    child_env["CAI_PY_RPC_WRITE"] = str(req_w)
    child_env["CAI_PY_SANDBOX"] = str(sandbox)
    # the agent's own selected tools, minus python itself: the child turns each
    # into a directly-callable function in the snippet's namespace (layer that
    # makes the modified env look ordinary - fs__read_file(...) just works),
    # resolved per call so it tracks the live selection.
    names = []
    for name in agent.tools:
        if name == PY_TOOL_NAME: continue
        names.append(name)
    child_env["CAI_PY_TOOLS"] = json.dumps(names)

    # the mount point the child's kernel jail pivots onto. the tmpfs and binds
    # exist only in the child's mount namespace - in ours it stays an empty dir,
    # removed when the child is done.
    staging = None
    if sandbox != "hook":
        staging = tempfile.mkdtemp(prefix="cai-py-")
        child_env["CAI_PY_STAGING"] = staging

    out = tempfile.TemporaryFile()
    proc = None
    timed_out = False
    try:
        proc = subprocess.Popen([python, "-c", _bootstrap_source()],
                                stdin=subprocess.PIPE,
                                stdout=out,
                                stderr=subprocess.STDOUT,
                                pass_fds=(req_w, rep_r),
                                env=child_env)
    finally:
        # the child owns its ends now; the parent keeps req_r/rep_w. closed even
        # on a spawn failure so no fd leaks.
        os.close(req_w)
        os.close(rep_r)

    try:
        proc.stdin.write(code.encode("utf-8"))
        proc.stdin.close()
        timed_out = _serve(agent, proc, req_r, rep_w, timeout)
        proc.wait()
    finally:
        os.close(req_r)
        os.close(rep_w)
        if staging:
            shutil.rmtree(staging, ignore_errors=True)

    if timed_out:
        return f"Error: run timed out after {timeout}s"

    out.seek(0)
    text = out.read().decode("utf-8", errors="replace")
    text = _truncate(text.rstrip("\n"))
    if proc.returncode != 0:
        if text:
            text += "\n"
        text += f"[exit code {proc.returncode}]"
    if not text:
        return "(no output)"
    return text


_DOC = """Run a Python snippet in cai's sandbox and return its output (stdout + stderr).

    One-shot: fresh interpreter every call, no variables survive between calls.

    YOUR OWN tools are callable right in the snippet: each is a plain function
    of the same name (e.g. fs__read_file(file_path='big.log')), and
    tool_call(name, **kwargs) does the same by name.

    Sandbox: the tool can read files and list directories under the working
    directory and the session scratch dir, and WRITE only under the scratch dir
    (os.environ['CAI_SCRATCH']) - everywhere else is read-only.

    To change or perform any action outside of the python interpreter or
    scratch - use the provided dedicated tools.

    Args:
        code:    Python source to execute.
        timeout: Seconds before the run is killed (default 60).
    """


def make_python(agent):
    """build the python tool bound to `agent`; tool_call() reaches the agent's live
    tools/dispatch through the closure."""
    def python(code: str, timeout: int = 60) -> str:
        return _run_python(agent, code, timeout)
    python.__doc__ = _DOC
    # the concrete tool list the model sees comes from the python skill's
    # {{tools}} slot (registry.signatures); python must not list itself there -
    # it cannot call itself (see _dispatch_gated).
    python._cai_hide_from_tool_list = True
    return python


def python_tools(agent):
    """the python tool(s) bound to `agent` - the agent-tools factory analog of
    subagent_tools. registered unselected on every agent; the `python` skill
    selects it."""
    return [make_python(agent)]
