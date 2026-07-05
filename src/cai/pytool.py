"""pytool: the builtin `python` tool - run Python in cai's sandbox, with a
callback into the agent's own tools.

`python(code, timeout=60)` runs a snippet in a subprocess of a cai-MANAGED
virtualenv (config.venv_dir(), created lazily, empty by default - stdlib only,
so its package surface is well-defined and no compiled escape hatch ships in
it). Before the snippet runs, the child installs a sys.addaudithook jail - the
same confinement cai.safe_path gives the fs tools: writes only under the working
directory and the session scratch dir; subprocess/exec/fork, sockets, ctypes and
cffi blocked. A denied op raises PermissionError, whose traceback is the result.

The snippet is also handed a `call(name, **kwargs) -> str` builtin: it names one
of the agent's OWN selected tools, and the call is dispatched IN the cai process
(so every tool keeps its own confinement and reaches live agent state), through
the run's before/after_tool_call hooks (so a gate still vetoes it), and the
result string comes back into the snippet. Only what the snippet print()s becomes
the tool result - so a script can read a big tool result, reduce it in Python,
and return just the answer, the intermediate data never entering model context.

The tool is bound to its Agent (like the sub-agent tools) - that is what gives
call() a live dispatch. It is registered on every agent but only offered to the
model when the `python` skill selects it.

This is a guardrail, the same posture as safe_path - not a hard boundary against
hostile code. Run cai itself in a container for untrusted use."""

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


# the child bootstrap: read code from stdin, install the audit-hook jail, wire
# call() over the two inherited RPC fds, exec the code. stdlib-only so it runs
# under the managed venv. the jail mirrors cai.safe_path - cwd + CAI_SCRATCH for
# writes, plus the interpreter prefixes for reads (imports open files there).
# raw os.read/os.write on the inherited fds emit no audit events, so call()
# needs no exception carved into the jail.
_BOOTSTRAP = r'''
import sys
import os
import json

code = sys.stdin.read()

write_roots = [os.path.realpath(os.getcwd())]
scratch = os.environ.get("CAI_SCRATCH", "")
if scratch:
    write_roots.append(os.path.realpath(scratch))
read_roots = list(write_roots)
for prefix in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
    real = os.path.realpath(prefix)
    if real in read_roots: continue
    read_roots.append(real)

BLOCKED = (
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "pty.spawn",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.connect",
    "socket.bind",
    "socket.sendto",
    "socket.sendmsg",
)

WRITE_EVENTS = (
    "os.remove",
    "os.rename",
    "os.mkdir",
    "os.rmdir",
    "os.link",
    "os.symlink",
    "os.chmod",
    "os.chown",
    "os.truncate",
    "os.utime",
    "shutil.rmtree",
    "shutil.move",
)

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

in_hook = [False]


def under(path, roots):
    """True when path resolves into one of roots. non-path args (fds, None)
    pass - the open that produced the fd was already checked."""
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    if not isinstance(path, str):
        return True
    resolved = os.path.realpath(path)
    for root in roots:
        if resolved == root:
            return True
        if resolved.startswith(root + os.sep):
            return True
    return False


def wants_write(mode, flags):
    if isinstance(mode, str):
        for ch in "wax+":
            if ch in mode:
                return True
        return False
    if not isinstance(flags, int):
        return False
    return bool(flags & WRITE_FLAGS)


def check(event, args):
    if event == "import":
        if args[0] in ("ctypes", "_ctypes", "cffi", "_cffi_backend"):
            raise PermissionError(f"cai sandbox: {args[0]} is blocked")
        return
    if event in BLOCKED:
        raise PermissionError(f"cai sandbox: {event} is blocked")
    if event == "open":
        path, mode, flags = args
        roots = read_roots
        if wants_write(mode, flags):
            roots = write_roots
        if not under(path, roots):
            raise PermissionError(f"cai sandbox: open outside the working directory: {path!r}")
        return
    if event in WRITE_EVENTS:
        for arg in args:
            if under(arg, write_roots): continue
            raise PermissionError(f"cai sandbox: {event} outside the working directory: {arg!r}")


def hook(event, args):
    if in_hook[0]:
        return
    in_hook[0] = True
    try:
        check(event, args)
    finally:
        in_hook[0] = False


_RPC_RD = int(os.environ["CAI_PY_RPC_READ"])
_RPC_WR = int(os.environ["CAI_PY_RPC_WRITE"])
_rpc_buf = bytearray()


def _rpc_readline():
    while True:
        nl = _rpc_buf.find(b"\n")
        if nl >= 0:
            line = bytes(_rpc_buf[:nl])
            del _rpc_buf[:nl + 1]
            return line
        chunk = os.read(_RPC_RD, 65536)
        if not chunk:
            raise RuntimeError("cai sandbox: tool channel closed")
        _rpc_buf.extend(chunk)


def call(name, **kwargs):
    """dispatch one of the agent's own tools and return its result string. the
    call runs in the cai process, through cai's tool gates; only what you
    print() reaches the model, so reduce a big result here first."""
    request = {}
    request["name"] = name
    request["kwargs"] = kwargs
    os.write(_RPC_WR, json.dumps(request).encode("utf-8") + b"\n")
    reply = json.loads(_rpc_readline().decode("utf-8"))
    return reply["result"]


sys.addaudithook(hook)
exec(compile(code, "<python>", "exec"), {"__name__": "__main__", "call": call})
'''


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
    """the managed venv's python, materialized on first use. lazy and idempotent -
    created under a lock so two agents first-using at once don't collide."""
    python = config.venv_python()
    if _venv_usable(python):
        return python
    with _venv_lock:
        if _venv_usable(python):
            return python
        base = config.load_optional("python_base", sys.executable)
        shutil.rmtree(config.venv_dir(), ignore_errors=True)
        subprocess.run([base, "-m", "venv", config.venv_dir()], check=True)
        return config.venv_python()


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
        return f"Error: tool {name!r} is not available to call() - callable tools are the agent's selected ones"
    gate = hooks.current_gate()
    if gate is None:
        return agent.tools_registry.dispatch(name, kwargs)
    return hooks.gated_dispatch(gate, name, kwargs, call_id=PY_TOOL_NAME)


def _handle_request(agent, line, rep_w):
    """serve one child call() request: dispatch it and write the reply back."""
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
    """single-threaded driver: while the child runs, serve its call() requests on
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

    req_r, req_w = os.pipe()   # child -> parent: call() requests
    rep_r, rep_w = os.pipe()   # parent -> child: results
    child_env = dict(os.environ)
    if scratch:
        child_env["CAI_SCRATCH"] = scratch
    child_env["CAI_PY_RPC_READ"] = str(rep_r)
    child_env["CAI_PY_RPC_WRITE"] = str(req_w)

    out = tempfile.TemporaryFile()
    proc = None
    timed_out = False
    try:
        proc = subprocess.Popen([python, "-c", _BOOTSTRAP],
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

    Each call is a fresh interpreter - variables do NOT carry over between calls;
    persist state via files. print() what you want to see, e.g.
    code="print(2 ** 100)".

    The snippet also gets a call(name, **kwargs) builtin that runs one of YOUR
    OWN currently-available tools and returns its result as a string - the call
    executes in cai (respecting each tool's confinement and gates), so you can
    read a large tool result, process it in Python, and print() only the answer,
    keeping the intermediate data out of the conversation. Example:
    code="text = call('fs__read_file', file_path='big.log')\\nprint(text.count('ERROR'))".

    Sandbox: writing files is confined to the working directory plus the session
    scratch dir (os.environ['CAI_SCRATCH']); subprocess, network, ctypes and cffi
    are blocked; a denied op raises PermissionError. The interpreter is a managed
    virtualenv with the standard library only.

    Args:
        code:    Python source to execute.
        timeout: Seconds before the run is killed (default 60).
    """


def make_python(agent):
    """build the python tool bound to `agent`; call() reaches the agent's live
    tools/dispatch through the closure."""
    def python(code: str, timeout: int = 60) -> str:
        return _run_python(agent, code, timeout)
    python.__doc__ = _DOC
    return python


def python_tools(agent):
    """the python tool(s) bound to `agent` - the agent-tools factory analog of
    subagent_tools. registered unselected on every agent; the `python` skill
    selects it."""
    return [make_python(agent)]
