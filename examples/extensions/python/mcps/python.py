"""python.py - example MCP server: run Python inside cai's sandbox.

A FastMCP stdio server exposing one tool, surfaced as ``python__run``: execute
a Python snippet in a fresh subprocess of this same interpreter. Before the
snippet runs, the subprocess installs a sys.addaudithook jail - audit hooks
cannot be removed once installed, so the snippet executes entirely inside it.
The jail enforces the same confinement cai.safe_path gives the fs tools:

  - writes (open for write, remove, rename, mkdir, ...) only under the
    working directory and the session scratch dir ($CAI_SCRATCH)
  - reads additionally allow the interpreter's own install prefixes, so
    imports of stdlib and site-packages keep working
  - subprocess / exec / fork, sockets, and ctypes are blocked outright -
    each is a trivial way around a python-level jail

A blocked operation raises PermissionError inside the snippet; the traceback
comes back as the tool result, so the model sees exactly what was denied.
This is a guardrail against accidents - the same posture as safe_path - not
a hard security boundary against hostile code.

Each call is one-shot: variables do not survive between runs; state persists
via files in the working directory or scratch."""

import sys
import subprocess

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="python")


# runs in the child before the snippet: read the code from stdin, install the
# audit-hook jail, exec the code. stdlib-only, so it works under any
# sys.executable. the roots mirror cai.safe_path: cwd + CAI_SCRATCH for
# writes, plus the interpreter prefixes for reads (imports open files there).
_BOOTSTRAP = r'''
import sys
import os

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
        if args[0] in ("ctypes", "_ctypes"):
            raise PermissionError("cai sandbox: ctypes is blocked")
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


sys.addaudithook(hook)
exec(compile(code, "<python__run>", "exec"), {"__name__": "__main__"})
'''


def _truncate(text):
    """keep a long run's head and tail - tracebacks live at the tail."""
    limit = 20000
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return text[:half] + f"\n[... {omitted} chars omitted ...]\n" + text[-half:]


@mcp.tool()
def run(code: str, timeout: int = 60) -> str:
    """Run a Python snippet and return its output (stdout + stderr combined).
    Each call is a fresh interpreter - variables do NOT carry over between
    calls; persist state via files. print() what you need to see, e.g.
    code="print(2 ** 100)".

    The snippet runs inside cai's sandbox: writing files is confined to the
    working directory plus the session scratch dir (address it as
    os.environ['CAI_SCRATCH']); subprocesses, network and ctypes are blocked.
    A denied operation raises PermissionError.

    Args:
        code:    Python source to execute.
        timeout: Seconds before the run is killed (default 60).
    """
    try:
        result = subprocess.run([sys.executable, "-c", _BOOTSTRAP],
                                input=code.encode("utf-8"),
                                capture_output=True,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Error: run timed out after {timeout}s"
    text = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    if stderr:
        if text and not text.endswith("\n"):
            text += "\n"
        text += stderr
    text = _truncate(text.rstrip("\n"))
    if result.returncode != 0:
        if text:
            text += "\n"
        text += f"[exit code {result.returncode}]"
    if not text:
        return "(no output)"
    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")
