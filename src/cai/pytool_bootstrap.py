"""The child side of cai's `python` tool - the script the tool feeds to the
managed venv's interpreter (as `python -c` SOURCE TEXT, read by cai.pytool;
this module is never imported, and importing it by accident runs nothing).

It reads the snippet from stdin, enters the kernel namespace jail, installs
the audit-hook jail, wires call() over the two inherited RPC fds, and execs
the snippet. stdlib-only, so it runs under the empty managed venv.

The kernel jail (default) pivots into a mount namespace where ONLY cwd +
CAI_SCRATCH + the interpreter prefixes exist, inside an empty network
namespace. The hook jail is READ-ONLY: reads (and directory listings) are
confined to the same roots; ALL writes - anywhere, cwd and scratch included -
are denied, as are subprocess/exec/fork, sockets, ctypes and cffi. raw
os.read/os.write on the inherited fds emit no audit events and namespaces do
not sever inherited fds, so call() needs no exception in either jail."""
import sys
import os
import json


# the jail roots - what the snippet may see (kernel jail) and read (audit
# hook). computed by main() before either jail goes up.
read_roots = []


def compute_read_roots():
    roots = [os.path.realpath(os.getcwd())]
    scratch = os.environ.get("CAI_SCRATCH", "")
    if scratch:
        roots.append(os.path.realpath(scratch))
    for prefix in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
        real = os.path.realpath(prefix)
        if real in roots: continue
        roots.append(real)
    return roots


# --- kernel jail: user+mount+net namespaces --------------------------------
# the mount namespace is pivoted onto a tmpfs holding recursive bind mounts of
# read_roots ONLY - every other path does not exist at the kernel level, however
# the syscall is issued. the fresh network namespace has no interfaces (not even
# loopback) so there is no network, and abstract unix sockets are per-netns so
# they die with it. inherited fds (stdin, the RPC pipes, the stdout tempfile)
# are untouched. runs before the audit hook exists, so its own opens/mkdirs are
# unrestricted.

CLONE_NEWNS = 0x00020000
CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000
MS_REC = 16384
MS_PRIVATE = 1 << 18
MS_BIND = 4096
MNT_DETACH = 2
PR_CAPBSET_DROP = 24
PR_SET_NO_NEW_PRIVS = 38
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
EINVAL = 22
PIVOT_ROOT_NR = {}
PIVOT_ROOT_NR["x86_64"] = 155
PIVOT_ROOT_NR["aarch64"] = 41


def bind_roots():
    """read_roots with nested paths folded away - a recursive bind of a parent
    already carries its children."""
    ordered = sorted(read_roots, key=len)
    kept = []
    for root in ordered:
        covered = False
        for prev in kept:
            if root == prev or root.startswith(prev + os.sep):
                covered = True
                break
        if covered: continue
        kept.append(root)
    return kept


def drop_caps(ctypes, libc, need):
    """the namespaces above were built with the full capabilities a fresh user
    namespace grants - the snippet must not keep them, or it could rearrange its
    own jail (remount, umount its binds)."""
    need(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")
    libc.prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0)
    cap = 0
    while cap < 64:
        rc = libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0)
        if rc != 0 and ctypes.get_errno() == EINVAL:
            break
        cap += 1

    class CapHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class CapData(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32),
                    ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]

    header = CapHeader()
    header.version = 0x20080522
    header.pid = 0
    data = (CapData * 2)()
    need(libc.capset(ctypes.byref(header), data), "capset")


def enter_kernel_jail():
    import ctypes
    staging = os.environ.get("CAI_PY_STAGING", "")
    if not staging:
        raise RuntimeError("no staging directory was provided")
    machine = os.uname().machine
    if machine not in PIVOT_ROOT_NR:
        raise RuntimeError(f"unsupported architecture {machine!r}")
    libc = ctypes.CDLL(None, use_errno=True)

    def need(rc, what):
        if rc == 0:
            return
        err = ctypes.get_errno()
        raise OSError(err, what + ": " + os.strerror(err))

    uid = os.getuid()
    gid = os.getgid()
    cwd = os.path.realpath(os.getcwd())
    need(libc.unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWNET), "unshare")
    with open("/proc/self/setgroups", "w") as f:
        f.write("deny")
    with open("/proc/self/uid_map", "w") as f:
        f.write(f"{uid} {uid} 1")
    with open("/proc/self/gid_map", "w") as f:
        f.write(f"{gid} {gid} 1")
    need(libc.mount(b"none", b"/", None, MS_REC | MS_PRIVATE, None), "make / private")
    need(libc.mount(b"tmpfs", staging.encode(), b"tmpfs", 0, None), "mount tmpfs")
    for root in bind_roots():
        target = staging + root
        os.makedirs(target, exist_ok=True)
        need(libc.mount(root.encode(), target.encode(), None, MS_BIND | MS_REC, None),
             f"bind {root}")
    os.chdir(staging)
    need(libc.syscall(PIVOT_ROOT_NR[machine], b".", b"."), "pivot_root")
    need(libc.umount2(b".", MNT_DETACH), "detach old root")
    os.chdir(cwd)
    drop_caps(ctypes, libc, need)
    # this bootstrap imported ctypes; evict it so the snippet's own import still
    # hits the audit hook instead of the sys.modules cache.
    for name in list(sys.modules):
        if name == "_ctypes" or name == "ctypes" or name.startswith("ctypes."):
            del sys.modules[name]


# --- hook jail: the read-only audit hook ------------------------------------

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

# listing a directory's entries is still a read - confine it to read_roots the
# same way open-for-read is, or the jail leaks the whole filesystem tree
# (os.listdir/scandir, and glob/os.walk/pathlib.iterdir which build on scandir).
# note the stat family (os.stat, os.path.exists/getsize) emits NO audit event -
# under the kernel jail that no longer matters (the paths don't exist), but the
# hook-only fallback keeps the historic existence/size residual.
READ_EVENTS = (
    "os.listdir",
    "os.scandir",
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
        if wants_write(mode, flags):
            raise PermissionError(f"cai sandbox: the python tool is read-only - cannot open {path!r} for writing")
        if not under(path, read_roots):
            raise PermissionError(f"cai sandbox: open outside the working directory: {path!r}")
        return
    if event in READ_EVENTS:
        for arg in args:
            if under(arg, read_roots): continue
            raise PermissionError(f"cai sandbox: {event} outside the working directory: {arg!r}")
        return
    if event in WRITE_EVENTS:
        raise PermissionError(f"cai sandbox: the python tool is read-only - {event} is disabled")


def hook(event, args):
    if in_hook[0]:
        return
    in_hook[0] = True
    try:
        check(event, args)
    finally:
        in_hook[0] = False


# --- the call() tool proxy ---------------------------------------------------

_RPC_RD = -1
_RPC_WR = -1
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


def main():
    global read_roots, _RPC_RD, _RPC_WR

    code = sys.stdin.read()
    read_roots = compute_read_roots()

    if os.environ.get("CAI_PY_SANDBOX", "kernel") != "hook":
        try:
            enter_kernel_jail()
        except Exception as e:
            sys.stderr.write(
                f"cai sandbox: could not enter the kernel namespace jail: {e}\n"
                "this host may not allow unprivileged user namespaces (common under\n"
                "default-hardened containers). if the environment itself is already a\n"
                'boundary, set "python_sandbox": "hook" in ~/.config/cai/config.json\n'
                "to run with the audit-hook sandbox only.\n")
            sys.exit(97)

    _RPC_RD = int(os.environ["CAI_PY_RPC_READ"])
    _RPC_WR = int(os.environ["CAI_PY_RPC_WRITE"])

    sys.addaudithook(hook)
    exec(compile(code, "<python>", "exec"), {"__name__": "__main__", "call": call})


if __name__ == "__main__":
    main()
