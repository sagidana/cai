"""paths: scratch_dir + safe_path - how cai tool code finds and jails paths.

scratch_dir() is the one way any tool code locates the session scratch
directory (the place tools exchange binary/bulky intermediates): an MCP server
subprocess reads the CAI_SCRATCH env var cai spawned it with; an in-process
function tool reads the per-dispatch context ToolsRegistry brackets around its
call. Same accessor, both contexts - "" when no session scratch exists (e.g.
under a bare MCP client), so callers can fall back to tempfile.

safe_path confines a model-supplied path to the current working directory plus
that scratch directory, and expands a leading $CAI_SCRATCH (the stable name the
model uses to address scratch without knowing its real, per-session path).
Extra roots can be granted via the CAI_ALLOWED_PATHS env var (os.pathsep-joined
files or directories; the --allowed-paths CLI flag sets it), which spawned MCP
servers and the python-tool child inherit like CAI_SCRATCH. A directory grants
its whole subtree; a file grants just that file.
Exposed as cai.safe_path / cai.scratch_dir so an extension's MCP servers and
function tools share one implementation instead of vendoring copies. A server
file cai spawns runs under the same interpreter, so `from cai import safe_path`
always resolves. Stdlib-only."""
import os
from contextvars import ContextVar


_SCRATCH_VAR = "CAI_SCRATCH"
_ALLOWED_VAR = "CAI_ALLOWED_PATHS"


# the in-process scratch source: ToolsRegistry sets it to its provider (a
# zero-arg callable) for the duration of one function-tool call, so the dir is
# still only created when a tool actually asks. holding the provider - not the
# path - keeps laziness; a ContextVar - not a global - keeps two agents
# dispatching in two threads isolated.
_scratch_provider = ContextVar("cai_scratch_provider", default=None)


def scratch_dir():
    """the session scratch directory, or "" when there is none. reads the
    dispatch context inside a function tool, the CAI_SCRATCH env var inside a
    spawned MCP server. note: a thread a tool spawns itself starts with a fresh
    context - capture the value before spawning."""
    provider = _scratch_provider.get()
    if provider is not None:
        return provider()
    return os.environ.get("CAI_SCRATCH", "")


def _expand_scratch(user_path):
    """expand a leading $CAI_SCRATCH / ${CAI_SCRATCH} token in user_path to the
    real scratch directory (materializing it), so the model addresses scratch by
    name without knowing its per-session path. only a whole leading path segment
    is a token - '$CAI_SCRATCHED/x' is left alone. raises ValueError when the
    path names scratch but no session scratch exists."""
    for token in ("${" + _SCRATCH_VAR + "}", "$" + _SCRATCH_VAR):
        if user_path != token and not user_path.startswith(token + "/"):
            continue
        scratch = scratch_dir()
        if not scratch:
            raise ValueError(f"Error: no session scratch directory for {user_path!r}")
        return scratch + user_path[len(token):]
    return user_path


def allowed_paths():
    """the extra files/directories granted via CAI_ALLOWED_PATHS, realpath'd,
    or [] when the var is unset/empty."""
    raw = os.environ.get(_ALLOWED_VAR, "")
    roots = []
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        roots.append(os.path.realpath(entry))
    return roots


def _under(resolved, root):
    return resolved == root or resolved.startswith(root + os.sep)


def safe_path(user_path):
    """resolve user_path relative to the cwd and reject traversal outside it;
    the session scratch directory (scratch_dir(), when set) and the
    CAI_ALLOWED_PATHS grants are allowed too, and a leading $CAI_SCRATCH
    addresses scratch by name."""
    user_path = _expand_scratch(user_path)
    cwd = os.path.realpath(os.getcwd())
    resolved = os.path.realpath(os.path.join(cwd, user_path))
    if _under(resolved, cwd):
        return resolved
    scratch = scratch_dir()
    if scratch and _under(resolved, os.path.realpath(scratch)):
        return resolved
    for root in allowed_paths():
        if _under(resolved, root):
            return resolved
    raise ValueError(f"Error: path outside working directory: {user_path!r}")
