"""paths: safe_path - the one path jail cai tools share.

A tool that takes a filesystem path from the model resolves it through
safe_path, which confines it to the current working directory plus the session
scratch directory ($CAI_SCRATCH, when set - the place tools exchange
binary/bulky intermediates; see ToolsRegistry). Exposed as cai.safe_path so an
extension's MCP servers and function tools share one implementation - with the
scratch semantics included - instead of vendoring copies. A server file cai
spawns runs under the same interpreter, so `from cai import safe_path` always
resolves. Stdlib-only."""
import os


def safe_path(user_path):
    """resolve user_path relative to the cwd and reject traversal outside it;
    the session scratch directory ($CAI_SCRATCH, when set) is allowed too."""
    cwd = os.path.realpath(os.getcwd())
    resolved = os.path.realpath(os.path.join(cwd, user_path))
    if resolved == cwd or resolved.startswith(cwd + os.sep):
        return resolved
    scratch = os.environ.get("CAI_SCRATCH", "")
    if scratch:
        scratch = os.path.realpath(scratch)
        if resolved == scratch or resolved.startswith(scratch + os.sep):
            return resolved
    raise ValueError(f"Error: path outside working directory: {user_path!r}")
