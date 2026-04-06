import os

CWD = os.path.realpath(".")


def safe_path(user_path: str) -> str:
    """Resolve user_path relative to CWD and reject any traversal outside it.

    Uses realpath so symlinks are resolved before the boundary check —
    a symlink inside CWD that points outside is also rejected.
    """
    resolved = os.path.realpath(os.path.join(CWD, user_path))
    if resolved != CWD and not resolved.startswith(CWD + os.sep):
        raise ValueError(f"Error: path outside working directory: {user_path!r}")
    return resolved
