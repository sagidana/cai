import subprocess

from cai.utils import safe_path


def register(mcp):
    @mcp.tool()
    def git_log(file_path: str = "", n: int = 10) -> str:
        """Show the last n git commits. Pass file_path to scope to a specific file."""
        cmd = ["git", "log", f"-{n}", "--oneline", "--no-decorate"]
        if file_path:
            try:
                safe = safe_path(file_path)
            except ValueError as e:
                return str(e)
            cmd += ["--", safe]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout or result.stderr

    @mcp.tool()
    def git_diff(target: str = "HEAD") -> str:
        """Show a git diff. target can be a commit, branch, or 'HEAD' (default)."""
        result = subprocess.run(["git", "diff", target], capture_output=True, text=True)
        return result.stdout or result.stderr

    @mcp.tool()
    def git_blame(file_path: str) -> str:
        """Show git blame for a file — who last changed each line and when."""
        try:
            safe = safe_path(file_path)
        except ValueError as e:
            return str(e)
        result = subprocess.run(
            ["git", "blame", "--date=short", safe],
            capture_output=True, text=True
        )
        return result.stdout or result.stderr


if __name__ == "__main__":
    import sys

    class _MockMCP:
        def __init__(self): self._tools = {}
        def tool(self):
            def dec(fn): self._tools[fn.__name__] = fn; return fn
            return dec

    mcp = _MockMCP()
    register(mcp)
    T = mcp._tools

    _pass = _fail = 0
    def check(name, cond, got=""):
        global _pass, _fail
        if cond:
            print(f"  PASS  {name}")
            _pass += 1
        else:
            print(f"  FAIL  {name}  →  {got!r}")
            _fail += 1

    print("=== git_tools tests ===")

    # git_log default
    r = T["git_log"]()
    check("git_log returns output", bool(r.strip()), r)
    check("git_log format (hash + message)", len(r.splitlines()) > 0, r)

    # git_log scoped to a known file
    r = T["git_log"]("src/cai/tools.py", n=5)
    check("git_log scoped to file", bool(r.strip()), r)

    # git_log with safe_path traversal
    r = T["git_log"]("../../etc/passwd")
    check("git_log traversal rejected", r.startswith("Error:"), r)

    # git_diff HEAD (may be empty on clean tree, but must not error)
    r = T["git_diff"]("HEAD")
    check("git_diff HEAD no crash", not r.startswith("fatal:"), r)

    # git_diff invalid ref
    r = T["git_diff"]("refs/NONEXISTENT_BRANCH_XYZ")
    check("git_diff invalid ref returns output", bool(r.strip()), r)

    # git_blame a file that exists
    r = T["git_blame"]("src/cai/tools.py")
    check("git_blame known file", bool(r.strip()) and not r.startswith("Error:"), r)

    # git_blame traversal
    r = T["git_blame"]("../../etc/passwd")
    check("git_blame traversal rejected", r.startswith("Error:"), r)

    # git_blame missing file
    r = T["git_blame"]("src/cai/nonexistent_file_xyz.py")
    check("git_blame missing file returns output", bool(r.strip()), r)

    print(f"\n{_pass} passed, {_fail} failed")
    sys.exit(0 if _fail == 0 else 1)
