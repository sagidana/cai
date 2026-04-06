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
