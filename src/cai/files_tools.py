import json
import os
import re
import subprocess

from cai.utils import safe_path

def register(mcp):
    @mcp.tool()
    def read_file(file_path: str, line_start: int = None, line_end: int = None) -> str:
        """Read the contents of a file and return them as text.

        Think of this like opening a book. Normally you get the whole book, but
        you can also say "just show me pages 10 to 20" by using line_start and
        line_end.

        Args:
            file_path:  Path to the file you want to read, relative to the
                        working directory. For example: "src/main.py" or
                        "README.md". Paths that try to escape the working
                        directory (e.g. "../../etc/passwd") are rejected.

            line_start: The first line you want to see (counting from 1, like
                        normal people count). Optional — leave it out to start
                        from the very beginning of the file.
                        If the number is bigger than the total number of lines
                        in the file, you get back an empty string.

            line_end:   The last line you want to see (inclusive — that line IS
                        included in the result). Optional — leave it out to read
                        all the way to the end of the file.
                        If the number is bigger than the total number of lines,
                        it is quietly clamped to the last line so you still get
                        everything up to the end without an error.

        Returns:
            The text of the file (or the requested slice of it).
            Returns an error string starting with "Error:" if the path is
            invalid or the file cannot be opened.

        Examples:
            read_file("foo.py")           → whole file
            read_file("foo.py", 10)       → line 10 to end
            read_file("foo.py", 10, 20)   → lines 10–20 (inclusive)
            read_file("foo.py", 1, 5)     → first 5 lines
            read_file("foo.py", 999)      → "" (file has fewer than 999 lines)
        """
        try:
            safe = safe_path(file_path)
        except ValueError as e:
            return str(e)
        try:
            with open(safe, 'r') as f:
                if line_start is None and line_end is None:
                    return f.read()
                lines = f.readlines()
            total = len(lines)
            start_idx = (line_start - 1) if line_start is not None else 0
            end_idx   = line_end          if line_end   is not None else total
            # clamp end, reject start that is entirely out of range
            end_idx   = min(end_idx, total)
            if start_idx >= total or start_idx < 0:
                return ""
            return "".join(lines[start_idx:end_idx])
        except (IOError, OSError) as e:
            return f"Error: {e}"


    @mcp.tool()
    def list_files(path: str = ".", pattern: str = "") -> str:
        """Recursively list all files and directories under the given path, ordered by depth
        (breadth-first: root-level entries first, then their children, then grandchildren, etc.).

        Each line is one entry in the format:
            [file|dir]  <relative-path>

        This is the right tool when you need a complete picture of a directory tree — for example,
        to understand a project's structure, find where certain file types live, audit what exists
        before making changes, or pass a full file listing to another tool or prompt.

        Unlike a simple ls, this tool descends into every subdirectory so nothing is hidden from
        view. Unlike find/walk output, entries are sorted by depth first so the top-level layout is
        immediately visible at the top of the output without scrolling past deeply-nested paths.

        When a pattern is supplied, only entries whose relative path matches the regex are included
        in the output — useful for narrowing results to specific extensions, naming conventions, or
        path segments (e.g. r"\\.py$" for Python files, "test" for anything test-related,
        "src/.*\\.ts$" for TypeScript files under src/).  Directory traversal always goes full-depth
        regardless of the filter so nested matches are never missed.

        Args:
            path:    Root directory to list. Defaults to "." (current working directory).
                     Accepts absolute or relative paths.
            pattern: Optional regex pattern (Python re syntax) to filter results. Matched against
                     the relative path of each entry. Leave empty to return all entries.

        Returns:
            One entry per line: "file  <rel-path>" or "dir  <rel-path>".
            Returns "(empty)" if the directory contains no entries (or none match the pattern).
            Returns "Error: invalid pattern: ..." if the regex cannot be compiled.
        """
        from collections import deque

        try:
            root = safe_path(path)
        except ValueError as e:
            return str(e)

        rx = None
        if pattern:
            try:
                rx = re.compile(pattern)
            except re.error as e:
                return f"Error: invalid pattern: {e}"

        entries = []
        queue = deque([root])

        while queue:
            current = queue.popleft()
            try:
                children = sorted(os.listdir(current))
            except PermissionError:
                continue
            for name in children:
                if name in (".git", "__pycache__"):
                    continue
                full = os.path.join(current, name)
                rel = os.path.relpath(full, root)
                kind = "dir" if os.path.isdir(full) else "file"
                if rx is None or rx.search(rel):
                    entries.append(f"{kind}  {rel}")
                if os.path.isdir(full):
                    queue.append(full)

        return "\n".join(entries) if entries else "(empty)"

    @mcp.tool()
    def search(pattern: str, path: str = ".", file_glob: str = "") -> str:
        """Search for a regex pattern across files using ripgrep (rg), returning results in
        vimgrep format: one match per line as  <file>:<line>:<col>:<matched line text>.

        This is the right tool when you need to locate where something is defined or used —
        for example, finding a function definition, tracing where a variable is referenced,
        auditing log messages, or searching for a string across a large codebase. It is fast
        (ripgrep-backed), unicode-aware, and automatically skips binary files, hidden files,
        and paths listed in .gitignore.

        Args:
            pattern:   Regular expression to search for (ripgrep / Rust regex syntax).
                       Examples: "def train", "TODO|FIXME", r"class\\w+Model".
            path:      Directory or file to search in. Defaults to "." (current working dir).
            file_glob: Optional glob to restrict which files are searched, e.g. "*.py",
                       "**/*.{ts,tsx}", "src/**/*.rs". Leave empty to search all files.

        Returns:
            One match per line in vimgrep format: filepath:line:col:text
            Returns "No matches found." when the pattern does not match anything.
            Returns an error message prefixed with "Error:" if rg is unavailable or fails.
        """
        try:
            safe = safe_path(path)
        except ValueError as e:
            return str(e)
        cmd = ["rg", "--vimgrep", pattern]
        if file_glob:
            cmd += ["--glob", file_glob]
        cmd.append(safe)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout.strip()
            if result.returncode not in (0, 1):
                return f"Error: {result.stderr.strip() or 'rg exited with code ' + str(result.returncode)}"
            return output if output else "No matches found."
        except FileNotFoundError:
            return "Error: ripgrep (rg) is not installed or not on PATH."

    @mcp.tool()
    def create_file(file_path: str, content: str) -> str:
        """Create a new file at file_path with the given content.

        file_path must be relative to (or inside) the working directory;
        paths that escape it (e.g. '../../etc/cron.d/evil') are rejected.
        """
        try:
            safe = safe_path(file_path)
        except ValueError as e:
            return str(e)
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        with open(safe, 'w') as f:
            f.write(content)
        return f"Created {file_path} ({len(content)} bytes)"

    @mcp.tool()
    def rename_file(src_path: str, dst_path: str) -> str:
        """Move or rename a file within the working directory.

        Both src_path and dst_path must be relative to (or inside) the working
        directory; paths that escape it are rejected.
        Parent directories of dst_path are created automatically.
        """
        try:
            safe_src = safe_path(src_path)
            safe_dst = safe_path(dst_path)
        except ValueError as e:
            return str(e)
        if not os.path.exists(safe_src):
            return f"Error: source not found: {src_path}"
        if os.path.isdir(safe_src):
            return f"Error: {src_path!r} is a directory, not a file"
        os.makedirs(os.path.dirname(safe_dst), exist_ok=True)
        os.rename(safe_src, safe_dst)
        return f"Renamed {src_path} -> {dst_path}"

    @mcp.tool()
    def remove_file(file_path: str) -> str:
        """Remove a file at file_path.

        file_path must be relative to (or inside) the working directory;
        paths that escape it (e.g. '../../etc/passwd') are rejected.
        Directories are not removed — only regular files.
        """
        try:
            safe = safe_path(file_path)
        except ValueError as e:
            return str(e)
        if not os.path.exists(safe):
            return f"Error: file not found: {file_path}"
        if os.path.isdir(safe):
            return f"Error: {file_path!r} is a directory, not a file"
        os.remove(safe)
        return f"Removed {file_path}"

    @mcp.tool()
    def edit_file(file_path: str, old_text: str, new_text: str) -> str:
        """Replace the first occurrence of old_text with new_text in a file.

        file_path must be relative to (or inside) the working directory;
        paths that escape it are rejected.
        """
        try:
            safe = safe_path(file_path)
        except ValueError as e:
            return str(e)
        with open(safe, 'r') as f:
            original = f.read()
        if old_text not in original:
            return f"Error: old_text not found in {file_path}"
        updated = original.replace(old_text, new_text, 1)
        with open(safe, 'w') as f:
            f.write(updated)
        return f"Replaced 1 occurrence in {file_path}"


if __name__ == "__main__":
    import sys
    import tempfile

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

    print("=== files_tools tests ===")

    # Use a temp subdir inside CWD so safe_path accepts relative paths
    tmp = tempfile.mkdtemp(dir=".")
    rel = os.path.relpath(tmp)

    try:
        # create_file
        r = T["create_file"](f"{rel}/hello.txt", "line1\nline2\nline3\n")
        check("create_file", "Created" in r, r)

        # read_file - whole file
        r = T["read_file"](f"{rel}/hello.txt")
        check("read_file whole", r == "line1\nline2\nline3\n", r)

        # read_file - line range
        r = T["read_file"](f"{rel}/hello.txt", 2, 3)
        check("read_file line range", r == "line2\nline3\n", r)

        # read_file - start beyond EOF
        r = T["read_file"](f"{rel}/hello.txt", 999)
        check("read_file beyond EOF", r == "", r)

        # list_files
        r = T["list_files"](rel)
        check("list_files", "hello.txt" in r, r)

        # list_files with pattern
        r = T["list_files"](rel, r"\.txt$")
        check("list_files pattern match", "hello.txt" in r, r)

        r = T["list_files"](rel, r"\.py$")
        check("list_files pattern no match", r == "(empty)", r)

        # search
        r = T["search"]("line2", rel)
        if "rg) is not installed" in r:
            print("  SKIP  search found (rg not on PATH)")
            print("  SKIP  search no match (rg not on PATH)")
        else:
            check("search found", "line2" in r, r)
            r2 = T["search"]("zzznomatch", rel)
            check("search no match", "No matches" in r2, r2)

        # edit_file
        r = T["edit_file"](f"{rel}/hello.txt", "line1", "changed")
        check("edit_file", "Replaced 1 occurrence" in r, r)

        r = T["read_file"](f"{rel}/hello.txt", 1, 1)
        check("edit_file content updated", "changed" in r, r)

        r = T["edit_file"](f"{rel}/hello.txt", "zzznomatch", "x")
        check("edit_file old_text not found", r.startswith("Error:"), r)

        # rename_file
        r = T["rename_file"](f"{rel}/hello.txt", f"{rel}/renamed.txt")
        check("rename_file", "Renamed" in r, r)

        check("rename_file source gone", not os.path.exists(f"{tmp}/hello.txt"))
        check("rename_file dest exists", os.path.exists(f"{tmp}/renamed.txt"))

        # remove_file
        r = T["remove_file"](f"{rel}/renamed.txt")
        check("remove_file", "Removed" in r, r)

        r = T["remove_file"](f"{rel}/renamed.txt")
        check("remove_file missing", r.startswith("Error:"), r)

        # safe_path rejection
        r = T["read_file"]("../../etc/passwd")
        check("safe_path traversal rejected", r.startswith("Error:"), r)

        r = T["create_file"]("../outside.txt", "x")
        check("create_file traversal rejected", r.startswith("Error:"), r)

    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{_pass} passed, {_fail} failed")
    sys.exit(0 if _fail == 0 else 1)
