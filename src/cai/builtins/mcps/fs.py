"""fs.py - the built-in file-system MCP server, shipped with cai.

A self-contained FastMCP stdio server (stdlib only). Its tools are surfaced
prefixed with the file name: ``fs__search``, ``fs__read_file``,
``fs__list_files`` (read-only), and ``fs__create_file``, ``fs__edit_file``,
``fs__rename_file``, ``fs__move_file``, ``fs__copy_file``,
``fs__remove_file``, ``fs__create_directory``, ``fs__move_directory``
(read-write).

Pair them with the ``fs-read-only`` skill for inspection, or the ``fs`` skill
which adds the mutating tools. Every path is confined to the current working
directory via an inlined ``safe_path`` - traversal outside it is rejected.
"""

import os
import re
import shutil
from collections import deque
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="fs")


def safe_path(user_path):
    """resolve user_path relative to the cwd and reject traversal outside it."""
    cwd = os.path.realpath(os.getcwd())
    resolved = os.path.realpath(os.path.join(cwd, user_path))
    if resolved != cwd and not resolved.startswith(cwd + os.sep):
        raise ValueError(f"Error: path outside working directory: {user_path!r}")
    return resolved


def _paginate(lines, start, end, unit):
    """return a 100-line window of `lines` as text, with a footer when more
    remain. start/end are 1-based and inclusive."""
    total = len(lines)
    if start is None:
        start = 1
    if start < 1:
        start = 1
    if end is None:
        end = start + 99
    window = lines[start - 1:end]
    text = "\n".join(window)
    if end < total:
        shown = min(end, total)
        text += f"\n[Showing {start}-{shown} of {total} {unit}; call again with start={end + 1}]"
    return text


@mcp.tool()
def search(pattern: str, path: str = ".", file_glob: str = "",
           start: Optional[int] = None, end: Optional[int] = None) -> str:
    """Search for a regex pattern across files using ripgrep (rg), returning results in
    vimgrep format: one match per line as  <file>:<line>:<col>:<matched line text>.

    This is the right tool to locate where something is defined or used. At most 100
    result lines are returned per call (lines 1-100 by default); paginate with
    start/end (e.g. start=101, end=200 for the next page).

    Args:
        pattern:   Regular expression to search for (ripgrep / Rust regex syntax).
        path:      Directory or file to search in. Defaults to "." (cwd).
        file_glob: Optional glob restricting which files are searched, e.g. "*.py".
        start:     1-based index of the first result line to return (default 1).
        end:       1-based index of the last result line (default start + 99).
    """
    import subprocess

    try:
        safe = safe_path(path)
    except ValueError as e:
        return str(e)

    cmd = ["rg", "--vimgrep", "--max-columns", "120", "--max-columns-preview"]
    if file_glob:
        cmd += ["--glob", file_glob]
    # '--' ends options so a pattern/path starting with '-' is never parsed as a flag.
    cmd += ["--", pattern, safe]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return "Error: ripgrep (rg) is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    if result.returncode not in (0, 1):
        return f"Error: {result.stderr.strip() or 'rg exited with code ' + str(result.returncode)}"
    output = result.stdout.strip()
    if not output:
        return "No matches found."
    return _paginate(output.split("\n"), start, end, "matches")


@mcp.tool()
def read_file(file_path: str, line_start: Optional[int] = None,
              line_end: Optional[int] = None) -> str:
    """Read the contents of a file and return them as text.

    Reads a 200-line window by default; use line_start/line_end to move the window
    (only the requested lines are read off disk, so this is safe on very large files).

    Args:
        file_path:  Path to the file, relative to the working directory.
        line_start: First line to read (1-based). Defaults to 1.
        line_end:   Last line to read (inclusive). Defaults to line_start + 199.
    """
    try:
        safe = safe_path(file_path)
    except ValueError as e:
        return str(e)
    start = line_start
    if start is None:
        start = 1
    if start < 1:
        start = 1
    end = line_end
    if end is None:
        end = start + 199
    if end < start:
        return ""
    out = []
    more = False
    try:
        with open(safe, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if i < start:
                    continue
                if i > end:
                    more = True
                    break
                out.append(line)
    except (IOError, OSError) as e:
        return f"Error: {e}"
    if not out:
        return ""
    text = "".join(out)
    if more:
        text += f"\n[lines {start}-{end} shown; file has more - call again with line_start={end + 1}]"
    return text


@mcp.tool()
def list_files(path: str = ".", pattern: str = "",
               start: Optional[int] = None, end: Optional[int] = None) -> str:
    """Recursively list files and directories under `path`, shallowest first.

    Each line is "file  <rel-path>" or "dir  <rel-path>". At most 100 lines are
    returned per call; paginate with start/end. Pass a regex `pattern` to filter
    entries by their relative path (traversal still goes full-depth).

    Args:
        path:    Root directory to list. Defaults to "." (cwd).
        pattern: Optional regex (Python re) matched against each entry's rel path.
        start:   1-based index of the first result line (default 1).
        end:     1-based index of the last result line (default start + 99).
    """
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
            if name in (".git", "__pycache__"): continue
            full = os.path.join(current, name)
            rel = os.path.relpath(full, root)
            kind = "file"
            if os.path.isdir(full):
                kind = "dir"
            if rx is None or rx.search(rel):
                entries.append(f"{kind}  {rel}")
            if os.path.isdir(full):
                queue.append(full)

    if not entries:
        return "(empty)"

    def depth_then_name(entry):
        rel = entry.split("  ", 1)[1]
        return rel.count(os.sep), rel

    entries.sort(key=depth_then_name)
    return _paginate(entries, start, end, "entries")


@mcp.tool()
def create_file(file_path: str, content: str) -> str:
    """Create (or overwrite) a file at file_path with the given content.

    file_path must be inside the working directory; paths that escape it are
    rejected. Parent directories are created automatically.
    """
    try:
        safe = safe_path(file_path)
    except ValueError as e:
        return str(e)
    os.makedirs(os.path.dirname(safe), exist_ok=True)
    with open(safe, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Created {file_path} ({len(content)} bytes)"


@mcp.tool()
def edit_file(file_path: str, old_text: str, new_text: str,
              replace_all: bool = False) -> str:
    """Replace old_text with new_text in a file.

    old_text must identify a UNIQUE span: if it occurs more than once the edit is
    rejected (add surrounding context to make it unique, or pass replace_all=True).
    file_path must be inside the working directory.

    Args:
        file_path:   File to edit, relative to the working directory.
        old_text:    Exact text to find. Must be unique unless replace_all.
        new_text:    Replacement text.
        replace_all: When True, replace every occurrence.
    """
    try:
        safe = safe_path(file_path)
    except ValueError as e:
        return str(e)
    try:
        with open(safe, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
    except (IOError, OSError) as e:
        return f"Error: {e}"
    count = original.count(old_text)
    if count == 0:
        return f"Error: old_text not found in {file_path}"
    if old_text == new_text:
        return f"Error: old_text and new_text are identical - nothing to change in {file_path}"
    if count > 1 and not replace_all:
        return (f"Error: old_text is ambiguous - found {count} occurrences in "
                f"{file_path}. Add surrounding context to make it unique, "
                f"or pass replace_all=True to replace all of them.")
    updated = original.replace(old_text, new_text)
    with open(safe, "w", encoding="utf-8") as f:
        f.write(updated)
    plural = "s"
    if count == 1:
        plural = ""
    return f"Replaced {count} occurrence{plural} in {file_path}"


@mcp.tool()
def rename_file(src_path: str, dst_path: str) -> str:
    """Move or rename a file within the working directory.

    Both paths must be inside the working directory; parent directories of
    dst_path are created automatically.
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
def move_file(src_path: str, dst_path: str) -> str:
    """Move a file to a new location within the working directory.

    If dst_path is an existing directory the file is placed inside it keeping its
    original name. Both paths must be inside the working directory.
    """
    try:
        safe_src = safe_path(src_path)
        safe_dst = safe_path(dst_path)
    except ValueError as e:
        return str(e)
    if not os.path.exists(safe_src):
        return f"Error: source not found: {src_path}"
    if os.path.isdir(safe_src):
        return f"Error: {src_path!r} is a directory, not a file - use move_directory instead"
    if os.path.isdir(safe_dst):
        safe_dst = os.path.join(safe_dst, os.path.basename(safe_src))
    os.makedirs(os.path.dirname(safe_dst), exist_ok=True)
    shutil.move(safe_src, safe_dst)
    dst_rel = os.path.relpath(safe_dst, os.path.realpath("."))
    return f"Moved {src_path} -> {dst_rel}"


@mcp.tool()
def copy_file(src_path: str, dst_path: str) -> str:
    """Copy a file to a new location within the working directory.

    If dst_path is an existing directory the file is placed inside it keeping its
    original name. Both paths must be inside the working directory; parent
    directories of dst_path are created automatically.
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
    if os.path.isdir(safe_dst):
        safe_dst = os.path.join(safe_dst, os.path.basename(safe_src))
    os.makedirs(os.path.dirname(safe_dst), exist_ok=True)
    shutil.copy2(safe_src, safe_dst)
    dst_rel = os.path.relpath(safe_dst, os.path.realpath("."))
    return f"Copied {src_path} -> {dst_rel}"


@mcp.tool()
def remove_file(file_path: str) -> str:
    """Remove a regular file at file_path (directories are never removed).

    file_path must be inside the working directory; paths that escape it are
    rejected.
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
def create_directory(dir_path: str) -> str:
    """Create a directory and any missing parents ('mkdir -p' semantics).

    dir_path must be inside the working directory; paths that escape it are
    rejected.
    """
    try:
        safe = safe_path(dir_path)
    except ValueError as e:
        return str(e)
    already_existed = os.path.isdir(safe)
    os.makedirs(safe, exist_ok=True)
    if already_existed:
        return f"Directory already exists: {dir_path}"
    return f"Created directory {dir_path}"


@mcp.tool()
def move_directory(src_path: str, dst_path: str) -> str:
    """Move or rename a directory within the working directory.

    If dst_path is an existing directory the source is placed inside it (shell
    'mv' behaviour). Both paths must be inside the working directory.
    """
    try:
        safe_src = safe_path(src_path)
        safe_dst = safe_path(dst_path)
    except ValueError as e:
        return str(e)
    if not os.path.exists(safe_src):
        return f"Error: source not found: {src_path}"
    if not os.path.isdir(safe_src):
        return f"Error: {src_path!r} is not a directory - use move_file instead"
    real_src = os.path.realpath(safe_src)
    real_dst = os.path.realpath(safe_dst)
    if real_dst == real_src or real_dst.startswith(real_src + os.sep):
        return f"Error: cannot move {src_path!r} into itself"
    os.makedirs(os.path.dirname(safe_dst), exist_ok=True)
    shutil.move(safe_src, safe_dst)
    dst_rel = os.path.relpath(real_dst, os.path.realpath("."))
    return f"Moved {src_path} -> {dst_rel}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
