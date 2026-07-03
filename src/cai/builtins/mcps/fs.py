"""fs.py - the built-in file-system MCP server, shipped with cai.

A FastMCP stdio server (stdlib + cai.safe_path). Its tools are surfaced
prefixed with the file name: ``fs__search``, ``fs__read_file``,
``fs__list_files`` (read-only), and ``fs__create_file``, ``fs__edit_file``,
``fs__rename_file``, ``fs__move_file``, ``fs__copy_file``, ``fs__copy_bytes``,
``fs__remove_file``, ``fs__create_directory``, ``fs__move_directory``
(read-write).

Binary files (NUL in the first 8KB) are first-class under the same tools:
read_file returns an xxd-style hexdump addressed by byte offsets, create_file
takes hex/base64 content via encoding=, edit_file replaces byte spans via
encoding='hex', copy_file is byte-exact anyway, and copy_bytes moves a byte
range between files (extract, append, patch) without the bytes ever passing
through the model. search stays text-only.

Pair them with the ``fs-read-only`` skill for inspection, or the ``fs`` skill
which adds the mutating tools. Every path is confined via ``cai.safe_path`` to
the current working directory - plus the session scratch directory when cai
hands one down as ``CAI_SCRATCH``, so scratch artifacts (bulky tool outputs,
binary intermediates) stay searchable and readable with these same tools.
"""

import os
import re
import shutil
from collections import deque
from typing import Optional

from mcp.server.fastmcp import FastMCP

from cai import safe_path

mcp = FastMCP(name="fs")


def _is_binary(safe):
    """the same sniff grep/git use: binary if the first 8KB contain a NUL.
    (UTF-16 text classifies as binary - the hexdump gutter makes that obvious.)"""
    with open(safe, "rb") as f:
        chunk = f.read(8192)
    return b"\x00" in chunk


def _hexdump(data, offset):
    """xxd-style dump: '00000010: 7f45 4c46 ...  .ELF...', 16 bytes per line,
    offsets absolute so a paged read lines up with the previous page."""
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        groups = []
        for j in range(0, len(chunk), 2):
            groups.append(chunk[j:j + 2].hex())
        hexpart = " ".join(groups)
        gutter = ""
        for byte in chunk:
            if 32 <= byte < 127:
                gutter += chr(byte)
            else:
                gutter += "."
        lines.append(f"{offset + i:08x}: {hexpart:<39}  {gutter}")
    return "\n".join(lines)


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

    Text only: a recursive search skips binary files; naming a binary file
    directly reports 'binary file matches' without content - inspect it with
    read_file (offset_start/offset_end give an xxd-style hexdump) instead.

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
def read_file(file_path: str,
              line_start: Optional[int] = None,
              line_end: Optional[int] = None,
              offset_start: Optional[int] = None,
              offset_end: Optional[int] = None) -> str:
    """Read a file: text is returned as lines, binary as an xxd-style hexdump.

    A file is treated as binary if its first 8KB contain a NUL byte. A call with
    no range arguments works on any file: text shows a 200-line window, binary
    the first 3200 bytes (200 dump lines). Only the requested range is read off
    disk, so this is safe on very large files.

    Args:
        file_path:    Path to the file, relative to the working directory.
        line_start:   TEXT ONLY. First line to read (1-based). Defaults to 1.
        line_end:     TEXT ONLY. Last line to read (inclusive). Defaults to
                      line_start + 199.
        offset_start: BINARY ONLY. First byte to read (0-based). Defaults to 0.
        offset_end:   BINARY ONLY. End byte (exclusive). Defaults to
                      offset_start + 3200.
    """
    try:
        safe = safe_path(file_path)
    except ValueError as e:
        return str(e)
    wants_lines = line_start is not None or line_end is not None
    wants_offsets = offset_start is not None or offset_end is not None
    if wants_lines and wants_offsets:
        return ("Error: give line_start/line_end (text) or "
                "offset_start/offset_end (binary), not both")
    try:
        binary = _is_binary(safe)
    except (IOError, OSError) as e:
        return f"Error: {e}"
    if binary and wants_lines:
        return (f"Error: {file_path} is binary - read it with "
                f"offset_start/offset_end (byte offsets)")
    if not binary and wants_offsets:
        return f"Error: {file_path} is text - read it with line_start/line_end"
    if binary:
        return _read_binary(safe, offset_start, offset_end)
    return _read_text(safe, line_start, line_end)


def _read_text(safe, line_start, line_end):
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


def _read_binary(safe, offset_start, offset_end):
    start = offset_start
    if start is None:
        start = 0
    if start < 0:
        start = 0
    end = offset_end
    if end is None:
        end = start + 3200
    if end <= start:
        return ""
    try:
        size = os.path.getsize(safe)
        with open(safe, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
    except (IOError, OSError) as e:
        return f"Error: {e}"
    if not data:
        return f"[no bytes at offset {start}; file is {size} bytes]"
    dump = _hexdump(data, start)
    shown = start + len(data)
    if shown < size:
        dump += f"\n[bytes {start}-{shown} of {size}; call again with offset_start={shown}]"
    return dump


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


def _decode_content(content, encoding):
    """decode a create/edit payload to bytes per its declared encoding; returns
    (data, error_string_or_None). hex tolerates whitespace, so hexdump-styled
    input ('7f45 4c46') pastes straight in."""
    if encoding == "text":
        return content.encode("utf-8"), None
    if encoding == "hex":
        try:
            return bytes.fromhex(content), None
        except ValueError as e:
            return None, f"Error: invalid hex content: {e}"
    if encoding == "base64":
        import base64
        try:
            return base64.b64decode(content, validate=True), None
        except Exception as e:
            return None, f"Error: invalid base64 content: {e}"
    return None, f"Error: unknown encoding {encoding!r} (use 'text', 'hex' or 'base64')"


@mcp.tool()
def create_file(file_path: str, content: str, encoding: str = "text") -> str:
    """Create (or overwrite) a file at file_path with the given content.

    encoding declares how `content` is to be interpreted: 'text' (default,
    written as UTF-8), 'hex' (pairs of hex digits, whitespace ignored - hexdump
    style '7f45 4c46' works) or 'base64' - use hex/base64 to write binary files.
    file_path must be inside the working directory; paths that escape it are
    rejected. Parent directories are created automatically.
    """
    try:
        safe = safe_path(file_path)
    except ValueError as e:
        return str(e)
    data, err = _decode_content(content, encoding)
    if err:
        return err
    os.makedirs(os.path.dirname(safe), exist_ok=True)
    with open(safe, "wb") as f:
        f.write(data)
    return f"Created {file_path} ({len(data)} bytes)"


@mcp.tool()
def edit_file(file_path: str, old_text: str, new_text: str,
              replace_all: bool = False, encoding: str = "text") -> str:
    """Replace old_text with new_text in a file.

    encoding declares how old_text/new_text are to be interpreted: 'text'
    (default) for text files, 'hex' (pairs of hex digits, whitespace ignored -
    hexdump style '7f45 4c46' works) for binary files. A text edit on a binary
    file, or a hex edit on a text file, is an error - a file is binary if its
    first 8KB contain a NUL byte, matching read_file.

    old_text must identify a UNIQUE span: if it occurs more than once the edit is
    rejected (add surrounding context to make it unique, or pass replace_all=True).
    file_path must be inside the working directory.

    Args:
        file_path:   File to edit, relative to the working directory.
        old_text:    Exact content to find. Must be unique unless replace_all.
        new_text:    Replacement content (may be empty to delete).
        replace_all: When True, replace every occurrence.
        encoding:    'text' (default) or 'hex'.
    """
    try:
        safe = safe_path(file_path)
    except ValueError as e:
        return str(e)
    if encoding not in ("text", "hex"):
        return f"Error: unknown encoding {encoding!r} (use 'text' or 'hex')"
    try:
        binary = _is_binary(safe)
        with open(safe, "rb") as f:
            original = f.read()
    except (IOError, OSError) as e:
        return f"Error: {e}"
    if binary and encoding == "text":
        return f"Error: {file_path} is binary - edit it with encoding='hex'"
    if not binary and encoding == "hex":
        return f"Error: {file_path} is text - edit it with encoding='text'"
    old, err = _decode_content(old_text, encoding)
    if err:
        return err
    new, err = _decode_content(new_text, encoding)
    if err:
        return err
    if not old:
        return "Error: old_text is empty"
    count = original.count(old)
    if count == 0:
        return f"Error: old_text not found in {file_path}"
    if old == new:
        return f"Error: old_text and new_text are identical - nothing to change in {file_path}"
    if count > 1 and not replace_all:
        return (f"Error: old_text is ambiguous - found {count} occurrences in "
                f"{file_path}. Add surrounding context to make it unique, "
                f"or pass replace_all=True to replace all of them.")
    updated = original.replace(old, new)
    with open(safe, "wb") as f:
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
    """Copy a file (text or binary, byte-exact) within the working directory.

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
def copy_bytes(src_path: str, dst_path: str,
               src_offset_start: Optional[int] = None,
               src_offset_end: Optional[int] = None,
               dst_offset_start: Optional[int] = None,
               dst_offset_end: Optional[int] = None) -> str:
    """Copy a byte range from one file into another - bytes move directly
    between the files, never through the conversation. src and dst may be the
    same file.

    The source range defaults to the whole file: src_offset_start omitted
    starts at the beginning, src_offset_end omitted reads to the end (offsets
    0-based, end exclusive). Where the bytes land is chosen by the dst offsets:

      - both omitted:          append to dst_path, creating it if missing
      - only dst_offset_start: the bytes overwrite dst starting there
      - only dst_offset_end:   the bytes overwrite dst ending there
      - both:                  the bytes replace dst[start:end] (sizes may
                               differ - the tail shifts)

    Patching (any dst offset given) requires dst_path to exist and the replace
    range to be within its size. Returns a message naming what happened
    (created / appended / replaced).
    """
    try:
        safe_src = safe_path(src_path)
        safe_dst = safe_path(dst_path)
    except ValueError as e:
        return str(e)

    try:
        src_size = os.path.getsize(safe_src)
    except (IOError, OSError) as e:
        return f"Error: {e}"
    a = src_offset_start
    if a is None:
        a = 0
    if a < 0:
        a = 0
    b = src_offset_end
    if b is None:
        b = src_size
    if b > src_size:
        b = src_size
    if b <= a:
        return f"Error: empty source range [{a}:{b}] ({src_path} is {src_size} bytes)"
    try:
        with open(safe_src, "rb") as f:
            f.seek(a)
            data = f.read(b - a)
    except (IOError, OSError) as e:
        return f"Error: {e}"

    if os.path.isdir(safe_dst):
        return f"Error: {dst_path!r} is a directory, not a file"

    if dst_offset_start is None and dst_offset_end is None:
        existed = os.path.exists(safe_dst)
        os.makedirs(os.path.dirname(safe_dst), exist_ok=True)
        with open(safe_dst, "ab") as f:
            f.write(data)
        if existed:
            return f"Appended {len(data)} bytes to {dst_path} (from {src_path}[{a}:{b}])"
        return f"Created {dst_path} ({len(data)} bytes from {src_path}[{a}:{b}])"

    if not os.path.isfile(safe_dst):
        return f"Error: {dst_path} does not exist - omit the dst offsets to create it"
    try:
        with open(safe_dst, "rb") as f:
            original = f.read()
    except (IOError, OSError) as e:
        return f"Error: {e}"
    dst_size = len(original)
    x = dst_offset_start
    y = dst_offset_end
    if x is None:
        x = y - len(data)
    if y is None:
        y = x + len(data)
    if x < 0 or y < x or y > dst_size:
        return f"Error: replace range [{x}:{y}] is outside {dst_path} ({dst_size} bytes)"
    updated = original[:x] + data + original[y:]
    with open(safe_dst, "wb") as f:
        f.write(updated)
    return (f"Replaced {dst_path}[{x}:{y}] ({y - x} bytes) with "
            f"{len(data)} bytes from {src_path}[{a}:{b}]")


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
