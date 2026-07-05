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
through the model. search covers binary files too - a binary hit comes back
as its byte offset plus a hexdump of the match with context.

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


def _hexline(chunk):
    """one xxd-style line body, no offset column: '7f45 4c46 ...  .ELF...'"""
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
    return f"{hexpart:<39}  {gutter}"


def _hexdump(data, offset):
    """xxd-style dump: '00000010: 7f45 4c46 ...  .ELF...', 16 bytes per line,
    offsets absolute so a paged read lines up with the previous page."""
    lines = []
    for i in range(0, len(data), 16):
        lines.append(f"{offset + i:08x}: {_hexline(data[i:i + 16])}")
    return "\n".join(lines)


def _paginate(lines, start, end, unit):
    """return a window of `lines` as text (100 lines by default), with a footer
    when more remain. start/end are 1-based and inclusive."""
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


def _rg(cmd):
    """run an rg command; returns (stdout_bytes, error) - exactly one is
    None. stdout stays bytes because binary match text can be anything."""
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except FileNotFoundError:
        return None, "Error: ripgrep (rg) is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return None, "Error: search timed out"
    if result.returncode not in (0, 1):
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        return None, f"Error: {stderr or 'rg exited with code ' + str(result.returncode)}"
    return result.stdout, None


def _binary_match_lines(path, offset, length):
    """render one binary match: a '<file>:<offset>: <n>-byte match' header,
    then an xxd-style hexdump of the match with 32 bytes of context either
    side (aligned to 16 so offsets line up with read_file pages)."""
    start = offset - 32
    if start < 0:
        start = 0
    start = start - (start % 16)
    end = offset + length + 32
    try:
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
    except (IOError, OSError):
        return []
    lines = [f"{path}:{offset}: {length}-byte match"]
    lines += _hexdump(data, start).split("\n")
    lines.append("")
    return lines


def _search_binary(pattern, safe, file_glob, binary_only):
    """rg pass over binary content: 'rg -a -o -b' reports each match as
    '<path>NUL<offset>:<match bytes>', parsed as bytes so the match length
    is exact. Returns (lines, error) - exactly one is None."""
    cmd = ["rg", "--text", "--only-matching", "--byte-offset",
           "--no-line-number", "--with-filename", "--null"]
    if file_glob:
        cmd += ["--glob", file_glob]
    cmd += ["--", pattern, safe]
    raw, err = _rg(cmd)
    if err:
        return None, err
    out = []
    binary_cache = {}
    for line in raw.split(b"\n"):
        if b"\x00" not in line:
            continue
        path, _, rest = line.partition(b"\x00")
        offset, sep, match = rest.partition(b":")
        if not sep or not offset.isdigit():
            continue
        path = path.decode("utf-8", errors="replace")
        if binary_only:
            known = binary_cache.get(path)
            if known is None:
                try:
                    known = _is_binary(path)
                except (IOError, OSError):
                    known = False
                binary_cache[path] = known
            if not known:
                continue
        out += _binary_match_lines(path, int(offset), len(match))
    return out, None


@mcp.tool()
def search(pattern: str,
           path: str = ".",
           file_glob: str = "",
           start: Optional[int] = None,
           end: Optional[int] = None) -> str:
    """Regex-search file CONTENTS - every file, text and binary alike (to
    find files by NAME use list_files, not this). A match in a text file is
    a vimgrep line <file>:<line>:<col>:<text>; a match in a binary file is
    a '<file>:<byte-offset>: <n>-byte match' line followed by an xxd-style
    hexdump of the match with 32 bytes of context either side.

    Args:
        pattern:   Regex (ripgrep syntax). In binary files it matches raw
                   bytes: e.g. '\\x7fELF', or '(?-u:\\xde\\xad\\xbe\\xef)'
                   for byte sequences that aren't valid UTF-8.
        path:      Dir or file to search (default ".").
        file_glob: Optional glob, e.g. "*.py".
        start/end: 1-based result-line window (default 1..100; paginate
                   with start=101 etc, or ask for a larger range).
    """
    try:
        safe = safe_path(path)
    except ValueError as e:
        return str(e)
    if not pattern:
        return "Error: empty pattern"

    lines = []
    single_binary = False
    if os.path.isfile(safe):
        try:
            single_binary = _is_binary(safe)
        except (IOError, OSError) as e:
            return f"Error: {e}"
    if not single_binary:
        cmd = ["rg", "--vimgrep", "--max-columns", "120", "--max-columns-preview"]
        if file_glob:
            cmd += ["--glob", file_glob]
        # '--' ends options so a pattern/path starting with '-' is never parsed as a flag.
        cmd += ["--", pattern, safe]
        raw, err = _rg(cmd)
        if err:
            return err
        output = raw.decode("utf-8", errors="replace").strip()
        if output:
            lines = output.split("\n")

    # second pass for the binary files the vimgrep pass skipped.
    bin_lines, err = _search_binary(pattern, safe, file_glob, binary_only=not single_binary)
    if err:
        return err
    lines += bin_lines
    if lines and lines[-1] == "":
        lines.pop()

    if not lines:
        return "No matches found."
    return _paginate(lines, start, end, "lines")


@mcp.tool()
def read_file(file_path: str,
              line_start: Optional[int] = None,
              line_end: Optional[int] = None,
              offset_start: Optional[int] = None,
              offset_end: Optional[int] = None) -> str:
    """Read a file: text as lines, binary (NUL in first 8KB) as an xxd-style
    hexdump. Only the requested range is read, so any file size is safe. With
    no range arguments: text shows 200 lines, binary the first 3200 bytes.

    Args:
        file_path:    Path to the file.
        line_start:   TEXT ONLY. First line, 1-based (default 1).
        line_end:     TEXT ONLY. Last line, inclusive (default line_start+199).
        offset_start: BINARY ONLY. First byte, 0-based (default 0).
        offset_end:   BINARY ONLY. End byte, exclusive (default offset_start+3200).
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
    """Recursively list files and directories under `path`, shallowest first -
    this is the tool for finding files by NAME (search looks inside files,
    not at names). Each line is "file  <rel-path>" or "dir  <rel-path>".

    Args:
        path:      Root directory (default ".").
        pattern:   Optional regex on the relative path - the way to find a
                   file by name, e.g. "conftest" or "\\.toml$" (traversal
                   still goes full-depth).
        start/end: 1-based result-line window (default 1..100; paginate
                   with start=101 etc, or ask for a larger range).
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
    """Create (or overwrite) a file with the given content. Parent directories
    are created automatically.

    Args:
        file_path: Path to the file.
        content:   The content to write, interpreted per `encoding`.
        encoding:  How `content` is decoded: 'text' (default, UTF-8), or
                   'hex' / 'base64' for binary files (hex ignores whitespace,
                   so hexdump-style '7f45 4c46' pastes straight in).
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

    Args:
        file_path:   Path to the file.
        old_text:    The span to replace. Must be UNIQUE in the file - text and
                     binary alike - or the edit is rejected; add surrounding
                     context to disambiguate, or pass replace_all=True.
        new_text:    The replacement; may be empty to delete old_text.
        replace_all: Replace EVERY occurrence of old_text instead of requiring
                     it to be unique (default False).
        encoding:    Must match the file: 'text' (default) for text files,
                     'hex' (whitespace ignored, hexdump-style '7f45 4c46'
                     works) for binary files (NUL in first 8KB) - a mismatch
                     is an error.
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
    """Rename a file. Parent directories of dst_path are created automatically."""
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
    """Move a file. If dst_path is an existing directory the file is placed
    inside it, keeping its name."""
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
    """Copy a file (byte-exact, works on binary). If dst_path is an existing
    directory the file is placed inside it, keeping its name."""
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
    """Copy a byte range from one file into another - the bytes never pass
    through the conversation. src and dst may be the same file.

    Source range: src_offset_start..src_offset_end (0-based, end exclusive),
    defaulting to the whole file. Destination, chosen by the dst offsets:

      - both omitted:          append to dst_path, creating it if missing
      - only dst_offset_start: overwrite dst starting there
      - only dst_offset_end:   overwrite dst ending there
      - both:                  replace dst[start:end] (sizes may differ -
                               the tail shifts)

    Patching (any dst offset given) requires dst_path to exist and the range
    to be within its size.
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
    """Remove a regular file (directories are never removed)."""
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
    """Create a directory and any missing parents ('mkdir -p' semantics)."""
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
    """Move or rename a directory. If dst_path is an existing directory the
    source is placed inside it (shell 'mv' behaviour)."""
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
