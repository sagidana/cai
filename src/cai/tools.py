from mcp.server.fastmcp import FastMCP
import subprocess
import sys
import json
import os
import logging

logging.basicConfig(
    filename="/tmp/cai.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("cai.tools")

# ─── Language registry ─────────────────────────────────────────────────────────
#
# To add a new language:
#   1. Add one entry to LANGUAGE_CONFIGS below.
#   2. No other code changes are needed.
#
# Required capture names:
#   class_query  →  @class_def, @class_name, @method_name, @method_params
#   func_query   →  @func_name, @func_params
#
# Set class_query or func_query to None when the language has no classes /
# no top-level functions.
#
# Smali note: every .smali file is exactly one class.  The trick is to capture
# the root `source_file` node as @class_def — the ancestor walk from any method
# node then correctly terminates there.

LANGUAGE_CONFIGS = {
    'python': {
        'parser_name': 'python',
        'extensions': ['.py'],
        'method_prefix': 'def ',
        'class_query': """
            (class_definition
                name: (identifier) @class_name
                body: (block
                    (function_definition
                        name: (identifier) @method_name
                        parameters: (parameters) @method_params
                    )
                )
            ) @class_def
        """,
        'func_query': """
            (module
                (function_definition
                    name: (identifier) @func_name
                    parameters: (parameters) @func_params
                )
            )
        """,
    },

    'java': {
        'parser_name': 'java',
        'extensions': ['.java'],
        'method_prefix': '',
        'class_query': """
            (class_declaration
                name: (identifier) @class_name
                body: (class_body
                    (method_declaration
                        name: (identifier) @method_name
                        parameters: (formal_parameters) @method_params
                    )
                )
            ) @class_def
        """,
        'func_query': None,
    },

    'c': {
        'parser_name': 'c',
        'extensions': ['.c', '.h'],
        'method_prefix': '',
        'class_query': None,
        'func_query': """
            (translation_unit
                (function_definition
                    declarator: (function_declarator
                        declarator: (identifier) @func_name
                        parameters: (parameter_list) @func_params
                    )
                )
            )
        """,
    },

    'cpp': {
        'parser_name': 'cpp',
        'extensions': ['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.hh', '.h++'],
        'method_prefix': '',
        'class_query': """
            (class_specifier
                name: (type_identifier) @class_name
                body: (field_declaration_list
                    (function_definition
                        declarator: (function_declarator
                            declarator: (field_identifier) @method_name
                            parameters: (parameter_list) @method_params
                        )
                    )
                )
            ) @class_def
        """,
        'func_query': """
            (translation_unit
                (function_definition
                    declarator: (function_declarator
                        declarator: (identifier) @func_name
                        parameters: (parameter_list) @func_params
                    )
                )
            )
        """,
    },

    'smali': {
        'parser_name': 'smali',
        'extensions': ['.smali'],
        'method_prefix': '',
        # Capture source_file as @class_def so the ancestor walk from any
        # method node terminates at the file root, which holds the class name.
        'class_query': """
            (source_file
                (class_header
                    (class_identifier) @class_name
                )
                (method
                    (method_name) @method_name
                    (method_signature) @method_params
                )
            ) @class_def
        """,
        'func_query': None,
    },
}

# Flat extension → language name map built once at import time.
EXT_TO_LANG = {
    ext: lang
    for lang, cfg in LANGUAGE_CONFIGS.items()
    for ext in cfg['extensions']
}


def _parse_file(source: bytes, config: dict) -> dict:
    """Return {class: {method: prototype}, func: prototype} for one source file."""
    from tree_sitter_language_pack import get_language, get_parser
    import tree_sitter

    try:
        parser   = get_parser(config['parser_name'])
        language = get_language(config['parser_name'])
    except Exception:
        return {}

    root   = parser.parse(source).root_node
    prefix = config.get('method_prefix', '')
    info   = {}

    # ── Class methods ──────────────────────────────────────────────────────────
    cq_src = config.get('class_query')
    if cq_src:
        try:
            caps = tree_sitter.QueryCursor(language.query(cq_src)).captures(root)

            class_defs    = caps.get('class_def',     [])
            class_names   = caps.get('class_name',    [])
            method_names  = caps.get('method_name',   [])
            method_params = caps.get('method_params', [])

            # Build ID → class-name map (same node appears once per method, idempotent).
            class_node_map = {}
            for cls_node, cls_name_node in zip(class_defs, class_names):
                name = cls_name_node.text.decode()
                class_node_map[cls_node.id] = name
                info.setdefault(name, {})

            # Walk ancestors from each method node to find its owning class.
            for meth_node, param_node in zip(method_names, method_params):
                meth   = meth_node.text.decode()
                params = param_node.text.decode()
                anc    = meth_node.parent
                while anc is not None and anc.id not in class_node_map:
                    anc = anc.parent
                if anc is not None:
                    info[class_node_map[anc.id]][meth] = f"{prefix}{meth}{params}"
        except Exception:
            pass

    # ── Top-level functions ────────────────────────────────────────────────────
    fq_src = config.get('func_query')
    if fq_src:
        try:
            caps = tree_sitter.QueryCursor(language.query(fq_src)).captures(root)

            for fn_node, fp_node in zip(caps.get('func_name', []),
                                        caps.get('func_params', [])):
                name   = fn_node.text.decode()
                params = fp_node.text.decode()
                info[name] = f"{prefix}{name}{params}"
        except Exception:
            pass

    return info


def send_rpc(process, method, params, request_id):
    message = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params
    }
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())

def call_tool(tool_name, arguments):
    log.info("call_tool: %s %s", tool_name, arguments)
    process = subprocess.Popen(
        [sys.executable, "-m", "cai.tools"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        send_rpc(process,
                 "initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "manual-subproc-client", "version": "1.0" }
                 }, 1)


        process.stdin.write(json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized"
        }) + "\n")
        process.stdin.flush()

        response = send_rpc(process,
                            "tools/call", {
                                "name": tool_name,
                                "arguments": arguments
                            }, 2)

        result = response.get("result", {}).get("content", [{}])[0].get("text")
        # log.info("call_tool result: %s: %s", tool_name, result)
        return result
    except Exception as e:
        log.error("call_tool exception: %s", e)
    finally:
        process.terminate()

def get_tools():
    process = subprocess.Popen(
        [sys.executable, "-m", "cai.tools"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        send_rpc(process,
                 "initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "tool-lister", "version": "1.0"}
                }, 1)

        response = send_rpc(process, "tools/list", {}, 2)

        mcp_tools = response.get("result", {}).get("tools", [])
        openai_tools = []
        for tool in mcp_tools:
            openai_tools.append({
                "type": "function",
                "function":{
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                    }
                })
        return openai_tools
    finally:
        process.terminate()

def call_external_tool(server_path, tool_name, arguments):
    log.info("call_external_tool: %s %s %s", server_path, tool_name, arguments)
    process = subprocess.Popen(
        ["python", server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        send_rpc(process,
                 "initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "manual-subproc-client", "version": "1.0" }
                 }, 1)


        process.stdin.write(json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized"
        }) + "\n")
        process.stdin.flush()

        response = send_rpc(process,
                            "tools/call", {
                                "name": tool_name,
                                "arguments": arguments
                            }, 2)

        result = response.get("result", {}).get("content", [{}])[0].get("text")
        # log.info("call_external_tool result: %s: %s", tool_name, result)
        return result
    except Exception as e:
        log.error("call_external_tool exception: %s", e)
    finally:
        process.terminate()

def get_external_tools(server_path):
    process = subprocess.Popen(
        ["python", server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        send_rpc(process,
                 "initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "tool-lister", "version": "1.0"}
                }, 1)

        response = send_rpc(process, "tools/list", {}, 2)

        mcp_tools = response.get("result", {}).get("tools", [])
        openai_tools = []
        for tool in mcp_tools:
            openai_tools.append({
                "type": "function",
                "function":{
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                    }
                })
        return openai_tools
    finally:
        process.terminate()

if __name__ == '__main__':
    import re
    import stat
    import urllib.request
    from datetime import datetime

    mcp = FastMCP(name="Tools Server")

    CWD = os.path.realpath(".")

    def _safe_path(user_path: str) -> str:
        """Resolve user_path relative to CWD and reject any traversal outside it.

        Uses realpath so symlinks are resolved before the boundary check —
        a symlink inside CWD that points outside is also rejected.
        """
        resolved = os.path.realpath(os.path.join(CWD, user_path))
        if resolved != CWD and not resolved.startswith(CWD + os.sep):
            raise ValueError(f"Error: path outside working directory: {user_path!r}")
        return resolved

    # ── Low tier ───────────────────────────────────────────────────────────────

    @mcp.tool()
    def read(file_path: str) -> str:
        """Read and return the full contents of a file."""
        try:
            safe = _safe_path(file_path)
        except ValueError as e:
            return str(e)
        with open(safe, 'r') as f:
            return f.read()

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
            root = _safe_path(path)
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
                full = os.path.join(current, name)
                rel = os.path.relpath(full, root)
                kind = "dir" if os.path.isdir(full) else "file"
                if rx is None or rx.search(rel):
                    entries.append(f"{kind}  {rel}")
                if os.path.isdir(full):
                    queue.append(full)

        result = "\n".join(entries) if entries else "(empty)"
        return result

    @mcp.tool()
    def pattern_search(pattern: str, path: str = ".", file_glob: str = "") -> str:
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
        import subprocess
        try:
            safe_path = _safe_path(path)
        except ValueError as e:
            return str(e)
        cmd = ["rg", "--vimgrep", pattern]
        if file_glob:
            cmd += ["--glob", file_glob]
        cmd.append(safe_path)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout.strip()
            if result.returncode not in (0, 1):
                return f"Error: {result.stderr.strip() or 'rg exited with code ' + str(result.returncode)}"
            return output if output else "No matches found."
        except FileNotFoundError:
            return "Error: ripgrep (rg) is not installed or not on PATH."

    @mcp.tool()
    def git_log(file_path: str = "", n: int = 10) -> str:
        """Show the last n git commits. Pass file_path to scope to a specific file."""
        cmd = ["git", "log", f"-{n}", "--oneline", "--no-decorate"]
        if file_path:
            try:
                safe = _safe_path(file_path)
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
            safe = _safe_path(file_path)
        except ValueError as e:
            return str(e)
        result = subprocess.run(
            ["git", "blame", "--date=short", safe],
            capture_output=True, text=True
        )
        return result.stdout or result.stderr

    @mcp.tool()
    def symbol_search(symbol: str, path: str = ".") -> str:
        """Find definitions of a function or class named symbol across source files."""
        try:
            safe_path = _safe_path(path)
        except ValueError as e:
            return str(e)
        patterns = [
            rf"^\s*(def|class|function|func|fn)\s+{re.escape(symbol)}\b",
            rf"^\s*(public|private|protected|static).*\s+{re.escape(symbol)}\s*\(",
        ]
        rx = re.compile("|".join(patterns))
        results = []
        for root, dirs, files in os.walk(safe_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for filename in files:
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', errors='replace') as f:
                        for lineno, line in enumerate(f, 1):
                            if rx.search(line):
                                results.append(f"{filepath}:{lineno}: {line.rstrip()}")
                except (IOError, OSError):
                    continue
        return "\n".join(results) if results else f"No definition found for '{symbol}'."

    @mcp.tool()
    def fetch_url(url: str) -> str:
        """Fetch the content of a URL and return it as text (e.g. docs, READMEs)."""
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")

    @mcp.tool()
    def file_code_outline(file_path: str) -> str:
        """Return the class/method/function structure of a single source file as JSON."""
        try:
            safe = _safe_path(file_path)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        ext = os.path.splitext(safe)[1].lower()
        if ext not in EXT_TO_LANG:
            return json.dumps({"error": f"Unsupported file type: {ext}"})
        try:
            with open(safe, 'rb') as f:
                source = f.read()
        except (IOError, OSError) as e:
            return json.dumps({"error": str(e)})
        result = _parse_file(source, LANGUAGE_CONFIGS[EXT_TO_LANG[ext]])
        return json.dumps(result, indent=2)

    @mcp.tool()
    def project_code_outline() -> str:
        """Walk the working directory and extract class/method/function structure
        from Python, Java, C, C++, and Smali source files.  Language is
        auto-detected from file extension; mixed-language projects are handled
        transparently.  Returns JSON: {file: {class: {method: prototype}}}."""
        cwd = "."
        infra = {}
        for root, dirs, files in os.walk(cwd):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in EXT_TO_LANG:
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'rb') as f:
                        source = f.read()
                except (IOError, OSError):
                    continue
                info = _parse_file(source, LANGUAGE_CONFIGS[EXT_TO_LANG[ext]])
                if info:
                    infra[os.path.relpath(filepath, cwd)] = info
        return json.dumps(infra, indent=2)

    @mcp.tool()
    def create_file(file_path: str, content: str) -> str:
        """Create a new file at file_path with the given content.

        file_path must be relative to (or inside) the working directory;
        paths that escape it (e.g. '../../etc/cron.d/evil') are rejected.
        """
        try:
            safe = _safe_path(file_path)
        except ValueError as e:
            return str(e)
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        with open(safe, 'w') as f:
            f.write(content)
        return f"Created {file_path} ({len(content)} bytes)"

    @mcp.tool()
    def read_lines(file_path: str, start: int, end: int) -> str:
        """Read a specific range of lines from a file (1-based, inclusive).

        Args:
            file_path: Path relative to the working directory.
            start:     First line to return (1-based).
            end:       Last line to return (inclusive).

        Returns the requested lines with their line numbers prefixed, or an error string.
        """
        try:
            safe = _safe_path(file_path)
        except ValueError as e:
            return str(e)
        if start < 1:
            return "Error: start must be >= 1"
        if end < start:
            return "Error: end must be >= start"
        try:
            with open(safe, 'r') as f:
                lines = f.readlines()
        except (IOError, OSError) as e:
            return f"Error: {e}"
        slice_ = lines[start - 1:end]
        if not slice_:
            return f"Error: file has only {len(lines)} lines"
        return "".join(f"{start + i}\t{line}" for i, line in enumerate(slice_))

    @mcp.tool()
    def rename_file(src_path: str, dst_path: str) -> str:
        """Move or rename a file within the working directory.

        Both src_path and dst_path must be relative to (or inside) the working
        directory; paths that escape it are rejected.
        Parent directories of dst_path are created automatically.
        """
        try:
            safe_src = _safe_path(src_path)
            safe_dst = _safe_path(dst_path)
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
            safe = _safe_path(file_path)
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
            safe = _safe_path(file_path)
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

    mcp.run(transport="stdio")
