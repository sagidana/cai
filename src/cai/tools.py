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
                if name in (".git", "__pycache__"):
                    continue
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

    # ── Smali XRef Tools ──────────────────────────────────────────────────────
    #
    # Four tools for cross-reference analysis of apktool/baksmali smali codebases.
    # All use ripgrep for fast pre-filtering then tree-sitter for precise parsing.
    # Node types from the smali grammar:
    #   class_definition > class_directive > class_identifier, access_modifiers
    #   class_definition > super_directive > class_identifier
    #   class_definition > implements_directive > class_identifier
    #   class_definition > method_definition > method_signature > method_identifier
    #   method_definition > expression > opcode, body > full_method_signature

    def _smali_parse(fpath: str):
        """Parse a smali file. Returns (root_node, lines_list)."""
        from tree_sitter_language_pack import get_parser
        with open(fpath, 'rb') as _f:
            _src = _f.read()
        _root = get_parser('smali').parse(_src).root_node
        return _root, _src.decode('utf-8', errors='replace').splitlines()

    def _smali_walk(node, node_type: str):
        """Yield all descendant nodes of node_type (depth-first)."""
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type == node_type:
                yield n
            stack.extend(reversed(n.children))

    def _smali_class_info(root):
        """Return (class_descriptor, modifiers_set) from a parsed smali root."""
        class_dir = next((c for c in root.children if c.type == 'class_directive'), None)
        if not class_dir:
            return '', set()
        class_id = next((c for c in class_dir.children if c.type == 'class_identifier'), None)
        class_desc = class_id.text.decode() if class_id else ''
        access = next((c for c in class_dir.children if c.type == 'access_modifiers'), None)
        mods = set()
        if access:
            for mod in access.children:
                if mod.type == 'access_modifier':
                    mods.add(mod.text.decode())
        return class_desc, mods

    def _smali_method_sig(method_node):
        """Return the method_signature text from a method_definition node."""
        sig = next((c for c in method_node.children if c.type == 'method_signature'), None)
        return sig.text.decode() if sig else ''

    def _smali_enclosing_method(node):
        """Walk up the AST to find the nearest method_definition ancestor."""
        n = node.parent
        while n is not None:
            if n.type == 'method_definition':
                return n
            n = n.parent
        return None

    def _smali_rg_files(needle: str, safe_path: str) -> list:
        """Return smali files whose text contains needle (fixed-string, via rg)."""
        r = subprocess.run(
            ['rg', '--fixed-strings', '-l', needle, '--glob', '*.smali', safe_path],
            capture_output=True, text=True
        )
        return [f.strip() for f in r.stdout.splitlines() if f.strip()] if r.returncode in (0, 1) else []

    _RUNNABLE_PREFIXES = (
        'Ljava/lang/Runnable;->run()',
        'Ljava/util/concurrent/Callable;->call()',
        'Landroid/os/Handler;->post(',
        'Landroid/os/Handler;->postDelayed(',
        'Ljava/util/concurrent/ExecutorService;->submit(',
        'Ljava/util/concurrent/ExecutorService;->execute(',
        'Ljava/lang/Thread;->start()',
        'Landroid/os/AsyncTask;->execute(',
        'Landroid/os/AsyncTask;->executeOnExecutor(',
        'Ljava/util/concurrent/ScheduledExecutorService;->schedule(',
    )

    @mcp.tool()
    def smali_find_callers(descriptor: str, path: str = ".") -> str:
        """Find all smali methods that invoke the given method descriptor via any invoke-* opcode.

        Searches the smali codebase for every call site that targets descriptor.
        Uses ripgrep for fast pre-filtering, then tree-sitter for precise structural parsing
        so caller context (enclosing class and method) is always correct.

        Args:
            descriptor: Full or partial smali method descriptor.
                        Full: "Lcom/example/Foo;->bar(I)V"
                        Partial filter: "->encrypt(" or "Lcom/example/Crypto;"
            path:       Directory to search. Defaults to "." (working directory).

        Returns:
            JSON array of caller objects:
              caller_file       — relative path to the smali file
              caller_class      — class descriptor of the caller ("Lcom/example/Bar;")
              caller_method     — method signature of the caller ("doWork(I)V")
              caller_descriptor — full caller descriptor
              line              — 1-based line number of the invoke instruction
              invoke_line       — raw invoke instruction text
              invoke_kind       — "virtual", "static", "interface", "direct", or "super"
            Returns "[]" if no callers found. Returns "Error: ..." on path violations.
        """
        try:
            safe_path = _safe_path(path)
        except ValueError as e:
            return str(e)

        callers = []
        for fpath in _smali_rg_files(descriptor, safe_path):
            try:
                root, lines = _smali_parse(fpath)
            except Exception:
                continue
            class_desc, _ = _smali_class_info(root)
            for expr in _smali_walk(root, 'expression'):
                opcode_node = next((c for c in expr.children if c.type == 'opcode'), None)
                if opcode_node is None:
                    continue
                opcode_text = opcode_node.text.decode()
                if not opcode_text.startswith('invoke-'):
                    continue
                body_node = next((c for c in expr.children if c.type == 'body'), None)
                if body_node is None:
                    continue
                full_sig = next((c for c in body_node.children if c.type == 'full_method_signature'), None)
                if full_sig is None or descriptor not in full_sig.text.decode():
                    continue
                lineno = expr.start_point[0] + 1
                invoke_line = lines[lineno - 1].strip() if lineno <= len(lines) else ''
                method_node = _smali_enclosing_method(expr)
                method_sig = _smali_method_sig(method_node) if method_node else ''
                kind = opcode_text[len('invoke-'):]
                callers.append({
                    'caller_file': os.path.relpath(fpath, safe_path),
                    'caller_class': class_desc,
                    'caller_method': method_sig,
                    'caller_descriptor': f'{class_desc}->{method_sig}',
                    'line': lineno,
                    'invoke_line': invoke_line,
                    'invoke_kind': kind,
                })
        return json.dumps(callers)

    @mcp.tool()
    def smali_find_callees(descriptor: str, path: str = ".") -> str:
        """Find all methods invoked from within the body of the given smali method.

        Args:
            descriptor: Full smali method descriptor: "Lcom/example/Foo;->bar(I)V".
                        The class part is used to locate the smali file.
            path:       Directory to search. Defaults to "." (working directory).

        Returns:
            JSON array of callee objects:
              opcode           — the full invoke opcode (e.g. "invoke-virtual")
              callee_descriptor — full callee descriptor
              callee_class      — class part of the callee
              callee_method     — method signature of the callee
              line              — 1-based line number of the invoke instruction
              is_runnable_like  — true if callee looks like a deferred/async execution target
                                  (Runnable.run, Handler.post, ExecutorService.submit, etc.)
            Returns "Error: ..." on failure or if method not found.
        """
        try:
            safe_path = _safe_path(path)
        except ValueError as e:
            return str(e)

        if '->' not in descriptor:
            return f'Error: descriptor must contain "->": {descriptor}'
        class_part, method_part = descriptor.split('->', 1)
        if not class_part.endswith(';'):
            class_part += ';'

        callees = []
        found_method = False
        for fpath in _smali_rg_files(class_part, safe_path):
            try:
                root, lines = _smali_parse(fpath)
            except Exception:
                continue
            class_desc, _ = _smali_class_info(root)
            if class_desc != class_part:
                continue
            # Find matching method_definition
            target_method = None
            for m in _smali_walk(root, 'method_definition'):
                sig = _smali_method_sig(m)
                if sig == method_part or sig.startswith(method_part.split('(')[0] + '('):
                    target_method = m
                    break
            if target_method is None:
                continue
            found_method = True
            for expr in _smali_walk(target_method, 'expression'):
                opcode_node = next((c for c in expr.children if c.type == 'opcode'), None)
                if opcode_node is None:
                    continue
                opcode_text = opcode_node.text.decode()
                if not opcode_text.startswith('invoke-'):
                    continue
                body_node = next((c for c in expr.children if c.type == 'body'), None)
                if body_node is None:
                    continue
                full_sig = next((c for c in body_node.children if c.type == 'full_method_signature'), None)
                if full_sig is None:
                    continue
                callee_full = full_sig.text.decode()
                callee_class_node = next((c for c in full_sig.children if c.type == 'class_identifier'), None)
                callee_sig_node = next((c for c in full_sig.children if c.type == 'method_signature'), None)
                callee_class = callee_class_node.text.decode() if callee_class_node else ''
                callee_method = callee_sig_node.text.decode() if callee_sig_node else ''
                lineno = expr.start_point[0] + 1
                is_runnable = any(callee_full.startswith(p) for p in _RUNNABLE_PREFIXES)
                callees.append({
                    'opcode': opcode_text,
                    'callee_descriptor': callee_full,
                    'callee_class': callee_class,
                    'callee_method': callee_method,
                    'line': lineno,
                    'is_runnable_like': is_runnable,
                })
        if not found_method:
            return f'Error: method not found: {descriptor}'
        return json.dumps(callees)

    @mcp.tool()
    def smali_resolve_descriptor(query: str, path: str = ".") -> str:
        """Resolve a partial class or method name to full smali descriptors.

        Useful when you know "MainActivity" or "encrypt" but need the full descriptor
        "Lcom/example/app/MainActivity;->encrypt(I)V". Handles obfuscated names
        (e.g. "La;") by returning all candidates — use signature to disambiguate.

        Args:
            query: Partial class name, method name, package path, or full/partial descriptor.
                   If already a full smali descriptor (starts with "L", contains "->"),
                   returned as-is. Case-insensitive matching.
            path:  Directory to search. Defaults to "." (working directory).

        Returns:
            JSON array (up to 20 results):
              descriptor     — full smali descriptor ("Lclass;->method(sig)ret" or "Lclass;")
              file           — relative path to the smali file
              kind           — "class" or "method"
              class_modifier — "concrete", "abstract", or "interface"
            Returns "[]" if nothing matches. Returns "Error: ..." on path violations.
        """
        try:
            safe_path = _safe_path(path)
        except ValueError as e:
            return str(e)

        # Already a full descriptor — return immediately
        if query.startswith('L') and '->' in query and '(' in query:
            return json.dumps([{'descriptor': query, 'file': '', 'kind': 'method', 'class_modifier': 'unknown'}])

        results = []
        seen: set = set()

        rg_r = subprocess.run(
            ['rg', '-i', '-l', query, '--glob', '*.smali', safe_path],
            capture_output=True, text=True
        )
        candidate_files = [f.strip() for f in rg_r.stdout.splitlines() if f.strip()] if rg_r.returncode in (0, 1) else []

        for fpath in candidate_files:
            try:
                root, _ = _smali_parse(fpath)
            except Exception:
                continue
            class_desc, mods = _smali_class_info(root)
            if not class_desc:
                continue
            if 'interface' in mods:
                modifier = 'interface'
            elif 'abstract' in mods:
                modifier = 'abstract'
            else:
                modifier = 'concrete'
            rel_file = os.path.relpath(fpath, safe_path)

            if query.lower() in class_desc.lower() and class_desc not in seen:
                seen.add(class_desc)
                results.append({'descriptor': class_desc, 'file': rel_file, 'kind': 'class', 'class_modifier': modifier})

            for m in _smali_walk(root, 'method_definition'):
                sig = _smali_method_sig(m)
                if query.lower() in sig.lower():
                    full = f'{class_desc}->{sig}'
                    if full not in seen:
                        seen.add(full)
                        results.append({'descriptor': full, 'file': rel_file, 'kind': 'method', 'class_modifier': modifier})
            if len(results) >= 20:
                break

        return json.dumps(results[:20])

    @mcp.tool()
    def smali_find_implementations(interface_descriptor: str, method_name: str = "", path: str = ".", max_depth: int = 4) -> str:
        """Find all concrete classes implementing an interface or extending an abstract class.

        Recursively resolves through abstract intermediate classes so only
        instantiable (concrete) leaf implementations are returned.
        Essential for resolving invoke-interface call sites and tracking
        Runnable/Callable implementations passed as callbacks.

        Args:
            interface_descriptor: Full smali class descriptor of the interface or abstract class.
                                  e.g. "Ljava/lang/Runnable;" or "Lcom/example/IProcessor;"
            method_name:          Optional method name or signature to filter results.
                                  e.g. "run()V" or just "run". Leave empty to return all.
            path:                 Directory to search. Defaults to "."
            max_depth:            Maximum inheritance depth to recurse (default 4).

        Returns:
            JSON array of concrete implementation objects:
              kind              — "implementation" (implements interface) or "override" (extends class)
              implementor_class — full class descriptor of the concrete implementor
              implementor_file  — relative path to the smali file
              method_descriptor — full descriptor of the matching method (empty if method_name omitted)
              depth             — depth in the hierarchy (0 = direct implementor)
            Returns "[]" if no concrete implementations found. Returns "Error: ..." on path violations.
        """
        try:
            safe_path = _safe_path(path)
        except ValueError as e:
            return str(e)

        results: list = []
        seen_classes: set = set()

        # Determine kind from the target class itself
        base_kind = 'implementation'
        for fpath in _smali_rg_files(interface_descriptor, safe_path):
            try:
                root, _ = _smali_parse(fpath)
            except Exception:
                continue
            cd, mods = _smali_class_info(root)
            if cd == interface_descriptor:
                base_kind = 'implementation' if 'interface' in mods else 'override'
                break

        def _recurse(target: str, depth: int):
            if depth > max_depth or target in seen_classes:
                return
            seen_classes.add(target)
            for fpath in _smali_rg_files(target, safe_path):
                try:
                    root, _ = _smali_parse(fpath)
                except Exception:
                    continue
                class_desc, mods = _smali_class_info(root)
                if not class_desc or class_desc == target:
                    continue
                # Check if this class implements/extends target
                has_target = any(
                    next((c for c in d.children if c.type == 'class_identifier' and c.text.decode() == target), None)
                    for d in root.children if d.type in ('implements_directive', 'super_directive')
                )
                if not has_target:
                    continue
                rel_file = os.path.relpath(fpath, safe_path)
                is_abstract = 'abstract' in mods or 'interface' in mods
                if is_abstract:
                    _recurse(class_desc, depth + 1)
                else:
                    if method_name:
                        for m in _smali_walk(root, 'method_definition'):
                            sig = _smali_method_sig(m)
                            if method_name.lower() in sig.lower():
                                results.append({
                                    'kind': base_kind,
                                    'implementor_class': class_desc,
                                    'implementor_file': rel_file,
                                    'method_descriptor': f'{class_desc}->{sig}',
                                    'depth': depth,
                                })
                    else:
                        results.append({
                            'kind': base_kind,
                            'implementor_class': class_desc,
                            'implementor_file': rel_file,
                            'method_descriptor': '',
                            'depth': depth,
                        })

        _recurse(interface_descriptor, 0)
        return json.dumps(results)

    mcp.run(transport="stdio")
