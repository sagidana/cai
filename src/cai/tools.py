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

TOOL_PROFILES = {
    "low": [
        "cat",
        "list_directory",
        "file_info",
        "search_files",
        "get_file_tree",
    ],
    "medium": [
        "cat",
        "list_directory",
        "file_info",
        "search_files",
        "get_file_tree",
        "git_log",
        "git_diff",
        "git_blame",
        "symbol_search",
        "fetch_url",
        "code_outline",
    ],
    "high": [
        "cat",
        "list_directory",
        "file_info",
        "search_files",
        "get_file_tree",
        "git_log",
        "git_diff",
        "git_blame",
        "symbol_search",
        "fetch_url",
        "code_outline",
        "fetch_codebase_metadata",
        "run_command",
        "create_file",
        "edit_file",
    ],
}


if __name__ == '__main__':
    import re
    import stat
    import urllib.request
    from datetime import datetime

    mcp = FastMCP(name="Tools Server")

    # ── Low tier ───────────────────────────────────────────────────────────────

    @mcp.tool()
    def cat(file_path: str) -> str:
        """Read and return the full contents of a file."""
        with open(file_path, 'r') as f:
            return f.read()

    @mcp.tool()
    def list_directory(path: str = ".") -> str:
        """List files and subdirectories at the given path."""
        entries = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            kind = "dir" if os.path.isdir(full) else "file"
            entries.append(f"{kind}  {name}")
        return "\n".join(entries) if entries else "(empty)"

    @mcp.tool()
    def file_info(file_path: str) -> str:
        """Return metadata for a file: size, line count, and last-modified time."""
        s = os.stat(file_path)
        size = s.st_size
        modified = datetime.fromtimestamp(s.st_mtime).isoformat()
        try:
            with open(file_path, 'r') as f:
                lines = sum(1 for _ in f)
        except Exception:
            lines = "n/a (binary?)"
        return json.dumps({"path": file_path, "size_bytes": size, "lines": lines, "modified": modified})

    @mcp.tool()
    def search_files(pattern: str, path: str = ".", file_glob: str = "*") -> str:
        """Search files matching file_glob under path for lines matching a regex pattern.
        Returns at most 200 matches as 'filepath:lineno: line'."""
        import fnmatch
        results = []
        rx = re.compile(pattern)
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for filename in files:
                if not fnmatch.fnmatch(filename, file_glob):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', errors='replace') as f:
                        for lineno, line in enumerate(f, 1):
                            if rx.search(line):
                                results.append(f"{filepath}:{lineno}: {line.rstrip()}")
                                if len(results) >= 200:
                                    results.append("... (truncated at 200 matches)")
                                    return "\n".join(results)
                except (IOError, OSError):
                    continue
        return "\n".join(results) if results else "No matches found."

    @mcp.tool()
    def get_file_tree(path: str = ".", max_depth: int = 3) -> str:
        """Return the directory tree rooted at path up to max_depth levels deep."""
        lines = []
        def _walk(current, prefix, depth):
            if depth > max_depth:
                return
            try:
                entries = sorted(os.listdir(current))
            except PermissionError:
                return
            entries = [e for e in entries if not e.startswith('.')]
            for i, name in enumerate(entries):
                connector = "└── " if i == len(entries) - 1 else "├── "
                full = os.path.join(current, name)
                lines.append(prefix + connector + name)
                if os.path.isdir(full):
                    extension = "    " if i == len(entries) - 1 else "│   "
                    _walk(full, prefix + extension, depth + 1)
        lines.append(path)
        _walk(path, "", 1)
        return "\n".join(lines)

    # ── Medium tier ────────────────────────────────────────────────────────────

    @mcp.tool()
    def git_log(file_path: str = "", n: int = 10) -> str:
        """Show the last n git commits. Pass file_path to scope to a specific file."""
        cmd = ["git", "log", f"-{n}", "--oneline", "--no-decorate"]
        if file_path:
            cmd += ["--", file_path]
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
        result = subprocess.run(
            ["git", "blame", "--date=short", file_path],
            capture_output=True, text=True
        )
        return result.stdout or result.stderr

    @mcp.tool()
    def symbol_search(symbol: str, path: str = ".") -> str:
        """Find definitions of a function or class named symbol across source files."""
        patterns = [
            rf"^\s*(def|class|function|func|fn)\s+{re.escape(symbol)}\b",
            rf"^\s*(public|private|protected|static).*\s+{re.escape(symbol)}\s*\(",
        ]
        rx = re.compile("|".join(patterns))
        results = []
        for root, dirs, files in os.walk(path):
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
        req = urllib.request.Request(url, headers={"User-Agent": "cai/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")

    @mcp.tool()
    def code_outline(file_path: str) -> str:
        """Return the class/method/function structure of a single source file as JSON."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in EXT_TO_LANG:
            return json.dumps({"error": f"Unsupported file type: {ext}"})
        try:
            with open(file_path, 'rb') as f:
                source = f.read()
        except (IOError, OSError) as e:
            return json.dumps({"error": str(e)})
        result = _parse_file(source, LANGUAGE_CONFIGS[EXT_TO_LANG[ext]])
        return json.dumps(result, indent=2)

    # ── High tier ──────────────────────────────────────────────────────────────

    @mcp.tool()
    def fetch_codebase_metadata() -> str:
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
    def run_command(command: str, timeout: int = 30) -> str:
        """Execute a shell command and return its stdout and stderr."""
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        out = result.stdout
        err = result.stderr
        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        return "\n".join(parts) if parts else "(no output)"

    @mcp.tool()
    def create_file(file_path: str, content: str) -> str:
        """Create a new file at file_path with the given content."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
        return f"Created {file_path} ({len(content)} bytes)"

    @mcp.tool()
    def edit_file(file_path: str, old_text: str, new_text: str) -> str:
        """Replace the first occurrence of old_text with new_text in a file."""
        with open(file_path, 'r') as f:
            original = f.read()
        if old_text not in original:
            return f"Error: old_text not found in {file_path}"
        updated = original.replace(old_text, new_text, 1)
        with open(file_path, 'w') as f:
            f.write(updated)
        return f"Replaced 1 occurrence in {file_path}"

    mcp.run(transport="stdio")
