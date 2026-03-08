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
    mcp = FastMCP(name="Tools Server")

    @mcp.tool()
    def cat(file_path: str) -> str:
        """Read and return the contents of a file."""
        with open(file_path, 'r') as f:
            return f.read()

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

                file_info = _parse_file(source, LANGUAGE_CONFIGS[EXT_TO_LANG[ext]])
                if file_info:
                    infra[os.path.relpath(filepath, cwd)] = file_info

        return json.dumps(infra, indent=2)

    mcp.run(transport="stdio")
