from mcp.server.fastmcp import FastMCP
import subprocess
import sys
import json
import os
import re
import logging
import threading
import atexit
import shlex
import inspect

logging.basicConfig(
    filename="/tmp/cai.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("cai.tools")

# The internal MCP server command — treated exactly like any external server.
INTERNAL_SERVER = f"{sys.executable} -m cai.tools"

# ── Registry ──────────────────────────────────────────────────────────────────
_registry_lock = threading.Lock()
_servers: dict = {}        # cmd_str -> MCPServerConnection
_server_labels: dict = {}  # cmd_str -> label string
_dispatch: dict = {}       # prefixed_name -> (cmd_str, original_name)
_local_functions: dict = {}  # name -> (callable, openai_schema_dict, label)


# ── Connection ────────────────────────────────────────────────────────────────

class MCPServerConnection:
    def __init__(self, cmd):
        self._cmd = cmd
        self._lock = threading.Lock()
        self._req_id = 0
        self._process = None
        self._start()
        atexit.register(self.close)

    def _start(self):
        self._process = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self._req_id = 0
        self._send_rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "cai-client", "version": "1.0"}
        })
        self._process.stdin.write(json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized"
        }) + "\n")
        self._process.stdin.flush()
        log.info("MCPServerConnection started: %s", self._cmd)

    def _ensure_alive(self):
        if self._process.poll() is not None:
            log.warning("MCPServerConnection process died, restarting: %s", self._cmd)
            self._start()

    def _send_rpc(self, method, params):
        self._req_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params
        }
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()
        return json.loads(self._process.stdout.readline())

    def call(self, method, params):
        with self._lock:
            self._ensure_alive()
            return self._send_rpc(method, params)

    def close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            log.info("MCPServerConnection terminated: %s", self._cmd)


# ── Label derivation ──────────────────────────────────────────────────────────

def _derive_label(cmd_str: str) -> str:
    """Derive a short human-readable label from an MCP server command string."""
    if cmd_str == INTERNAL_SERVER:
        return "cai"
    parts = shlex.split(cmd_str)
    if not parts:
        return "mcp"
    exe = os.path.basename(parts[0])
    # If the executable is an interpreter, use the next argument as the target
    if exe in ('python', 'python3', 'node', 'npx', 'uvx', 'deno', 'bun') and len(parts) > 1:
        target = parts[1]
    else:
        target = parts[0]
    # For npm scoped packages like @scope/package-name, take after the last /
    if '/' in target and os.sep not in target:
        target = target.rsplit('/', 1)[1]
    # Strip file extension and take basename
    target = os.path.splitext(os.path.basename(target))[0]
    # Sanitize: replace anything not alphanumeric or underscore with underscore
    target = re.sub(r'[^a-zA-Z0-9_]', '_', target).strip('_')
    return target or "mcp"


# ── Public API ────────────────────────────────────────────────────────────────

def register_server(cmd_str: str, label: str = None) -> None:
    """Register and start an MCP server. Idempotent — safe to call multiple times."""
    if cmd_str not in _servers:
        with _registry_lock:
            if cmd_str not in _servers:
                _servers[cmd_str] = MCPServerConnection(shlex.split(cmd_str))
                _server_labels[cmd_str] = label or _derive_label(cmd_str)
                log.info("register_server: %r -> label=%r", cmd_str, _server_labels[cmd_str])


# ── Local function registration (in-process, no MCP subprocess) ──────────────
# Used by the SDK to expose plain Python callables as tools. Registered
# alongside MCP tools in the same dispatch + schema flow.

_PY_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _derive_function_schema(fn):
    """Build an OpenAI tool schema dict from a Python callable.

    Param types come from annotations; description is the first line of
    the docstring (empty if absent). Unannotated params default to string.
    Unsupported annotation types raise TypeError.
    """
    sig = inspect.signature(fn)
    properties = {}
    required = []
    for pname, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            json_type = "string"
        elif ann in _PY_TO_JSON_TYPE:
            json_type = _PY_TO_JSON_TYPE[ann]
        else:
            # Handle generic aliases like list[int] by checking origin.
            origin = getattr(ann, "__origin__", None)
            if origin in _PY_TO_JSON_TYPE:
                json_type = _PY_TO_JSON_TYPE[origin]
            else:
                raise TypeError(
                    f"register_local_functions: unsupported annotation "
                    f"{ann!r} on parameter {pname!r} of {fn.__name__}"
                )
        properties[pname] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    doc = (fn.__doc__ or "").strip()
    description = doc.splitlines()[0] if doc else ""

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def register_local_functions(functions, label: str = "local") -> None:
    """Register Python callables as tools, dispatched in-process.

    Name collisions against any registered tool (MCP or local) raise
    ValueError. Idempotent per-function: re-registering the same callable
    object is a no-op.

    :param label: surfaced by :func:`get_tool_entries` (and thus the /tools UI)
                  to tell different registration sources apart — e.g. "init"
                  for tools registered via ~/.config/cai/init.py vs "local"
                  for tools passed directly to ``Harness(functions=...)``.
    """
    with _registry_lock:
        for fn in functions:
            name = fn.__name__
            # Idempotent if the exact same callable object was already registered.
            existing = _local_functions.get(name)
            if existing and existing[0] is fn:
                continue
            if existing or name in _dispatch:
                raise ValueError(f"tool name collision: {name!r}")
            schema = _derive_function_schema(fn)
            _local_functions[name] = (fn, schema, label)
            log.info("register_local_functions: %r registered (label=%s)", name, label)


def get_all_tools() -> list:
    """Return the unified OpenAI-format tool list from all registered servers.

    Built-in (internal) tool names are exposed as-is.  External MCP tool names
    are prefixed as ``{label}__{original_name}`` to avoid cross-server
    collisions.  Also rebuilds the internal dispatch table used by call_tool().
    """
    global _dispatch
    new_dispatch = {}
    openai_tools = []

    for cmd_str, conn in _servers.items():
        label = _server_labels[cmd_str]
        try:
            response = conn.call("tools/list", {})
            mcp_tools = response.get("result", {}).get("tools", [])
        except Exception as e:
            log.error("get_all_tools: failed listing tools for %r: %s", cmd_str, e)
            mcp_tools = []

        is_internal = (cmd_str == INTERNAL_SERVER)
        for tool in mcp_tools:
            original_name = tool["name"]
            exposed_name = original_name if is_internal else f"{label}__{original_name}"
            new_dispatch[exposed_name] = (cmd_str, original_name)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": exposed_name,
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                }
            })

    # Also surface any in-process Python functions registered via the SDK.
    for name, (_fn, schema, _label) in _local_functions.items():
        openai_tools.append(schema)
        # Dispatch entry marker: cmd_str=None signals "local function".
        new_dispatch[name] = (None, name)

    _dispatch = new_dispatch
    log.info("get_all_tools: %d tools from %d servers (+%d local)",
             len(openai_tools), len(_servers), len(_local_functions))
    return openai_tools


def call_tool(prefixed_name: str, arguments: dict):
    """Call a tool by its prefixed name, dispatching to the correct MCP server
    or — if registered locally — invoking a Python callable in-process."""
    log.info("call_tool: %s %s", prefixed_name, arguments)
    entry = _dispatch.get(prefixed_name)
    if entry is None:
        log.error("call_tool: unknown tool %r", prefixed_name)
        return f"Error: unknown tool '{prefixed_name}'"
    cmd_str, original_name = entry
    # Local function path
    if cmd_str is None:
        fn, _schema, _label = _local_functions[original_name]
        try:
            result = fn(**arguments) if arguments else fn()
        except Exception as e:
            log.error("call_tool local %r raised: %s", prefixed_name, e)
            return f"Error: tool {prefixed_name!r} raised: {e}"
        return "" if result is None else str(result)
    # MCP subprocess path
    conn = _servers[cmd_str]
    try:
        response = conn.call("tools/call", {"name": original_name, "arguments": arguments})
        return response.get("result", {}).get("content", [{}])[0].get("text")
    except Exception as e:
        log.error("call_tool exception for %r: %s", prefixed_name, e)


def get_tool_entries() -> list:
    """Return [(prefixed_name, label), ...] for all registered tools — used by the /tools UI."""
    entries = []
    for prefixed_name, (cmd_str, original) in _dispatch.items():
        if cmd_str is None:
            label = _local_functions[original][2]
        else:
            label = _server_labels[cmd_str]
        entries.append((prefixed_name, label))
    return entries


def select_tools(available_tools: list, selected_names) -> list:
    """Filter the unified tool list down to the names the user has selected."""
    names = selected_names if selected_names is not None else set()
    return [
        tool for tool in available_tools
        if tool.get('function', {}).get('name') in names
    ]


if __name__ == '__main__':
    from cai import adb_tools, files_tools, frida_tools, git_tools, smali_tools, web_tools

    mcp = FastMCP(name="Tools Server")

    adb_tools.register(mcp)
    files_tools.register(mcp)
    frida_tools.register(mcp)
    git_tools.register(mcp)
    smali_tools.register(mcp)
    web_tools.register(mcp)

    mcp.run(transport="stdio")
