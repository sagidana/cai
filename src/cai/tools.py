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


def get_all_tools() -> list:
    """Return the unified OpenAI-format tool list from all registered servers.

    Tool names are prefixed as ``{label}__{original_name}`` to avoid
    cross-server collisions.  Also rebuilds the internal dispatch table used
    by call_tool().
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

        for tool in mcp_tools:
            original_name = tool["name"]
            prefixed_name = f"{label}__{original_name}"
            new_dispatch[prefixed_name] = (cmd_str, original_name)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": prefixed_name,
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                }
            })

    _dispatch = new_dispatch
    log.info("get_all_tools: %d tools from %d servers", len(openai_tools), len(_servers))
    return openai_tools


def call_tool(prefixed_name: str, arguments: dict):
    """Call a tool by its prefixed name, dispatching to the correct MCP server."""
    log.info("call_tool: %s %s", prefixed_name, arguments)
    entry = _dispatch.get(prefixed_name)
    if entry is None:
        log.error("call_tool: unknown tool %r", prefixed_name)
        return f"Error: unknown tool '{prefixed_name}'"
    cmd_str, original_name = entry
    conn = _servers[cmd_str]
    try:
        response = conn.call("tools/call", {"name": original_name, "arguments": arguments})
        return response.get("result", {}).get("content", [{}])[0].get("text")
    except Exception as e:
        log.error("call_tool exception for %r: %s", prefixed_name, e)


def get_tool_entries() -> list:
    """Return [(prefixed_name, label), ...] for all registered tools — used by the /tools UI."""
    return [
        (prefixed_name, _server_labels[_dispatch[prefixed_name][0]])
        for prefixed_name in _dispatch
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
