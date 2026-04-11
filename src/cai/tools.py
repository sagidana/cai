from mcp.server.fastmcp import FastMCP
import subprocess
import sys
import json
import os
import logging
import threading
import atexit

logging.basicConfig(
    filename="/tmp/cai.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("cai.tools")

# Guards creation of singleton connections
_registry_lock = threading.Lock()
_internal_conn = None
_external_conns = {}  # server_path -> MCPServerConnection


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
        # Send initialized notification (no response expected)
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


def _get_internal_conn():
    global _internal_conn
    if _internal_conn is None:
        with _registry_lock:
            if _internal_conn is None:
                _internal_conn = MCPServerConnection(
                    [sys.executable, "-m", "cai.tools"]
                )
    return _internal_conn


def _get_external_conn(server_path):
    if server_path not in _external_conns:
        with _registry_lock:
            if server_path not in _external_conns:
                _external_conns[server_path] = MCPServerConnection(
                    ["python", server_path]
                )
    return _external_conns[server_path]


def call_tool(tool_name, arguments):
    log.info("call_tool: %s %s", tool_name, arguments)
    try:
        conn = _get_internal_conn()
        response = conn.call("tools/call", {"name": tool_name, "arguments": arguments})
        return response.get("result", {}).get("content", [{}])[0].get("text")
    except Exception as e:
        log.error("call_tool exception: %s", e)


def get_tools():
    try:
        conn = _get_internal_conn()
        response = conn.call("tools/list", {})
        mcp_tools = response.get("result", {}).get("tools", [])
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                }
            }
            for tool in mcp_tools
        ]
    except Exception as e:
        log.error("get_tools exception: %s", e)
        return []


def call_external_tool(server_path, tool_name, arguments):
    log.info("call_external_tool: %s %s %s", server_path, tool_name, arguments)
    try:
        conn = _get_external_conn(server_path)
        response = conn.call("tools/call", {"name": tool_name, "arguments": arguments})
        return response.get("result", {}).get("content", [{}])[0].get("text")
    except Exception as e:
        log.error("call_external_tool exception: %s", e)


def get_external_tools(server_path):
    try:
        conn = _get_external_conn(server_path)
        response = conn.call("tools/list", {})
        mcp_tools = response.get("result", {}).get("tools", [])
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                }
            }
            for tool in mcp_tools
        ]
    except Exception as e:
        log.error("get_external_tools exception: %s", e)
        return []


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
