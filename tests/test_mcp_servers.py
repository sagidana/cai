"""Tests for MCP servers declared via cai.mcp_server: a local stdio server (a
command, spawned as a subprocess) and a remote server (a URL, spoken to over
Streamable HTTP). The local case really spawns the bundled fs server; the remote
case runs against a stub HTTP server in a background thread - no network either
way. The conftest fixture gives every test a fresh default Environment, so the
declared servers never leak between tests."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cai
from cai.environment import Environment, builtin_mcp_dir
from cai.tools import ToolsRegistry, RemoteMCPServer


def _tool_names(server):
    names = []
    for tool in server.list_tools():
        names.append(tool["name"])
    return names


# ─── local: a declared command server, really spawned ────────────────────────

def test_declared_local_command_server_lists_and_dispatches(tmp_path):
    import os
    import sys

    fs = os.path.join(builtin_mcp_dir(), "fs.py")
    (tmp_path / "hello.txt").write_text("hi")
    cai.mcp_server("myfs", command=[sys.executable, fs], cwd=str(tmp_path))

    # discovery surfaces its tools namespaced under the declared name
    available = Environment.default().available_tools()
    assert "myfs__list_files" in available

    # selecting + dispatching routes through the declared server (run in cwd)
    registry = ToolsRegistry.for_tools(["myfs__list_files"])
    try:
        assert registry.selected() == ["myfs__list_files"]
        out = registry.dispatch("myfs__list_files", {"directory": "."})
        assert "hello.txt" in out
    finally:
        registry.close()


def test_declared_server_shadows_on_disk_of_same_name():
    # a declared server named like a builtin (fs) wins over the bundled one.
    import sys

    fs = builtin_mcp_dir() + "/fs.py"
    cai.mcp_server("fs", command=[sys.executable, fs])

    registry = ToolsRegistry()
    server = registry._load_server("fs")
    try:
        spec = Environment.default().server_spec("fs")
        assert spec["command"] == [sys.executable, fs]
        assert "list_files" in _tool_names(server)
    finally:
        registry.close()


# ─── remote: a declared url server against a stub HTTP server ─────────────────

def _make_stub(sse):
    """an HTTPServer answering the MCP handshake, tools/list and tools/call.
    when sse is True it replies to non-handshake requests as an event-stream
    (with a comment line and an interleaved notification before the response) to
    exercise RemoteMCPServer's SSE parsing; otherwise as a plain JSON body."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length))
            if body.get("id") is None:                    # a notification
                self.send_response(202)
                self.end_headers()
                return
            method = body["method"]
            if method == "initialize":
                result = {"protocolVersion": "2024-11-05", "capabilities": {}}
            elif method == "tools/list":
                result = {"tools": [{"name": "ping", "description": "p",
                                     "inputSchema": {"type": "object", "properties": {}}}]}
            else:
                args = body["params"]["arguments"]
                result = {"content": [{"type": "text", "text": "pong:" + json.dumps(args)}]}
            message = {"jsonrpc": "2.0", "id": body["id"], "result": result}
            if sse and method != "initialize":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Mcp-Session-Id", "sess-1")
                self.end_headers()
                self.wfile.write(b": keep-alive comment\n\n")
                self.wfile.write(b'data: {"jsonrpc":"2.0","method":"notes/log"}\n\n')
                self.wfile.write(("data: " + json.dumps(message) + "\n\n").encode())
            else:
                payload = json.dumps(message).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Mcp-Session-Id", "sess-1")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

    return HTTPServer(("127.0.0.1", 0), Handler)


def _serve(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}/mcp"


def test_remote_server_json_reply():
    stub = _make_stub(sse=False)
    url = _serve(stub)
    try:
        server = RemoteMCPServer(url, "stub", headers={"Authorization": "Bearer t"})
        assert _tool_names(server) == ["ping"]
        assert server.call_tool("ping", {"a": 1}) == 'pong:{"a": 1}'
        assert server._session_id == "sess-1"      # captured from the header
    finally:
        stub.shutdown()


def test_remote_server_sse_reply_skips_comments_and_notifications():
    stub = _make_stub(sse=True)
    url = _serve(stub)
    try:
        server = RemoteMCPServer(url, "stub")
        assert _tool_names(server) == ["ping"]
        assert server.call_tool("ping", {"a": 1}) == 'pong:{"a": 1}'
    finally:
        stub.shutdown()


def test_declared_remote_server_loads_through_registry():
    stub = _make_stub(sse=False)
    url = _serve(stub)
    try:
        cai.mcp_server("stub", url=url)
        registry = ToolsRegistry.for_tools(["stub__ping"])
        try:
            assert registry.selected() == ["stub__ping"]
            assert registry.dispatch("stub__ping", {"a": 2}) == 'pong:{"a": 2}'
        finally:
            registry.close()
    finally:
        stub.shutdown()
