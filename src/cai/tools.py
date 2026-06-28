"""tools: the ToolsRegistry - the install-wide catalogue of tools and the
per-agent dispatch table, in one class.

Two kinds of tool feed it:

  - function tools: plain Python callables registered via the cai.tool
    decorator (an extension's tools/*.py, imported by UserConfig.load). A
    process-global store on ToolsRegistry holds them by name, the function-tool
    counterpart to the MCP servers found on disk.
  - MCP tools: each *.py under an extension's mcps/ dir (or the builtins/mcps/
    dir shipped with cai) is an MCP server (a FastMCP stdio program). This
    module spawns each as a subprocess, speaks the MCP JSON-RPC handshake over
    its stdin/stdout, and lists/calls its tools. MCP tool names are namespaced
    {label}__{name}, where label is the file's basename, so two servers can each
    expose a `search` without colliding.

A ToolsRegistry instance exposes the two things the agentic loop needs:

  registry.tools            - OpenAI-format tool schemas for the model
                              (call_llm tools=)
  registry.dispatch(name, args) - run a tool by name, returns its text result
                              (call_llm tools_dispatch=)

Minimal by design: blocking JSON-RPC (no timeouts/retries/threads), one
request/response at a time; a stdio server's stderr is discarded. Local stdio
servers (a command, run as a subprocess) and remote servers (a URL, spoken to
over Streamable HTTP) sit side by side; images and user-only display blocks are
still later layers."""
from __future__ import annotations

import os
import sys
import json
import typing
import atexit
import inspect
import logging
import subprocess

import requests

from cai import config
from cai.userconfig import UserConfig


log = logging.getLogger("cai")

MCP_PROTOCOL_VERSION = "2024-11-05"


class LocalMCPServer:
    """One MCP stdio server subprocess, spoken to over line-delimited JSON-RPC.
    The loop runs tools one at a time, so this stays single-threaded: write a
    request, read stdout lines until the matching id comes back."""

    def __init__(self, command, label, env=None, cwd=None):
        self.label = label
        self._command = command
        self._req_id = 0
        popen_env = None
        if env is not None:
            popen_env = dict(os.environ)
            popen_env.update(env)
        self._process = subprocess.Popen(command,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL,
                                         text=True,
                                         bufsize=1,
                                         env=popen_env,
                                         cwd=cwd)
        atexit.register(self.close)
        self._handshake()

    def _handshake(self):
        params = {}
        params["protocolVersion"] = MCP_PROTOCOL_VERSION
        params["capabilities"] = {}
        params["clientInfo"] = {"name": "cai", "version": "1.0"}
        self._request("initialize", params)
        self._notify("notifications/initialized")
        log.info("MCP server %r started", self.label)

    def _write(self, message):
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def _notify(self, method):
        message = {}
        message["jsonrpc"] = "2.0"
        message["method"] = method
        self._write(message)

    def _request(self, method, params):
        """send one JSON-RPC request and return its response dict, skipping any
        interleaved notification/log line until the matching id comes back."""
        self._req_id += 1
        req_id = self._req_id
        message = {}
        message["jsonrpc"] = "2.0"
        message["id"] = req_id
        message["method"] = method
        message["params"] = params
        self._write(message)

        while True:
            line = self._process.stdout.readline()
            if line == "":
                raise ConnectionError(f"MCP server {self.label!r} exited during {method!r}")
            line = line.strip()
            if not line: continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                log.warning("MCP %r: ignoring non-JSON line: %r", self.label, line[:200])
                continue
            if not isinstance(response, dict): continue
            if response.get("id") != req_id: continue
            return response

    def list_tools(self):
        response = self._request("tools/list", {})
        return _tools_from_response(response)

    def call_tool(self, name, arguments):
        response = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        return _text_from_response(response)

    def close(self):
        if self._process is None: return
        if self._process.poll() is not None: return
        self._process.terminate()


class RemoteMCPServer:
    """One remote MCP server reached over Streamable HTTP - the remote
    counterpart to LocalMCPServer, same list_tools / call_tool / close surface.
    Each JSON-RPC request is one POST to the server URL; the reply is either a
    JSON body or an SSE stream, from which the response matching the request id
    is read (any notifications ahead of it are skipped). A session id handed back
    on initialize (the Mcp-Session-Id header) is echoed on later requests. One
    request/response at a time, matching the loop - no retries or timeouts."""

    def __init__(self, url, label, headers=None):
        self.label = label
        self._url = url
        self._req_id = 0
        self._session_id = None
        self._headers = {}
        if headers is not None:
            self._headers = dict(headers)
        self._handshake()

    def _handshake(self):
        params = {}
        params["protocolVersion"] = MCP_PROTOCOL_VERSION
        params["capabilities"] = {}
        params["clientInfo"] = {"name": "cai", "version": "1.0"}
        self._request("initialize", params)
        self._notify("notifications/initialized")
        log.info("MCP server %r started", self.label)

    def _post(self, message, expect_reply):
        headers = dict(self._headers)
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json, text/event-stream"
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id
        response = requests.post(self._url, json=message, headers=headers, stream=expect_reply)
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id
        return response

    def _notify(self, method):
        message = {}
        message["jsonrpc"] = "2.0"
        message["method"] = method
        self._post(message, False)

    def _request(self, method, params):
        self._req_id += 1
        req_id = self._req_id
        message = {}
        message["jsonrpc"] = "2.0"
        message["id"] = req_id
        message["method"] = method
        message["params"] = params
        response = self._post(message, True)
        return self._read_reply(response, req_id)

    def _read_reply(self, response, req_id):
        """the JSON-RPC response dict matching req_id, from either a JSON body or
        an SSE stream of `data:` lines (notifications before it are skipped)."""
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            return response.json()
        for raw in response.iter_lines():
            if not raw: continue
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"): continue
            payload = line[len("data:"):].strip()
            if not payload: continue
            try:
                message = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict): continue
            if message.get("id") != req_id: continue
            return message
        raise ConnectionError(f"MCP server {self.label!r} closed the stream during request {req_id}")

    def list_tools(self):
        response = self._request("tools/list", {})
        return _tools_from_response(response)

    def call_tool(self, name, arguments):
        response = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        return _text_from_response(response)

    def close(self):
        pass


def _tools_from_response(response):
    """the tool definitions out of a tools/list response (the shared shape both
    a local and a remote server return)."""
    return response.get("result", {}).get("tools", [])


def _text_from_response(response):
    """the joined text blocks out of a tools/call response, or an Error: line
    when the server reported one."""
    if response.get("error") is not None:
        return f"Error: {response['error']}"
    blocks = response.get("result", {}).get("content", [])
    texts = []
    for block in blocks:
        if block.get("type") != "text": continue
        if block.get("text") is None: continue
        texts.append(block["text"])
    return "\n".join(texts)


_PY_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _namespaced_tool_name(name, origin):
    """the exposed name for a function tool defined in `origin`: namespaced
    '<extension>__<name>' (mirroring MCP tools) when origin sits inside an
    extension's dir, else the bare name for user-level and plain-SDK tools. the
    UserConfig import is local to avoid a tools<->userconfig import cycle."""
    from cai.userconfig import UserConfig
    extension = UserConfig.extension_for(origin)
    if extension is None:
        return name
    if extension == "user":
        return name
    return f"{extension}__{name}"


def _function_tool_name(fn):
    """the exposed name of a function tool: the '<extension>__<name>' stamped on
    it by ToolsRegistry.register_global, or the bare __name__ for a callable that
    was never registered globally (a direct tools= callable, user/SDK tool)."""
    return getattr(fn, "_cai_tool_name", fn.__name__)


def schema_from_function(fn):
    """build an OpenAI tool schema from a Python callable. param types come from
    annotations (unannotated default to string; unsupported types raise
    TypeError); the description is the first line of the docstring."""
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    properties = {}
    required = []
    for pname, param in sig.parameters.items():
        ann = hints.get(pname, param.annotation)
        item_type = None
        if ann is inspect.Parameter.empty:
            json_type = "string"
        elif ann in _PY_TO_JSON_TYPE:
            json_type = _PY_TO_JSON_TYPE[ann]
        else:
            # generic aliases like list[str]: read the origin, carry the element
            # type into the array schema when it is a known type.
            origin = getattr(ann, "__origin__", None)
            if origin not in _PY_TO_JSON_TYPE:
                raise TypeError(f"unsupported annotation {ann!r} on parameter "
                                f"{pname!r} of {fn.__name__}")
            json_type = _PY_TO_JSON_TYPE[origin]
            args = getattr(ann, "__args__", None) or ()
            if args and args[0] in _PY_TO_JSON_TYPE:
                item_type = _PY_TO_JSON_TYPE[args[0]]

        prop = {}
        prop["type"] = json_type
        if item_type is not None:
            prop["items"] = {"type": item_type}
        properties[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    description = ""
    doc = (fn.__doc__ or "").strip()
    if doc:
        description = doc.splitlines()[0]

    parameters = {}
    parameters["type"] = "object"
    parameters["properties"] = properties
    parameters["required"] = required

    function = {}
    function["name"] = _function_tool_name(fn)
    function["description"] = description
    function["parameters"] = parameters

    schema = {}
    schema["type"] = "function"
    schema["function"] = function
    return schema


def _mcp_tool_schema(exposed, tool):
    """OpenAI schema for one MCP tool definition, exposed under `exposed`."""
    function = {}
    function["name"] = exposed
    function["description"] = tool.get("description") or ""
    function["parameters"] = tool.get("inputSchema") or {"type": "object", "properties": {}}
    return {"type": "function", "function": function}


class ToolsRegistry:
    """The tools an Agent/Run can reach: in-process Python functions and MCP
    server tools, in one dispatch table.

    Registering a tool and selecting it are separate: register() makes a tool
    known (and dispatchable), select() marks it active. `tools` - what the model
    sees - is the selected subset; `dispatch` runs any registered tool. So a tool
    can be available without being offered to the model (e.g. the sub-agent tools
    an Agent always registers but only sends once a skill or the user selects).

    A bare tool name resolves against the process-global store of function tools
    registered via cai.tool (see register_global / the cai.tool decorator below):
    an Agent can select such a tool by name, since its callable is recoverable
    from the catalogue.

    An MCP tool referenced by name ('<mcp_name>__<tool_name>') is lazy: the
    server file <mcp_name>.py (under an extension's mcps/ dir) is spawned on
    first use - when its schema is read for the model, or when the tool is
    called - not before."""

    # process-global function tools registered via cai.tool, as
    # name -> (fn, origin) (origin is the file the tool was defined in). every
    # ToolsRegistry built afterward can resolve these by name, and
    # available_tools lists them. populated by register_global, cleared by
    # reset_global - the function-tool half of the install-wide catalogue, whose
    # other half is the MCP servers (on disk, plus the declared ones below).
    _registered = {}

    # process-global MCP servers declared via cai.mcp_server (init.py), as
    # name -> spec dict: {"command": [...], "env": {...}, "cwd": ...} for a stdio
    # server or {"url": ..., "headers": {...}} for a remote one. the programmatic
    # twin of dropping a <name>.py into an mcps/ dir, and consulted ahead of it -
    # so a declared server shadows an on-disk one of the same name. populated by
    # register_server, cleared by reset_global.
    _declared_servers = {}

    @classmethod
    def register_global(cls, fn):
        """record a function tool in the process-global store (cai.tool's
        backing), keyed by its exposed name. a tool defined in an extension's
        tools/ dir is namespaced '<extension>__<name>' (mirroring MCP tools);
        user-level and plain-SDK tools keep their bare __name__. the exposed name
        is stamped on fn so the schema, dispatch key and selection all agree on
        it. a later registration of the same name wins (the user's init runs
        last). the origin file is captured from fn so UserConfig.extension_for
        can attribute it later."""
        origin = None
        code = getattr(fn, "__code__", None)
        if code is not None:
            origin = code.co_filename
        fn._cai_tool_name = _namespaced_tool_name(fn.__name__, origin)
        cls._registered[fn._cai_tool_name] = (fn, origin)

    @classmethod
    def registered(cls):
        """the globally-registered function tools as name -> (fn, origin)."""
        return dict(cls._registered)

    @classmethod
    def global_function(cls, name):
        """the callable registered globally under `name` via cai.tool, or None
        when no function tool by that name is known."""
        entry = cls._registered.get(name)
        if entry is None:
            return None
        return entry[0]

    @classmethod
    def register_server(cls, name, command=None, url=None, env=None, headers=None, cwd=None):
        """declare a named MCP server in the process-global store (cai.mcp_server's
        backing), the programmatic counterpart to dropping a <name>.py into an
        mcps/ dir. exactly one of `command` (a stdio server: an argv list, with
        optional env/cwd) or `url` (a remote Streamable-HTTP server, with optional
        headers) must be given. its tools surface namespaced '<name>__<tool>' like
        any MCP server's. a later declaration of the same name wins."""
        if command is None and url is None:
            raise ValueError(f"MCP server {name!r}: give command= or url=")
        if command is not None and url is not None:
            raise ValueError(f"MCP server {name!r}: give command= or url=, not both")
        if url is not None and (env is not None or cwd is not None):
            raise ValueError(f"MCP server {name!r}: env/cwd apply to a command server, not a url")
        if command is not None and headers is not None:
            raise ValueError(f"MCP server {name!r}: headers apply to a url server, not a command")
        spec = {}
        if command is not None:
            if isinstance(command, str):
                raise ValueError(f"MCP server {name!r}: command must be an argv list, not a string")
            spec["command"] = list(command)
            if env is not None:
                spec["env"] = dict(env)
            if cwd is not None:
                spec["cwd"] = cwd
        else:
            spec["url"] = url
            if headers is not None:
                spec["headers"] = dict(headers)
        cls._declared_servers[name] = spec

    @classmethod
    def declared_servers(cls):
        """the programmatically-declared MCP servers as name -> spec dict."""
        return dict(cls._declared_servers)

    @staticmethod
    def _server_from_spec(name, spec):
        """build a live MCP server connection from a declared spec - a
        RemoteMCPServer for a url spec, a LocalMCPServer for a command one."""
        if "url" in spec:
            return RemoteMCPServer(spec["url"], name, headers=spec.get("headers"))
        return LocalMCPServer(spec["command"], name, env=spec.get("env"), cwd=spec.get("cwd"))

    @classmethod
    def reset_global(cls):
        """drop every globally-registered function tool and declared MCP server
        (test isolation)."""
        cls._registered = {}
        cls._declared_servers = {}

    def __init__(self):
        self._functions = {}     # name -> callable
        self._dispatch = {}      # exposed_name -> tagged entry
        self._schemas = {}       # exposed_name -> schema (eager, or resolved lazily)
        self._order = []         # exposed names, registration order
        self._selected = []      # exposed names that are active (sent to the model)
        # mcp_name -> LocalMCPServer/RemoteMCPServer: both the connected-once
        # cache and the set of servers to close.
        self._mcp_servers = {}

    @classmethod
    def for_tools(cls, tools):
        """build a registry from a mixed list, registering and selecting each: a
        callable becomes a Python function tool; a string is an MCP tool
        reference '<mcp>__<tool>'."""
        registry = cls()
        if not tools:
            return registry
        for item in tools:
            registry.select(item)
        return registry

    @classmethod
    def available_tools(cls):
        """every tool the install offers, as a flat list of names in the form
        `tools=` / for_tools accept: the function tools registered via cai.tool
        (their bare names) followed by every MCP tool ('<mcp_name>__<tool_name>')
        exposed by the servers - declared via cai.mcp_server first, then across
        the extension mcps/ dirs and the builtins. each MCP server is spawned (or,
        when remote, contacted) briefly to list its tools and then closed -
        nothing is left running. a declared server shadows an on-disk one of the
        same name, and a user file shadows a builtin; a server that fails to start
        is logged and skipped."""
        names = list(cls._registered.keys())
        seen_labels = set()
        for label in sorted(cls._declared_servers.keys()):
            seen_labels.add(label)
            server = None
            try:
                server = cls._server_from_spec(label, cls._declared_servers[label])
                for tool in server.list_tools():
                    names.append(f"{label}__{tool['name']}")
            except Exception as e:
                log.error("available_tools: declared server %r failed: %s", label, e)
            finally:
                if server is not None:
                    server.close()
        for directory in _mcp_search_dirs():
            if not os.path.isdir(directory): continue
            for filename in sorted(os.listdir(directory)):
                if not filename.endswith(".py"): continue
                if filename.startswith("_"): continue
                label = filename[:-len(".py")]
                if label in seen_labels: continue
                seen_labels.add(label)
                path = os.path.join(directory, filename)
                server = None
                try:
                    server = LocalMCPServer([sys.executable, path], label)
                    for tool in server.list_tools():
                        names.append(f"{label}__{tool['name']}")
                except Exception as e:
                    log.error("available_tools: %r failed: %s", path, e)
                finally:
                    if server is not None:
                        server.close()
        return names

    def register_function(self, fn):
        """expose a Python callable as a tool. its schema is derived from the
        signature + docstring, and a call runs it in-process. the tool name is
        the function's exposed name (namespaced '<extension>__<name>' for an
        extension tool, else its __name__); a duplicate name raises ValueError."""
        name = _function_tool_name(fn)
        self._is_name_free(name)
        self._functions[name] = fn
        self._dispatch[name] = ("function", name)
        self._schemas[name] = schema_from_function(fn)
        self._order.append(name)

    def register_mcp_tool(self, name):
        """register an MCP tool reference '<mcp_name>__<tool_name>'. the server
        file <mcp_name>.py under an extension's mcps/ dir is loaded into this registry now -
        lazily, in the sense that only referenced servers load, once each - so
        the dispatcher has a live connection to use. a load failure or unknown
        tool is logged and the tool is skipped."""
        if "__" not in name:
            raise ValueError(f"MCP tool {name!r} must be '<mcp_name>__<tool_name>'")
        self._is_name_free(name)
        mcp_name, tool_name = name.split("__", 1)
        try:
            server = self._load_server(mcp_name)
            schema = self._find_tool_schema(server, tool_name, name)
        except Exception as e:
            log.error("failed loading MCP tool %r: %s", name, e)
            return
        if schema is None:
            log.error("MCP server %r exposes no tool %r", mcp_name, tool_name)
            return
        self._dispatch[name] = ("mcp", mcp_name, tool_name)
        self._schemas[name] = schema
        self._order.append(name)

    def _find_tool_schema(self, server, tool_name, exposed):
        for tool in server.list_tools():
            if tool["name"] != tool_name: continue
            return _mcp_tool_schema(exposed, tool)
        return None

    def register(self, tool, override=False):
        """make one tool known (dispatchable) without selecting it. `tool` may be
        a callable (a function tool), the name of a function tool registered
        globally via cai.tool, or an MCP tool reference '<mcp>__<tool>'.
        a tool already registered is left as-is, unless override=True, which
        replaces the existing registration with this tool - e.g. to bind an
        inherited tool name to this agent's own tool. like any fresh
        registration, the replacement is left unselected."""
        name = _function_tool_name(tool) if callable(tool) else tool
        if self.has(name):
            if not override: return
            self.remove(name)
        if callable(tool):
            self.register_function(tool)
            return
        fn = ToolsRegistry.global_function(tool)
        if fn is not None:
            self.register_function(fn)
            return
        self.register_mcp_tool(tool)

    def select(self, tool, auto_register=True):
        """mark `tool` active (sent to the model). `tool` may be a callable or an
        MCP-ref name.

        with auto_register=True (default) the tool is registered first if it isn't
        known yet (a malformed name raises, from register; an MCP name whose
        server fails to load registers nothing and so selects nothing). with
        auto_register=False the tool must already be registered - an unknown tool
        is left unselected, so registering and selecting stay fully separate."""
        if auto_register:
            self.register(tool)
        name = _function_tool_name(tool) if callable(tool) else tool
        if not self.has(name): return
        if name in self._selected: return
        self._selected.append(name)

    def deselect(self, name):
        """make a tool inactive (no longer sent to the model). it stays
        registered, so it can be selected again. unknown/unselected names are
        ignored."""
        if name not in self._selected: return
        self._selected.remove(name)

    def remove(self, name):
        """fully unregister a tool by its exposed name (and deselect it); unknown
        names are ignored. an MCP server connection stays cached (closed with the
        registry), since other tools may still use it."""
        self.deselect(name)
        if name not in self._dispatch: return
        del self._dispatch[name]
        self._schemas.pop(name, None)
        self._functions.pop(name, None)
        self._order.remove(name)

    def has(self, name):
        """True if a tool with this exposed name is registered (selected or not)."""
        return name in self._dispatch

    def names(self):
        """exposed names of every registered tool, selected or not, in
        registration order."""
        return list(self._order)

    def selected(self):
        """exposed names of the active tools (the subset sent to the model), in
        selection order."""
        return list(self._selected)

    def get(self, name):
        """the original tool behind `name`, for re-registering it elsewhere (a
        sub-agent inheriting a parent's tool): the callable for a function tool,
        or the '<mcp>__<tool>' name string for an MCP tool. None if unknown."""
        entry = self._dispatch.get(name)
        if entry is None:
            return None
        if entry[0] == "function":
            return self._functions[name]
        return name

    def _is_name_free(self, name):
        if name in self._dispatch:
            raise ValueError(f"tool name collision: {name!r}")

    @property
    def tools(self):
        """OpenAI-format schemas of the selected tools, in selection order, or
        None when none are selected so call_llm omits the tools field entirely.
        only selected tools are offered to the model; registered-but-unselected
        ones stay dispatchable but hidden."""
        if not self._selected:
            return None
        schemas = []
        for exposed in self._selected:
            schemas.append(self._schemas[exposed])
        return schemas

    def dispatch(self, name, arguments):
        """run one tool by its exposed name. matches call_llm's tools_dispatch
        contract: always returns a string, never raises."""
        entry = self._dispatch.get(name)
        if entry is None:
            return f"Error: unknown tool '{name}'"
        if entry[0] == "function":
            return self._call_function(entry[1], arguments)
        _kind, mcp_name, tool_name = entry
        try:
            server = self._load_server(mcp_name)
            return server.call_tool(tool_name, arguments)
        except Exception as e:
            log.exception("tool %s failed", name)
            return f"Error: tool '{name}' failed: {e}"

    def _load_server(self, mcp_name):
        """the live server for mcp_name, connected once and cached in this
        registry. a server declared via cai.mcp_server wins; otherwise its source
        file is resolved from the extension mcps dirs first, then the builtins
        shipped with cai, and spawned as a stdio subprocess."""
        server = self._mcp_servers.get(mcp_name)
        if server is not None:
            return server
        spec = ToolsRegistry._declared_servers.get(mcp_name)
        if spec is not None:
            server = ToolsRegistry._server_from_spec(mcp_name, spec)
        else:
            path = _mcp_server_path(mcp_name)
            if path is None:
                raise FileNotFoundError(
                    f"no MCP server {mcp_name!r} declared, or in any extension mcps dir or builtins")
            server = LocalMCPServer([sys.executable, path], mcp_name)
        self._mcp_servers[mcp_name] = server
        return server

    def _call_function(self, name, arguments):
        fn = self._functions[name]
        try:
            if arguments:
                result = fn(**arguments)
            else:
                result = fn()
        except Exception as e:
            log.exception("tool %s raised", name)
            return f"Error: tool '{name}' raised: {e}"
        if result is None:
            return ""
        return str(result)

    def close(self):
        for server in self._mcp_servers.values():
            try:
                server.close()
            except Exception:
                log.exception("closing MCP server %r failed", server.label)


def builtin_mcp_dir():
    """the MCP servers shipped with cai by default, in builtins/mcps/ beside
    this module - searched after the extension mcps/ dirs."""
    return os.path.join(os.path.dirname(__file__), "builtins", "mcps")


def _mcp_search_dirs():
    """the MCP source dirs in resolution order: each extension's mcps/ dir (an
    earlier one shadows a later one) then the bundled builtins."""
    dirs = list(UserConfig.mcp_dirs())
    dirs.append(builtin_mcp_dir())
    return dirs


def _mcp_server_path(mcp_name):
    """resolve <mcp_name>.py to a source file, searching the extension mcps
    dirs first then the bundled builtins. None when neither has it."""
    filename = mcp_name + ".py"
    for directory in _mcp_search_dirs():
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return None


def tool(fn):
    """decorator: register a function as a global function tool, e.g.

        @cai.tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

    the tool's name is the function's __name__; its schema comes from the
    signature and the first docstring line (see schema_from_function). it is
    baked into the process-global store, so once UserConfig.load() imports the
    extensions every agent can select it by name. see ToolsRegistry."""
    ToolsRegistry.register_global(fn)
    return fn


def mcp_server(name, command=None, url=None, env=None, headers=None, cwd=None):
    """declare an MCP server from init.py, the programmatic counterpart to
    dropping a <name>.py into ~/.config/cai/mcps/. give exactly one of:

        cai.mcp_server("github",
                       command=["npx", "-y", "@modelcontextprotocol/server-github"],
                       env={"GITHUB_TOKEN": "..."})        # local stdio server

        cai.mcp_server("linear",
                       url="https://mcp.linear.app/mcp",
                       headers={"Authorization": "Bearer ..."})   # remote server

    its tools surface namespaced '<name>__<tool>'; Harness callers still list
    them in tools=[...] to expose them for a run. see ToolsRegistry."""
    ToolsRegistry.register_server(name,
                                  command=command,
                                  url=url,
                                  env=env,
                                  headers=headers,
                                  cwd=cwd)
