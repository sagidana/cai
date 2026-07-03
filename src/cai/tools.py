"""tools: the ToolsRegistry - one Agent's dispatch table over the two kinds of
tool its Environment offers.

  - function tools: plain Python callables registered via the cai.tool
    decorator (an extension's tools/*.py, imported by Environment.load). The
    env holds them by exposed name; a registry resolves a bare name against it.
    cai.wrap registers a function tool that builds on another tool: its first
    parameter receives the target's dispatch at call time (see wrap).
  - MCP tools: each *.py under an extension's mcps/ dir (or the builtins/mcps/
    dir shipped with cai) is an MCP server (a FastMCP stdio program), and
    cai.mcp_server declares one programmatically into the env. This module
    spawns each as a subprocess, speaks the MCP JSON-RPC handshake over its
    stdin/stdout, and lists/calls its tools. MCP tool names are namespaced
    {label}__{name}, where label is the file's basename, so two servers can each
    expose a `search` without colliding.

A ToolsRegistry instance exposes the two things the agentic loop needs:

  registry.tools            - OpenAI-format tool schemas for the model
                              (call_llm tools=)
  registry.dispatch(name, args) - run a tool by name, returns its text result
                              (call_llm tools_dispatch=)

The install-wide catalogue (which tools exist at all) is the Environment's:
env.available_tools() lists it, and each registry keeps an env reference to
resolve names and server specs against.

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

from cai import paths
from cai.environment import Environment


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


def _function_tool_name(fn):
    """the exposed name of a function tool: the '<extension>__<name>' stamped on
    it by Environment.register_tool, or the bare __name__ for a callable that was
    never registered on an env (a direct tools= callable, user/SDK tool)."""
    return getattr(fn, "_cai_tool_name", fn.__name__)


def server_from_spec(name, spec, extra_env=None):
    """build a live MCP server connection from a declared spec - a
    RemoteMCPServer for a url spec, a LocalMCPServer for a command one.
    extra_env (the registry's CAI_SCRATCH injection) is merged under a command
    spec's own env - a declared variable wins; a url server has no process to
    inject into, so it is ignored there."""
    if "url" in spec:
        return RemoteMCPServer(spec["url"], name, headers=spec.get("headers"))
    env = spec.get("env")
    if extra_env is not None:
        merged = dict(extra_env)
        merged.update(env or {})
        env = merged
    return LocalMCPServer(spec["command"], name, env=env, cwd=spec.get("cwd"))


def schema_from_function(fn):
    """build an OpenAI tool schema from a Python callable. param types come from
    annotations (unannotated default to string; unsupported types raise
    TypeError); the description is the first line of the docstring. a wrap
    tool's first parameter receives the injected target call, so it is left out
    of the schema the model sees."""
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    params = list(sig.parameters.items())
    if getattr(fn, "_cai_wrap_target", None) is not None:
        params = params[1:]

    properties = {}
    required = []
    for pname, param in params:
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

    A bare tool name resolves against the registry's Environment - the function
    tools registered there via cai.tool: an Agent can select such a tool by
    name, since its callable is recoverable from the env's catalogue.

    An MCP tool referenced by name ('<mcp_name>__<tool_name>') is lazy: the
    server - declared on the env via cai.mcp_server, or the file <mcp_name>.py
    under one of the env's mcp dirs - is spawned on first use, when its schema
    is read for the model or when the tool is called, not before."""

    def __init__(self, env=None, scratch=None):
        self.env = env or Environment.default()
        # scratch: a zero-arg callable returning the session scratch directory
        # (Agent wires its own; see Agent.scratch). every local MCP server this
        # registry spawns gets it as CAI_SCRATCH, so tools share one place to
        # exchange binary/bulky intermediates as files. None: no injection.
        self.scratch = scratch
        self._functions = {}     # name -> callable
        self._dispatch = {}      # exposed_name -> tagged entry
        self._schemas = {}       # exposed_name -> schema (eager, or resolved lazily)
        self._order = []         # exposed names, registration order
        self._selected = []      # exposed names that are active (sent to the model)
        # mcp_name -> LocalMCPServer/RemoteMCPServer: both the connected-once
        # cache and the set of servers to close.
        self._mcp_servers = {}

    @classmethod
    def for_tools(cls, tools, env=None):
        """build a registry from a mixed list, registering and selecting each: a
        callable becomes a Python function tool; a string is an MCP tool
        reference '<mcp>__<tool>'."""
        registry = cls(env)
        if not tools:
            return registry
        for item in tools:
            registry.select(item)
        return registry

    def register_function(self, fn):
        """expose a Python callable as a tool. its schema is derived from the
        signature + docstring, and a call runs it in-process. the tool name is
        the function's exposed name (namespaced '<extension>__<name>' for an
        extension tool, else its __name__); a duplicate name raises ValueError.

        a wrap tool (cai.wrap) also registers its target - dispatchable but
        unselected, so the model sees the wrapper and not the tool it builds
        on. a target that cannot be registered (its bundle not installed, its
        server failing) unregisters the wrapper again, logged and skipped like
        a failed MCP tool."""
        name = _function_tool_name(fn)
        self._is_name_free(name)
        target = getattr(fn, "_cai_wrap_target", None)
        if target == name:
            log.error("wrap tool %r wraps itself; skipped", name)
            return
        self._functions[name] = fn
        self._dispatch[name] = ("function", name)
        self._schemas[name] = schema_from_function(fn)
        self._order.append(name)
        if target is None: return
        try:
            self.register(target)
        except Exception:
            log.exception("wrap tool %r: registering target %r failed", name, target)
        if self.has(target): return
        log.error("wrap tool %r: target %r not available; skipped", name, target)
        self.remove(name)

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
        a callable (a function tool), the name of a function tool registered on
        the env via cai.tool, or an MCP tool reference '<mcp>__<tool>'.
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
        fn = self.env.function_tool(tool)
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
        registry. a server declared on the env via cai.mcp_server wins; otherwise
        its source file is resolved from the env's mcp dirs (the extension dirs
        first, then the builtins) and spawned as a stdio subprocess."""
        server = self._mcp_servers.get(mcp_name)
        if server is not None:
            return server
        extra_env = self._scratch_env()
        spec = self.env.server_spec(mcp_name)
        if spec is not None:
            server = server_from_spec(mcp_name, spec, extra_env=extra_env)
        else:
            path = _mcp_server_path(mcp_name, self.env.mcp_dirs())
            if path is None:
                raise FileNotFoundError(
                    f"no MCP server {mcp_name!r} declared, or in any extension mcps dir or builtins")
            server = LocalMCPServer([sys.executable, path], mcp_name, env=extra_env)
        self._mcp_servers[mcp_name] = server
        return server

    def _scratch_env(self):
        """the env injected into every local MCP spawn: the scratch directory
        as CAI_SCRATCH when a provider is wired, else None (no injection)."""
        if self.scratch is None:
            return None
        path = self.scratch()
        if not path:
            return None
        return {"CAI_SCRATCH": path}

    def _call_function(self, name, arguments):
        fn = self._functions[name]
        target = getattr(fn, "_cai_wrap_target", None)
        args = [] # a wrap tool's injected first argument: its target's dispatch
        if target is not None:
            def call(**kwargs):
                return self.dispatch(target, kwargs)
            args.append(call)
        # bracket the call with this registry's scratch provider, so in-process
        # tool code reaches it through cai.scratch_dir() the same way a spawned
        # server reaches CAI_SCRATCH.
        token = None
        if self.scratch is not None:
            token = paths._scratch_provider.set(self.scratch)
        try:
            if arguments:
                result = fn(*args, **arguments)
            else:
                result = fn(*args)
        except Exception as e:
            log.exception("tool %s raised", name)
            return f"Error: tool '{name}' raised: {e}"
        finally:
            if token is not None:
                paths._scratch_provider.reset(token)
        if result is None:
            return ""
        return str(result)

    def close(self):
        for server in self._mcp_servers.values():
            try:
                server.close()
            except Exception:
                log.exception("closing MCP server %r failed", server.label)


def _mcp_server_path(mcp_name, dirs):
    """resolve <mcp_name>.py to a source file, searching `dirs` in order (the
    env's mcp dirs: extensions first, then the bundled builtins). None when no
    dir has it."""
    filename = mcp_name + ".py"
    for directory in dirs:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return None


def tool(fn):
    """decorator: register a function tool on the current Environment, e.g.

        @cai.tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

    the tool's name is the function's __name__ (namespaced
    '<extension>__<name>' when an extension is being loaded); its schema comes
    from the signature and the first docstring line (see schema_from_function).
    it lands on the env being load()ed - else the process default - so once
    Environment.load() imports the extensions every agent on that env can select
    it by name. see Environment / ToolsRegistry."""
    Environment.target().register_tool(fn)
    return fn


def wrap(target):
    """decorator: register a function tool that wraps another tool, e.g.

        @cai.wrap("knowit__read_note")
        def read_note(call, id: str):
            \"\"\"Read a note from cai's memory vault.\"\"\"
            return call(id=id, cwd="~/.config/cai/notes")

    `target` names any registered tool - a function tool, an MCP tool
    ('<mcp>__<tool>'), or another wrapper. the wrapper is a normal function
    tool in every respect (namespaced, schema from its own signature +
    docstring), except its first parameter receives the target at dispatch
    time - a callable taking the target's kwargs and returning its text result
    (an 'Error: ...' string on failure, never an exception) - and is hidden
    from the model's schema. selecting the wrapper registers the target too,
    dispatchable but unselected. see ToolsRegistry.register_function.

    before_tool_call hooks fire on the tool the model calls - the wrapper -
    not on the inner call, so wrapping a gated tool creates an ungated path:
    ship a gate for the wrapper if its target had one."""
    if not isinstance(target, str):
        raise TypeError(f"cai.wrap: target must be a tool name string, got {target!r}")

    def decorator(fn):
        sig = inspect.signature(fn)
        if not sig.parameters:
            raise TypeError(f"wrap tool {fn.__name__!r} needs a first parameter "
                            f"to receive the target call")
        fn._cai_wrap_target = target
        Environment.target().register_tool(fn)
        return fn
    return decorator


def mcp_server(name, command=None, url=None, env=None, headers=None, cwd=None):
    """declare an MCP server from init.py, the programmatic counterpart to
    dropping a <name>.py into ~/.config/cai/mcps/. give exactly one of:

        cai.mcp_server("github",
                       command=["npx", "-y", "@modelcontextprotocol/server-github"],
                       env={"GITHUB_TOKEN": "..."})        # local stdio server

        cai.mcp_server("linear",
                       url="https://mcp.linear.app/mcp",
                       headers={"Authorization": "Bearer ..."})   # remote server

    it lands on the env being load()ed - else the process default. its tools
    surface namespaced '<name>__<tool>'; callers still list them in tools=[...]
    to expose them for a run. see Environment / ToolsRegistry."""
    Environment.target().register_server(name,
                                         command=command,
                                         url=url,
                                         env=env,
                                         headers=headers,
                                         cwd=cwd)
