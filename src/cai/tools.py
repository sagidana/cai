"""tools: load MCP servers into a tool registry from the user's
~/.config/cai/mcps/ and the builtins shipped with cai (builtins/tools/ beside
this module).

Each *.py file in either dir is an MCP server (a FastMCP stdio program). This
module spawns each as a subprocess, speaks the MCP JSON-RPC handshake over its
stdin/stdout, lists its tools, and exposes two things the agentic loop needs:

  registry.tools            - OpenAI-format tool schemas for the model
                              (call_llm tools=)
  registry.dispatch(name, args) - run a tool by name, returns its text result
                              (call_llm tools_dispatch=)

Tool names are namespaced {label}__{name}, where label is the file's basename,
so two servers can each expose a `search` without colliding.

Minimal by design: blocking line-by-line JSON-RPC (no timeouts/retries/threads),
stderr discarded, local stdio servers only - no remote servers, images, or
user-only display blocks. Those are later layers."""
from __future__ import annotations

import os
import sys
import json
import typing
import atexit
import inspect
import logging
import subprocess

from cai import config


log = logging.getLogger("cai")

MCP_PROTOCOL_VERSION = "2024-11-05"


class LocalMCPServer:
    """One MCP stdio server subprocess, spoken to over line-delimited JSON-RPC.
    The loop runs tools one at a time, so this stays single-threaded: write a
    request, read stdout lines until the matching id comes back."""

    def __init__(self, command, label):
        self.label = label
        self._command = command
        self._req_id = 0
        self._process = subprocess.Popen(command,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL,
                                         text=True,
                                         bufsize=1)
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
        return response.get("result", {}).get("tools", [])

    def call_tool(self, name, arguments):
        response = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        if response.get("error") is not None:
            return f"Error: {response['error']}"
        blocks = response.get("result", {}).get("content", [])
        texts = []
        for block in blocks:
            if block.get("type") != "text": continue
            if block.get("text") is None: continue
            texts.append(block["text"])
        return "\n".join(texts)

    def close(self):
        if self._process is None: return
        if self._process.poll() is not None: return
        self._process.terminate()


_PY_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


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
    function["name"] = fn.__name__
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


class ToolRegistry:
    """The tools an Agent/Run can reach: in-process Python functions and MCP
    server tools, in one dispatch table.

    Registering a tool and selecting it are separate: register() makes a tool
    known (and dispatchable), select() marks it active. `tools` - what the model
    sees - is the selected subset; `dispatch` runs any registered tool. So a tool
    can be available without being offered to the model (e.g. the sub-agent tools
    an Agent always registers but only sends once a skill or the user selects).

    An MCP tool referenced by name ('<mcp_name>__<tool_name>') is lazy: the
    server at ~/.config/cai/mcps/<mcp_name>.py is spawned on first use - when
    its schema is read for the model, or when the tool is called - not before."""

    def __init__(self):
        self._functions = {}     # name -> callable
        self._dispatch = {}      # exposed_name -> tagged entry
        self._schemas = {}       # exposed_name -> schema (eager, or resolved lazily)
        self._order = []         # exposed names, registration order
        self._selected = []      # exposed names that are active (sent to the model)
        # mcp_name -> LocalMCPServer: both the spawned-once cache and the set of
        # servers to close.
        self._local_mcp_servers = {}

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
        """discover every tool exposed by every MCP server in the user's mcps dir
        and the builtins, as a flat list of '<mcp_name>__<tool_name>' names (the
        form `tools=` / for_tools accept). each server is spawned briefly to list
        its tools and then closed - nothing is left running. a user file shadows
        a builtin of the same name; a server that fails to start is logged and
        skipped."""
        names = []
        seen_labels = set()
        for directory in (mcps_dir(), builtin_tools_dir()):
            if not os.path.isdir(directory): continue
            for filename in sorted(os.listdir(directory)):
                if not filename.endswith(".py"): continue
                if filename.startswith("_"): continue
                label = filename[:-len(".py")]
                if label in seen_labels: continue   # user file shadows a builtin
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
        the function's __name__; a duplicate name raises ValueError."""
        name = fn.__name__
        self._is_name_free(name)
        self._functions[name] = fn
        self._dispatch[name] = ("function", name)
        self._schemas[name] = schema_from_function(fn)
        self._order.append(name)

    def register_mcp_tool(self, name):
        """register an MCP tool reference '<mcp_name>__<tool_name>'. the server
        at ~/.config/cai/mcps/<mcp_name>.py is loaded into this registry now -
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
        """make one tool known (dispatchable) without selecting it: a callable
        becomes a function tool, a string is an MCP tool reference '<mcp>__<tool>'.
        a tool already registered is left as-is, unless override=True, which
        replaces the existing registration with this tool - e.g. to bind an
        inherited tool name to this agent's own tool. like any fresh
        registration, the replacement is left unselected."""
        name = tool.__name__ if callable(tool) else tool
        if self.has(name):
            if not override: return
            self.remove(name)
        if callable(tool):
            self.register_function(tool)
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
        name = tool.__name__ if callable(tool) else tool
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
        """the LocalMCPServer for mcp_name, spawned (once) from its source file
        and cached in this registry. the file is resolved from the user's
        ~/.config/cai/mcps/ first, then the builtins shipped with cai."""
        server = self._local_mcp_servers.get(mcp_name)
        if server is not None:
            return server
        path = _mcp_server_path(mcp_name)
        if path is None:
            raise FileNotFoundError(
                f"no MCP server {mcp_name!r} in {mcps_dir()} or {builtin_tools_dir()}")
        server = LocalMCPServer([sys.executable, path], mcp_name)
        self._local_mcp_servers[mcp_name] = server
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
        for server in self._local_mcp_servers.values():
            try:
                server.close()
            except Exception:
                log.exception("closing MCP server %r failed", server.label)


def mcps_dir():
    return os.path.join(config.config_dir(), "mcps")


def builtin_tools_dir():
    """the MCP servers shipped with cai by default, in builtins/tools/ beside
    this module - a second source searched in addition to the user's mcps dir."""
    return os.path.join(os.path.dirname(__file__), "builtins", "tools")


def _mcp_server_path(mcp_name):
    """resolve <mcp_name>.py to a source file, searching the user's mcps dir
    first (so a user file can shadow a builtin) then the bundled builtins.
    None when neither has it."""
    filename = mcp_name + ".py"
    for directory in (mcps_dir(), builtin_tools_dir()):
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return None
