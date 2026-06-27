"""Tests for the process-global hook/command/tool registries and the cai.hook /
cai.command / cai.tool decorators that feed them. The conftest fixture resets
them around every test, so each starts from empty."""
import pytest

import cai
from cai.hooks import HooksRegistry
from cai.commands import CommandsRegistry
from cai.tools import ToolsRegistry


def test_cai_hook_registers_and_bakes_into_new_registry():
    @cai.hook("after_turn")
    def h(ctx):
        return "x"

    registered = HooksRegistry.registered()
    assert len(registered) == 1
    event, fn, origin = registered[0]
    assert event == "after_turn"
    assert fn is h
    assert origin == h.__code__.co_filename

    assert ("after_turn", h) in HooksRegistry().pairs()


def test_cai_hook_rejects_unknown_event():
    with pytest.raises(ValueError):
        @cai.hook("not_a_real_event")
        def h(ctx):
            pass


def test_registries_start_empty_each_test():
    assert HooksRegistry.registered() == []
    assert CommandsRegistry.commands() == {}


def test_cai_command_bare_and_with_args():
    @cai.command
    def alpha(ctx):
        pass

    @cai.command(name="b", help="beta")
    def beta(ctx):
        pass

    commands = CommandsRegistry.commands()
    assert "alpha" in commands
    assert commands["alpha"].fn is alpha
    assert commands["b"].help == "beta"
    assert commands["b"].fn is beta


def test_later_command_of_same_name_wins():
    @cai.command(name="dup", help="first")
    def first(ctx):
        pass

    @cai.command(name="dup", help="second")
    def second(ctx):
        pass

    assert CommandsRegistry.commands()["dup"].help == "second"


def test_cai_tool_registers_globally():
    @cai.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    registered = ToolsRegistry.registered()
    assert "add" in registered
    fn, origin = registered["add"]
    assert fn is add
    assert origin == add.__code__.co_filename
    assert ToolsRegistry.global_function("add") is add


def test_cai_tool_listed_in_available_tools():
    @cai.tool
    def greet(name: str) -> str:
        """Greet someone."""
        return "hi " + name

    assert "greet" in ToolsRegistry.available_tools()


def test_agent_registry_selects_function_tool_by_name():
    @cai.tool
    def ping() -> str:
        """Ping."""
        return "pong"

    registry = ToolsRegistry()
    registry.select("ping")
    assert "ping" in registry.selected()
    assert registry.dispatch("ping", {}) == "pong"


def test_later_tool_of_same_name_wins():
    @cai.tool
    def dup() -> str:
        """first."""
        return "first"

    @cai.tool
    def dup() -> str:  # noqa: F811
        """second."""
        return "second"

    assert ToolsRegistry.global_function("dup")() == "second"


def test_cai_mcp_server_declares_local():
    cai.mcp_server("github",
                   command=["npx", "-y", "@modelcontextprotocol/server-github"],
                   env={"GITHUB_TOKEN": "x"},
                   cwd="/tmp")

    declared = ToolsRegistry.declared_servers()
    assert declared["github"]["command"] == ["npx", "-y", "@modelcontextprotocol/server-github"]
    assert declared["github"]["env"] == {"GITHUB_TOKEN": "x"}
    assert declared["github"]["cwd"] == "/tmp"


def test_cai_mcp_server_declares_remote():
    cai.mcp_server("linear",
                   url="https://mcp.linear.app/mcp",
                   headers={"Authorization": "Bearer t"})

    declared = ToolsRegistry.declared_servers()
    assert declared["linear"] == {"url": "https://mcp.linear.app/mcp",
                                  "headers": {"Authorization": "Bearer t"}}


def test_cai_mcp_server_rejects_bad_specs():
    with pytest.raises(ValueError):
        cai.mcp_server("x")                                   # neither command nor url
    with pytest.raises(ValueError):
        cai.mcp_server("x", command=["c"], url="http://y")    # both
    with pytest.raises(ValueError):
        cai.mcp_server("x", command="c")                      # string, not argv list
    with pytest.raises(ValueError):
        cai.mcp_server("x", url="http://y", env={"A": "1"})   # env on a url server
    with pytest.raises(ValueError):
        cai.mcp_server("x", command=["c"], headers={"A": "1"})  # headers on a command server


def test_later_mcp_server_of_same_name_wins():
    cai.mcp_server("dup", command=["first"])
    cai.mcp_server("dup", url="http://second")

    assert ToolsRegistry.declared_servers()["dup"] == {"url": "http://second"}


def test_reset_global_clears_both():
    @cai.hook("after_run")
    def h(ctx):
        pass

    @cai.command(name="c")
    def c(ctx):
        pass

    @cai.tool
    def t() -> str:
        """t."""
        return "t"

    cai.mcp_server("srv", command=["c"])

    HooksRegistry.reset_global()
    CommandsRegistry.reset_global()
    ToolsRegistry.reset_global()
    assert HooksRegistry.registered() == []
    assert CommandsRegistry.commands() == {}
    assert ToolsRegistry.registered() == {}
    assert ToolsRegistry.declared_servers() == {}
