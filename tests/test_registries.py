"""Tests for the cai.hook / cai.command / cai.tool / cai.mcp_server decorators
and the Environment stores they register into. Outside a load() they target the
process-default Environment; the conftest fixture makes that default fresh for
every test."""
import pytest

import cai
from cai.environment import Environment
from cai.tools import ToolsRegistry


def test_cai_hook_registers_on_the_default_env():
    @cai.hook("after_turn")
    def h(ctx):
        return "x"

    hooks = Environment.default().hooks()
    assert hooks == [("after_turn", h)]


def test_cai_hook_rejects_unknown_event():
    with pytest.raises(ValueError):
        @cai.hook("not_a_real_event")
        def h(ctx):
            pass


def test_default_env_starts_empty_each_test():
    env = Environment.default()
    assert env.hooks() == []
    assert env.commands() == {}
    assert env.function_tools() == {}
    assert env.declared_servers() == {}


def test_cai_command_bare_and_with_args():
    @cai.command
    def alpha(ctx):
        pass

    @cai.command(name="b", help="beta")
    def beta(ctx):
        pass

    commands = Environment.default().commands()
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

    assert Environment.default().commands()["dup"].help == "second"


def test_cai_tool_registers_on_the_default_env():
    @cai.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    env = Environment.default()
    assert env.function_tool("add") is add
    assert env.function_tools() == {"add": add}


def test_cai_tool_listed_in_available_tools():
    @cai.tool
    def greet(name: str) -> str:
        """Greet someone."""
        return "hi " + name

    assert "greet" in Environment.default().available_tools()


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

    assert Environment.default().function_tool("dup")() == "second"


def test_cai_wrap_registers_and_dispatches_through_target():
    @cai.tool
    def read_note(id: str, cwd: str = "~/notes") -> str:
        """Read a note."""
        return f"note {id} from {cwd}"

    @cai.wrap("read_note")
    def read_memory(call, id: str) -> str:
        """Read a note from the memory vault."""
        return call(id=id, cwd="~/vault")

    registry = ToolsRegistry()
    registry.select("read_memory")
    assert registry.dispatch("read_memory", {"id": "7"}) == "note 7 from ~/vault"


def test_wrap_target_registered_but_unselected():
    @cai.tool
    def inner() -> str:
        """inner."""
        return "x"

    @cai.wrap("inner")
    def outer(call) -> str:
        """outer."""
        return call()

    registry = ToolsRegistry()
    registry.select("outer")
    assert registry.selected() == ["outer"]
    assert registry.has("inner")
    assert registry.dispatch("inner", {}) == "x"


def test_wrap_schema_hides_the_injected_first_param():
    @cai.tool
    def inner(a: str) -> str:
        """inner."""
        return a

    @cai.wrap("inner")
    def outer(call, a: str, extra: int = 1) -> str:
        """outer."""
        return call(a=a)

    registry = ToolsRegistry()
    registry.select("outer")
    schema = registry.tools[0]["function"]
    assert list(schema["parameters"]["properties"].keys()) == ["a", "extra"]
    assert schema["parameters"]["required"] == ["a"]
    assert schema["description"] == "outer."


def test_wrap_can_rework_the_targets_result():
    @cai.tool
    def shout(text: str) -> str:
        """shout."""
        return text.upper()

    @cai.wrap("shout")
    def polite_shout(call, text: str) -> str:
        """polite."""
        return call(text=text) + ", please"

    registry = ToolsRegistry()
    registry.select("polite_shout")
    assert registry.dispatch("polite_shout", {"text": "go"}) == "GO, please"


def test_wrap_chain_composes():
    @cai.tool
    def base(text: str) -> str:
        """base."""
        return text

    @cai.wrap("base")
    def middle(call, text: str) -> str:
        """middle."""
        return call(text=text) + "+middle"

    @cai.wrap("middle")
    def top(call, text: str) -> str:
        """top."""
        return call(text=text) + "+top"

    registry = ToolsRegistry()
    registry.select("top")
    assert registry.dispatch("top", {"text": "t"}) == "t+middle+top"
    assert registry.has("middle")
    assert registry.has("base")


def test_wrap_with_missing_target_is_skipped():
    @cai.wrap("ghost__tool")
    def haunted(call) -> str:
        """haunted."""
        return call()

    registry = ToolsRegistry()
    registry.select("haunted")
    assert not registry.has("haunted")
    assert registry.selected() == []


def test_wrap_wrapping_itself_is_skipped():
    @cai.wrap("loop")
    def loop(call) -> str:
        """loop."""
        return call()

    registry = ToolsRegistry()
    registry.select("loop")
    assert not registry.has("loop")


def test_wrap_rejects_a_function_without_parameters():
    with pytest.raises(TypeError):
        @cai.wrap("whatever")
        def bare() -> str:
            """bare."""
            return "x"


def test_wrap_rejects_a_non_string_target():
    def target() -> str:
        """t."""
        return "t"

    with pytest.raises(TypeError):
        cai.wrap(target)


def test_cai_mcp_server_declares_local():
    cai.mcp_server("github",
                   command=["npx", "-y", "@modelcontextprotocol/server-github"],
                   env={"GITHUB_TOKEN": "x"},
                   cwd="/tmp")

    declared = Environment.default().declared_servers()
    assert declared["github"]["command"] == ["npx", "-y", "@modelcontextprotocol/server-github"]
    assert declared["github"]["env"] == {"GITHUB_TOKEN": "x"}
    assert declared["github"]["cwd"] == "/tmp"


def test_cai_mcp_server_declares_remote():
    cai.mcp_server("linear",
                   url="https://mcp.linear.app/mcp",
                   headers={"Authorization": "Bearer t"})

    declared = Environment.default().declared_servers()
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

    assert Environment.default().declared_servers()["dup"] == {"url": "http://second"}


def test_private_env_is_isolated_from_the_default():
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

    private = Environment()
    assert private.hooks() == []
    assert private.commands() == {}
    assert private.function_tools() == {}
    assert private.declared_servers() == {}
    assert Environment.default().hooks() != []


def test_registry_resolves_names_against_its_own_env():
    # two envs, two tools of the same name: each registry dispatches its own.
    env_a = Environment()
    env_b = Environment()

    def greet() -> str:
        """a."""
        return "from a"
    env_a.register_tool(greet)

    def greet() -> str:  # noqa: F811
        """b."""
        return "from b"
    env_b.register_tool(greet)

    registry_a = ToolsRegistry(env_a)
    registry_a.select("greet")
    registry_b = ToolsRegistry(env_b)
    registry_b.select("greet")
    assert registry_a.dispatch("greet", {}) == "from a"
    assert registry_b.dispatch("greet", {}) == "from b"
