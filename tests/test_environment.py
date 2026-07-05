"""Tests for cai.environment - extension discovery, resource dirs, and load()
importing each extension so its cai.hook / cai.command / cai.tool decorators
register onto the loading env (extension tools namespaced at load time). Fully
offline: each test gets its own ~/.config/cai via HOME, and the conftest
fixture makes the default env fresh between tests."""
import os
import sys
import textwrap
import subprocess

import pytest

from cai.environment import Environment, builtin_skills_dir, builtin_mcp_dir
from cai.environment import extensions_dir, init_path, list_extensions
from cai.tools import ToolsRegistry


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))


def _ext(name):
    path = os.path.join(extensions_dir(), name)
    os.makedirs(path, exist_ok=True)
    return path


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(text))


def test_no_extensions_dir_registers_nothing():
    assert list_extensions() == []
    env = Environment().load()
    assert env.extensions == []
    assert env.hooks() == []
    assert env.commands() == {}
    assert env.function_tools() == {}


def test_extensions_sorted_and_resource_dirs():
    _ext("bbb")
    _ext("aaa")
    names = []
    for extension in list_extensions():
        names.append(extension.name)
    assert names == ["aaa", "bbb"]

    root = extensions_dir()
    env = Environment(list_extensions())
    assert env.skill_dirs() == [os.path.join(root, "aaa", "skills"),
                                os.path.join(root, "bbb", "skills"),
                                builtin_skills_dir()]
    assert env.mcp_dirs() == [os.path.join(root, "aaa", "mcps"),
                              os.path.join(root, "bbb", "mcps"),
                              builtin_mcp_dir()]


def test_files_that_are_not_dirs_are_ignored():
    os.makedirs(extensions_dir(), exist_ok=True)
    _write(os.path.join(extensions_dir(), "README.md"), "not an ext")
    assert list_extensions() == []


def test_load_registers_hooks_and_commands_on_the_env():
    path = _ext("fs")
    _write(os.path.join(path, "hooks", "init.py"), """
        import cai
        @cai.hook("after_turn")
        def fold(ctx):
            return "h"
    """)
    _write(os.path.join(path, "commands", "init.py"), """
        import cai
        @cai.command(help="files")
        def fs(ctx):
            return None
    """)
    env = Environment().load()

    hooks = env.hooks()
    assert len(hooks) == 1
    event, fn = hooks[0]
    assert event == "after_turn"
    assert fn.__name__ == "fold"

    assert "fs" in env.commands()
    assert env.commands()["fs"].help == "files"


def test_load_leaves_the_default_env_untouched():
    path = _ext("fs")
    _write(os.path.join(path, "commands", "init.py"), """
        import cai
        @cai.command(help="files")
        def fs(ctx):
            return None
    """)
    env = Environment().load()
    assert "fs" in env.commands()
    assert Environment.default().commands() == {}


def test_extension_function_tool_is_namespaced_by_extension():
    path = _ext("web")
    _write(os.path.join(path, "tools", "net.py"), """
        import cai
        @cai.tool
        def fetch_url(url: str) -> str:
            \"\"\"Fetch a URL.\"\"\"
            return url
    """)
    env = Environment().load()

    # the tool surfaces as '<extension>__<name>', mirroring MCP tools, in the
    # env's store, the available list, and the schema sent to the model.
    fn = env.function_tool("web__fetch_url")
    assert fn is not None
    assert fn("http://x") == "http://x"
    assert "web__fetch_url" in env.available_tools()
    assert "fetch_url" not in env.available_tools()
    registry = ToolsRegistry.for_tools(["web__fetch_url"], env)
    assert registry.selected() == ["web__fetch_url"]
    schema = registry.tools[0]
    assert schema["function"]["name"] == "web__fetch_url"


def test_top_level_user_function_tool_keeps_its_bare_name():
    # a tool defined in the top-level ~/.config/cai/init.py is loaded as
    # 'user', which is not an extension, so it is not namespaced.
    _write(init_path(), """
        import cai
        @cai.tool
        def now() -> str:
            \"\"\"The time.\"\"\"
            return "now"
    """)
    env = Environment().load()

    assert env.function_tool("now") is not None
    assert "now" in env.available_tools()
    assert "user__now" not in env.available_tools()


def test_tools_underscore_files_are_not_imported():
    path = _ext("calc")
    _write(os.path.join(path, "tools", "_helper.py"), """
        import cai
        @cai.tool
        def hidden() -> str:
            \"\"\"Hidden.\"\"\"
            return "x"
    """)
    env = Environment().load()
    assert env.function_tool("hidden") is None
    assert env.function_tool("calc__hidden") is None


def test_agent_run_fires_env_hooks(monkeypatch):
    # the env's hooks reach a run: Agent composes env hooks + its own per run.
    path = _ext("fs")
    _write(os.path.join(path, "hooks", "init.py"), """
        import cai
        @cai.hook("after_run")
        def mark(ctx):
            ctx.messages.append({"role": "user", "content": "hooked"})
    """)
    env = Environment().load()

    class OneTurnApi:
        def chat(self, messages, model, **kw):
            return "done", "", None, {}

    from cai.agent import Agent
    agent = Agent(model="m", api=OneTurnApi(), env=env, stream=False)
    agent.run("hi").wait()
    assert agent.messages[-1] == {"role": "user", "content": "hooked"}


def test_top_level_init_is_user_attributed_and_overrides():
    path = _ext("fs")
    _write(os.path.join(path, "commands", "init.py"), """
        import cai
        @cai.command(name="x", help="from ext")
        def x_ext(ctx): pass
    """)
    _write(init_path(), """
        import cai
        @cai.command(name="x", help="from user")
        def x_user(ctx): pass
    """)
    env = Environment().load()
    command = env.commands()["x"]
    assert command.help == "from user"


def test_command_can_import_sibling_relatively():
    path = _ext("sib")
    _write(os.path.join(path, "helper.py"), "LABEL = 'from-sibling'\n")
    _write(os.path.join(path, "init.py"), """
        import cai
        from . import helper
        @cai.command(name="sib", help=helper.LABEL)
        def sib(ctx): pass
    """)
    env = Environment().load()
    assert env.commands()["sib"].help == "from-sibling"


def test_bare_sibling_import_still_fails():
    path = _ext("bare")
    _write(os.path.join(path, "helper.py"), "LABEL = 'x'\n")
    _write(os.path.join(path, "init.py"), """
        import helper
        import cai
        @cai.command(name="bare")
        def bare(ctx): pass
    """)
    env = Environment().load()
    assert "bare" not in env.commands()


def test_module_body_runs_once_across_loads():
    path = _ext("once")
    log = os.path.join(path, "execs.log")
    _write(os.path.join(path, "init.py"), f"""
        import cai
        with open({log!r}, "a") as f:
            f.write("x")
        @cai.command(name="once")
        def once(ctx): pass
    """)
    env = Environment()
    env.load()
    env.load()
    with open(log) as f:
        assert f.read() == "x"


def test_module_that_raises_is_isolated():
    good = _ext("aaa_good")
    bad = _ext("zzz_bad")
    _write(os.path.join(good, "init.py"), """
        import cai
        @cai.command(name="good")
        def good(ctx): pass
    """)
    _write(os.path.join(bad, "init.py"), """
        raise RuntimeError("boom")
    """)
    env = Environment().load()
    assert "good" in env.commands()


def test_init_can_tune_cai_settings_of_the_loading_env():
    _write(init_path(), """
        import cai
        cai.settings.show_reasoning = False
        cai.settings.skills = ["fs"]
    """)
    env = Environment().load()
    assert env.settings.show_reasoning is False
    assert env.settings.skills == ["fs"]
    assert Environment.default().settings.show_reasoning is True


def test_merge_activations_dedupes_explicit_first():
    merged = Environment.merge_activations(["a", "b"], ["b", "c"])
    assert merged == ["a", "b", "c"]
    assert Environment.merge_activations(None, None) == []


def test_importing_agent_does_not_pull_the_serving_stack():
    # the layer contract: cai.agent must not import cai.subagent (and through
    # it the wire/serving stack) - the env wires the sub-agent trio in at
    # Agent construction, so the import happens then, not at module load.
    # the parent's sys.path rides along: the fixture's HOME isolation hides
    # the user-site dir a `pip install -e --user` cai lives in.
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(sys.path)
    code = "import cai.agent, sys; assert 'cai.subagent' not in sys.modules"
    proc = subprocess.run([sys.executable, "-c", code], env=child_env)
    assert proc.returncode == 0


def test_agent_tools_default_is_the_subagent_and_python_tools():
    tools = Environment().agent_tools(object())
    names = []
    for tool in tools:
        names.append(tool.__name__)
    assert names == ["launch_agent", "wait_agent", "list_agents", "kill_agent", "python"]


def test_env_without_agent_tools_builds_bare_agents():
    # an embedder can strip the default factories: agents on such an env get
    # no bound tools at all.
    env = Environment()
    env._agent_tools = []

    class OneTurnApi:
        def chat(self, messages, model, **kw):
            return "done", "", None, {}

    from cai.agent import Agent
    agent = Agent(model="m", api=OneTurnApi(), env=env)
    assert agent.tools_registry.names() == []
