"""cai.py — the built-in sub-agent MCP server, shipped with cai.

Unlike a user server under ~/.config/cai/mcps/, this lives in the package's
builtins/ dir and loads by default. Its tools are surfaced prefixed with the
file name: ``cai__launch_agent``, ``cai__wait_agent``, ``cai__kill_agent``.

These are stubs: the signatures are the contract the real sub-agent layer will
bind to (a child agent run on a background thread, collected by the parent),
but for now each tool just reports that sub-agents are not wired up yet. The
implementation lands in a later layer.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="cai")

_NOT_WIRED = "Error: sub-agents are not implemented yet (cai builtin stub)."


@mcp.tool()
def launch_agent(prompt: str,
                 name: str,
                 tools: list[str] = None,
                 skills: list[str] = None,
                 model: str = "",
                 system_prompt: str = "") -> str:
    """Launch a background sub-agent to complete a self-contained task; name it
    with descriptive dash-delimited words (e.g. 'audit-auth-flow').

    The child shares your working directory but not your conversation, so
    ``prompt`` must be self-contained. ``tools`` and ``skills`` are lists of
    names a child may inherit (a subset of yours); ``model`` and
    ``system_prompt`` override the inherited model / replace the prompt. It runs
    in the background — collect its result with ``wait_agent``.
    """
    return _NOT_WIRED


@mcp.tool()
def wait_agent(agent_id: str, timeout: int = 300) -> str:
    """Block until the given sub-agent finishes and return its final answer.

    On timeout the sub-agent keeps running; call wait_agent again to keep
    waiting.
    """
    return _NOT_WIRED


@mcp.tool()
def kill_agent(agent_id: str) -> str:
    """Kill a running sub-agent now; it winds down in the background.

    Returns immediately. Collect whatever partial output it produced with
    ``wait_agent('<name>')``.
    """
    return _NOT_WIRED


if __name__ == "__main__":
    mcp.run(transport="stdio")
