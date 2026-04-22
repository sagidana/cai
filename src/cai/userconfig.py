"""
userconfig.py — user-facing config and hook surface.

Backs three public names re-exported from the cai package:

    cai.config      — attribute-style overrides merged on top of config.json
    cai.hook        — decorator to register a hook globally
    cai.load_init() — read (and on first run, create) ~/.config/cai/init.py

bootstrap() in core.py reads from here: it merges ``config._overrides`` into
the loaded config.json dict, and replaces ``llm.DEFAULT_HOOKS`` with whatever
``_user_hooks`` has collected. Importing this module has no side effects.
"""

import logging
import os

log = logging.getLogger("cai.userconfig")


class _Config:
    """Attribute-style view onto pending config overrides.

    Writes go into a private dict; bootstrap() drains it on top of config.json.
    Reads return whatever has been set here (not the merged value) — use the
    BootstrapContext.config dict or ctx.config.get(...) to see the resolved
    config after bootstrap.
    """

    def __init__(self):
        object.__setattr__(self, "_overrides", {})

    def __setattr__(self, key, value):
        self._overrides[key] = value

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._overrides[key]
        except KeyError:
            raise AttributeError(f"cai.config has no value set for {key!r}")

    def __delattr__(self, key):
        self._overrides.pop(key, None)

    def __repr__(self):
        return f"cai.config({self._overrides!r})"

    def _as_dict(self) -> dict:
        return dict(self._overrides)


config = _Config()

_user_hooks: list = []
_user_tools: list = []


def tool(fn):
    """Register a Python callable as a tool available to every Harness/CLI turn.

    Equivalent to passing ``functions=[fn]`` to a ``Harness(...)`` — but global,
    so every Harness/CLI session in this process sees it once bootstrap() has
    run. The OpenAI tool schema is derived from the function signature: type
    annotations drive param types, the first docstring line becomes the
    description. Parameters without defaults are required.

    Usage::

        @cai.tool
        def weather(city: str) -> str:
            '''Return current weather for a city.'''
            ...

    Registered tools join the shared ``available_tools`` list; Harness callers
    still need to list them in ``tools=[...]`` to expose them for a given run,
    matching how MCP-server tools behave.
    """
    _user_tools.append(fn)
    return fn


def hook(event: str):
    """Register a user hook at the given event.

    Equivalent to appending ``(event, fn)`` to a ``Harness(hooks=[...])`` list,
    but global — every Harness/CLI session in this process picks it up once
    bootstrap() has run.

    Usage::

        @cai.hook("before_tool_call")
        def veto_rm(ctx):
            if ctx["name"] == "run_shell" and "rm -rf" in ctx["args"].get("cmd", ""):
                return False

    Valid event names live in ``cai.llm.VALID_HOOK_EVENTS``.
    """
    def deco(fn):
        _user_hooks.append((event, fn))
        return fn
    return deco


_loaded_paths: set = set()


_DEFAULT_INIT_PY = '''\
"""init.py — cai user config.

Like vim's init.vim: executed once at startup to shape cai the way you want.
Values set here override ~/.config/cai/config.json. Hooks registered here
fire for every agent turn, both CLI and SDK (when cai.load_init() is called).

To reset to defaults, delete this file — cai will regenerate it on next run.
"""
import cai

# ─── Options ────────────────────────────────────────────────────────────────
# Any key from config.json is settable here. Uncomment to override.

# cai.config.model = "anthropic/claude-opus-4"
# cai.config.base_url = "https://openrouter.ai/api/v1"
# cai.config.prompt_mode = "local"            # or "sota"
# cai.config.ssl_verify = True
# cai.config.stuck_detection = False
# cai.config.observation_mask_pct = 0.60
# cai.config.observation_mask_keep = 3
# cai.config.context_budget_pct = 0.75
# cai.config.tool_result_max_chars = 40000

# ─── Tools ──────────────────────────────────────────────────────────────────
# Register any Python callable as a tool. Type annotations drive the schema;
# the first docstring line becomes the tool description. Params without
# defaults are required.
#
# Registered tools are added to available_tools; Harness callers still need
# to list them in `tools=[...]` to expose them for a given run.
#
#     @cai.tool
#     def word_count(text: str) -> int:
#         """Return the number of whitespace-separated words in text."""
#         return len(text.split())

# ─── Hooks ──────────────────────────────────────────────────────────────────
# Four events fire during each agent turn:
#   before_tool_call    — return False to veto the call
#   after_tool_call     — inspect/rewrite the result
#   after_turn          — fires once per LLM round trip
#   on_final_response   — return a str to rewrite the final reply
#
# The ctx dict passed to each hook is documented in examples/harnesses/sample.py.
#
# Built-in context-budget hooks (both off by default):
#
#     cai.hook("after_turn")(cai.mask_hook)      # mask old tool results
#     cai.hook("after_turn")(cai.compact_hook)   # LLM-summarise middle turns
#
# Your own:
#
#     @cai.hook("before_tool_call")
#     def block_rm(ctx):
#         if ctx["name"] == "run_shell" and "rm -rf" in ctx["args"].get("cmd", ""):
#             return False
'''


def load_init(config_dir=None) -> None:
    """Exec ``~/.config/cai/init.py`` if it exists; create a default on first run.

    Idempotent per path: repeated calls for the same config_dir are a no-op
    after the first. Errors in the user's init.py are logged and printed but
    do not abort — cai keeps running with whatever was set before the failure.

    :param config_dir: override the default ~/.config/cai location
    """
    from cai.core import CONFIG_DIR_DEFAULT

    config_dir = config_dir or CONFIG_DIR_DEFAULT
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    init_path = os.path.join(config_dir, "init.py")
    if init_path in _loaded_paths:
        return

    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write(_DEFAULT_INIT_PY)
        print(f"[*] Created default init.py at {init_path}")

    with open(init_path) as f:
        source = f.read()

    ns = {"__name__": "cai_user_init", "__file__": init_path}
    try:
        exec(compile(source, init_path, "exec"), ns)
    except Exception as e:
        log.exception("init.py at %s raised", init_path)
        print(f"[!] Error in {init_path}: {type(e).__name__}: {e}")

    _loaded_paths.add(init_path)
    log.info("load_init: loaded %s (overrides=%d, hooks=%d, tools=%d)",
             init_path, len(config._overrides), len(_user_hooks), len(_user_tools))
