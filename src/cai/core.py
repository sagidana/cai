"""
core.py — shared helpers for cai.

Everything in here is usable from both the CLI (cli.py) and any programmatic
consumer of cai (SDK). These helpers used to live as private functions inside
cli.py; they were lifted here so they stop being entangled with CLI module
globals and the TUI.

No module-level side effects: importing cai.core does not touch logging
configuration, register MCP servers, or read any files. All of that happens
inside explicit function calls (notably bootstrap()).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("cai.core")


# ─── Paths ────────────────────────────────────────────────────────────────────

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")
USER_SKILLS_DIR = os.path.join(os.path.expanduser("~/.config/cai"), "skills")

CONFIG_DIR_DEFAULT = os.path.expanduser("~/.config/cai")


# ─── Config ───────────────────────────────────────────────────────────────────

# Built-in defaults; users override via ~/.config/cai/init.py (see userconfig.py).
DEFAULT_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "model": "arcee-ai/trinity-mini:free",
    "tool_result_max_chars": 40000,
    "ssl_verify": True,
    "stuck_detection": False,
}


def load_api_key(config_dir: Optional[str] = None) -> str:
    """Read ~/.config/cai/api_key; create an empty file if missing."""
    config_dir = config_dir or CONFIG_DIR_DEFAULT
    api_key_path = os.path.join(config_dir, "api_key")
    if not os.path.exists(api_key_path):
        with open(api_key_path, "w") as f:
            f.write("")
        print(f"[*] Created empty api_key file at {api_key_path}")
    return open(api_key_path).read().strip()


def build_apis(config: dict, api_key: str):
    """Construct OpenAiApi + OpenRouterApi clients from config and api key."""
    from cai.api import OpenAiApi, OpenRouterApi
    openai_api = OpenAiApi(
        config.get('base_url'),
        api_key,
        ssl_verify=config.get('ssl_verify', True),
    )
    openrouter_api = OpenRouterApi(api_key)
    return openai_api, openrouter_api


# ─── Bootstrap ────────────────────────────────────────────────────────────────

@dataclass
class BootstrapContext:
    config: dict
    api_key: str
    openai_api: object
    openrouter_api: object
    available_tools: list = field(default_factory=list)


def bootstrap(overrides: Optional[dict] = None,
              config_dir: Optional[str] = None,
              diag_fn=None,
              load_user_init: bool = False) -> BootstrapContext:
    """Initialise cai's runtime dependencies.

    Reads the api key, builds the OpenAI/OpenRouter clients, registers
    the internal MCP tools server, and wires up the llm module. Returns a
    BootstrapContext exposing everything the caller may need.

    Config precedence, low → high: ``DEFAULT_CONFIG`` → ``init.py`` (if loaded,
    either now via ``load_user_init=True`` or earlier via ``cai.load_init()``)
    → ``overrides`` arg.

    :param overrides:       optional dict of config keys to override after loading
                            (e.g. {"model": "openai/gpt-4o", "base_url": "..."})
    :param config_dir:      override the default ~/.config/cai location
    :param diag_fn:         optional diagnostic function passed to llm.setup()
    :param load_user_init:  if True, exec ~/.config/cai/init.py (creating a default
                            on first run). The CLI opts in; the SDK leaves it off
                            for reproducibility, but SDK users can call
                            ``cai.load_init()`` explicitly before constructing a Harness.
    """
    import cai.tools as _cai_tools
    import cai.llm as _llm
    from cai import userconfig

    log.info("bootstrap: starting")

    _cai_tools.register_server(_cai_tools.INTERNAL_SERVER)

    config = dict(DEFAULT_CONFIG)

    if load_user_init:
        userconfig.load_init(config_dir)

    init_overrides = userconfig.config._as_dict()
    if init_overrides:
        config.update(init_overrides)

    if overrides:
        config.update(overrides)
    api_key = load_api_key(config_dir)
    openai_api, openrouter_api = build_apis(config, api_key)

    # Promote user-registered tools before listing: get_all_tools() also
    # rebuilds the dispatch table, so local functions must be registered
    # first to be callable. register_local_functions is idempotent per
    # callable identity, so repeated bootstraps don't raise.
    if userconfig._user_tools:
        _cai_tools.register_local_functions(userconfig._user_tools, label="init")

    available_tools = _cai_tools.get_all_tools()

    # Promote user-registered hooks to the global default. Replace rather than
    # extend so repeated bootstraps don't stack duplicates.
    _llm.DEFAULT_HOOKS = list(userconfig._user_hooks)

    _llm.setup(config, openai_api, _cai_tools.call_tool, diag_fn=diag_fn)

    log.info("bootstrap: done (base_url=%s, available_tools=%d, user_hooks=%d, user_tools=%d)",
             config.get('base_url'), len(available_tools),
             len(userconfig._user_hooks), len(userconfig._user_tools))

    return BootstrapContext(
        config=config,
        api_key=api_key,
        openai_api=openai_api,
        openrouter_api=openrouter_api,
        available_tools=available_tools,
    )


# ─── Skills ───────────────────────────────────────────────────────────────────

def list_skill_names() -> list:
    """Return sorted list of available skill names (built-in + user)."""
    names = set()
    for d in (SKILLS_DIR, USER_SKILLS_DIR):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.md'):
                    names.add(f[:-3])
    return sorted(names)


def parse_skill_file(text: str):
    """Parse a skill markdown file into (tools: set[str], prompt: str).

    Format:
        tools: tool_a, tool_b
        ---
        ## Skill: Name
        ...prompt body...

    Everything before the first '---' line is metadata (key: value).
    Everything after is the prompt body. If no '---' is present the whole
    file is treated as the prompt body with no tool declarations.
    """
    tools: set = set()
    if '\n---\n' in text:
        header, _, body = text.partition('\n---\n')
        for line in header.splitlines():
            if ':' not in line:
                continue
            key, _, val = line.partition(':')
            if key.strip().lower() == 'tools':
                tools = {t.strip() for t in val.split(',') if t.strip()}
        return tools, body.strip()
    return tools, text.strip()


def load_skills(names):
    """Load and parse skill files by name.

    Search order per name: user skills dir first, then built-in package dir.
    Returns (all_tools: set[str], prompts: list[str]).
    Unknown skill names are logged as warnings and skipped.
    """
    all_tools: set = set()
    prompts: list = []
    for name in names:
        candidates = [
            os.path.join(USER_SKILLS_DIR, f"{name}.md"),
            os.path.join(SKILLS_DIR, f"{name}.md"),
        ]
        for path in candidates:
            try:
                with open(path) as f:
                    tools, prompt = parse_skill_file(f.read())
                all_tools |= tools
                if prompt:
                    prompts.append(prompt)
                log.info("load_skills: loaded %r from %s (tools=%s)", name, path, tools)
                break
            except OSError:
                continue
        else:
            log.warning("load_skills: skill %r not found in %s or %s",
                        name, USER_SKILLS_DIR, SKILLS_DIR)
    return all_tools, prompts


# ─── System prompt assembly ───────────────────────────────────────────────────

def compose_system_prompt(base, skill_prompts) -> str:
    """Compose a system prompt from a user-supplied base and skill prompts.

    ``base`` is the user-supplied system prompt (may be ``None`` or empty).
    Skill prompts are appended after the base. Returns an empty string when
    neither is provided — the caller decides whether to attach a system
    message at all.

    Single source of truth for composing the system prompt; called from
    Harness.__init__, Harness.load, and the CLI :skill / :load handlers so
    that skill mutations reuse identical logic.
    """
    parts = []
    if base:
        parts.append(base)
    parts.extend(skill_prompts)
    return "\n\n".join(parts)
