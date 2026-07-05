"""config: where cai reads its bootstrap settings.

Two files under ~/.config/cai/:
  config.json - required; must define at least `base_url` and `model`.
  api_key     - required; the bearer token for `base_url`.

config.json is parsed into a Config dataclass so its fields are typed and
discoverable (cfg.base_url, cfg.model) rather than string dict keys. Nothing is
defaulted: a missing or incomplete config stops cai with a clear message.

Optional keys (read via load_optional, never required) may also appear:
  ssl_verify - false to skip TLS certificate verification on the api layer,
               for self-signed/internal endpoints; defaults to true."""
import os
import json
import logging
import dataclasses
from dataclasses import dataclass


log = logging.getLogger("cai")


@dataclass
class Config:
    base_url: str
    model: str


# the config.json keys cai requires - sourced from Config so the dataclass is
# the single place fields are declared.
REQUIRED_FIELDS = tuple(f.name for f in dataclasses.fields(Config))


def config_dir():
    return os.path.expanduser("~/.config/cai")


def config_path():
    return os.path.join(config_dir(), "config.json")


def api_key_path():
    return os.path.join(config_dir(), "api_key")


def venv_dir():
    """the managed virtualenv the python tool runs snippets under - separate
    from cai's own interpreter so its package surface is well-defined (empty by
    default). materialized lazily by cai.pytool, not here."""
    return os.path.join(config_dir(), "venv")


def venv_python():
    """the python interpreter inside the managed venv."""
    if os.name == "nt":
        return os.path.join(venv_dir(), "Scripts", "python.exe")
    return os.path.join(venv_dir(), "bin", "python")


def load_config():
    """read ~/.config/cai/config.json into a Config. it must exist and must
    define every required field."""
    path = config_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no config found at {path}\n"
            f'create it with at least:  {{"base_url": "https://openrouter.ai/api/v1", "model": "..."}}')
    with open(path) as f:
        try:
            data = json.load(f)
        except ValueError as e:
            raise ValueError(f"config {path} is not valid JSON: {e}")

    values = {}
    for field in REQUIRED_FIELDS:
        if not data.get(field):
            raise ValueError(f"config {path} is missing required field '{field}'")
        values[field] = data[field]
    return Config(**values)


def load_optional(key, default=None):
    """read an optional key from config.json - one that is not part of the
    required Config set - returning default when the file or key is absent or
    unreadable. for opt-in settings (e.g. default_context_size) that must never
    make an existing config invalid."""
    path = config_path()
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return default
    value = data.get(key)
    if value is None:
        return default
    return value


def load_api_key():
    """read the API key from ~/.config/cai/api_key, or raise with a message
    telling the user how to create it."""
    path = api_key_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no API key found at {path}\n"
            f"create it with:  mkdir -p {config_dir()} && echo 'sk-or-...' > {path}")
    with open(path) as f:
        key = f.read().strip()
    if not key:
        raise ValueError(f"API key file {path} is empty")
    return key
