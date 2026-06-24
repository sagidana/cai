"""config: where cai reads its bootstrap settings.

Minimal for now: the OpenRouter endpoint (hard-wired) and the API key, read
from ~/.config/cai/api_key. The model and prompt come from the command line.
A richer config.json layer (base_url, default model, ssl, ...) can come later."""
import os
import logging


log = logging.getLogger("cai")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.6-plus"


def config_dir():
    return os.path.expanduser("~/.config/cai")


def api_key_path():
    return os.path.join(config_dir(), "api_key")


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
