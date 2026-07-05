"""Tests for cai.config - the config.json bootstrap and the cai.settings shadow:
a non-None settings attribute of the same name wins over the config.json field,
for both required fields (load_config) and optional keys (load_optional). Each
test gets its own ~/.config/cai via HOME; the autouse fixture in conftest resets
the default Environment so a settings edit never leaks across tests."""
import json
import os

import pytest

from cai import config
from cai.environment import Environment


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _write_config(data):
    path = config.config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def test_load_optional_reads_the_file():
    _write_config({"base_url": "u", "model": "m", "ssl_verify": False})
    assert config.load_optional("ssl_verify", True) is False
    assert config.load_optional("absent", "d") == "d"


def test_settings_shadow_an_optional_key():
    _write_config({"base_url": "u", "model": "m", "python_sandbox": "kernel"})
    Environment.default().settings.python_sandbox = "hook"
    assert config.load_optional("python_sandbox", "kernel") == "hook"


def test_settings_shadow_a_required_field():
    _write_config({"base_url": "file-url", "model": "file-model"})
    Environment.default().settings.model = "init-model"
    cfg = config.load_config()
    assert cfg.model == "init-model"
    assert cfg.base_url == "file-url"


def test_unset_settings_fall_through_to_the_file():
    _write_config({"base_url": "u", "model": "m"})
    # the shadow fields default to None - nothing set, so the file wins
    assert config.load_optional("python_sandbox", "kernel") == "kernel"
    assert config.load_config().model == "m"


def test_python_venv_redirects_venv_dir():
    _write_config({"base_url": "u", "model": "m", "python_venv": "~/envs/proj"})
    assert config.venv_dir() == os.path.expanduser("~/envs/proj")
    assert config.venv_python() == os.path.join(
        os.path.expanduser("~/envs/proj"), "bin", "python")


def test_python_venv_settings_shadow_wins_over_file():
    _write_config({"base_url": "u", "model": "m", "python_venv": "/from/file"})
    Environment.default().settings.python_venv = "/from/init"
    assert config.venv_dir() == "/from/init"
