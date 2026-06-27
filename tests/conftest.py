"""Shared fixtures. cai.hook / cai.command / cai.tool register into process-global
stores on HooksRegistry / CommandsRegistry / ToolsRegistry; reset them around every
test so a hook, command or function tool registered by one test never leaks into
another."""
import pytest

from cai.hooks import HooksRegistry
from cai.commands import CommandsRegistry
from cai.tools import ToolsRegistry


@pytest.fixture(autouse=True)
def _reset_global_registries():
    HooksRegistry.reset_global()
    CommandsRegistry.reset_global()
    ToolsRegistry.reset_global()
    yield
    HooksRegistry.reset_global()
    CommandsRegistry.reset_global()
    ToolsRegistry.reset_global()
