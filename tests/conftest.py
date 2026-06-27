"""Shared fixtures. cai.hook / cai.command register into process-global stores
on HooksRegistry / CommandsRegistry; reset them around every test so a hook or
command registered by one test never leaks into another."""
import pytest

from cai.hooks import HooksRegistry
from cai.commands import CommandsRegistry


@pytest.fixture(autouse=True)
def _reset_global_registries():
    HooksRegistry.reset_global()
    CommandsRegistry.reset_global()
    yield
    HooksRegistry.reset_global()
    CommandsRegistry.reset_global()
