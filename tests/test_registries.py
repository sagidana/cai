"""Tests for the process-global hook/command registries and the cai.hook /
cai.command decorators that feed them. The conftest fixture resets both around
every test, so each starts from empty."""
import pytest

import cai
from cai.hooks import HooksRegistry
from cai.commands import CommandsRegistry


def test_cai_hook_registers_and_bakes_into_new_registry():
    @cai.hook("after_turn")
    def h(ctx):
        return "x"

    registered = HooksRegistry.registered()
    assert len(registered) == 1
    event, fn, origin = registered[0]
    assert event == "after_turn"
    assert fn is h
    assert origin == h.__code__.co_filename

    assert ("after_turn", h) in HooksRegistry().pairs()


def test_cai_hook_rejects_unknown_event():
    with pytest.raises(ValueError):
        @cai.hook("not_a_real_event")
        def h(ctx):
            pass


def test_registries_start_empty_each_test():
    assert HooksRegistry.registered() == []
    assert CommandsRegistry.commands() == {}


def test_cai_command_bare_and_with_args():
    @cai.command
    def alpha(ctx):
        pass

    @cai.command(name="b", help="beta")
    def beta(ctx):
        pass

    commands = CommandsRegistry.commands()
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

    assert CommandsRegistry.commands()["dup"].help == "second"


def test_reset_global_clears_both():
    @cai.hook("after_run")
    def h(ctx):
        pass

    @cai.command(name="c")
    def c(ctx):
        pass

    HooksRegistry.reset_global()
    CommandsRegistry.reset_global()
    assert HooksRegistry.registered() == []
    assert CommandsRegistry.commands() == {}
