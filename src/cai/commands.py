"""commands: the `:`-commands registered via cai.command.

A Command is a named handler registered through the cai.command decorator into
the process-global CommandsRegistry; cai.userconfig.load() imports the extensions
so theirs land there too. The tui routes an unrecognized `:`-name to it, calling
its fn with a CommandContext - the argument text after the name, the agent
client, and the screen to write back to."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Command:
    fn: object
    help: str = ""
    origin: str = None


class CommandsRegistry:
    """process-global :commands registered via cai.command, as name -> Command.
    the tui reads commands() to populate its `:`-dispatch and palette; once
    cai.userconfig.load() imports the extensions every session shares the set.
    a later registration of the same name wins (the user's init.py runs last)."""

    _registered = {}

    @classmethod
    def register_global(cls, name, fn, help=""):
        origin = None
        code = getattr(fn, "__code__", None)
        if code is not None:
            origin = code.co_filename
        cls._registered[name] = Command(fn=fn, help=help, origin=origin)

    @classmethod
    def commands(cls):
        """the registered commands as a name -> Command dict (a copy)."""
        return dict(cls._registered)

    @classmethod
    def reset_global(cls):
        """drop every registered command (test isolation)."""
        cls._registered = {}


def command(fn=None, *, name=None, help=""):
    """decorator: register a `:command`. use bare (@cai.command) or with args
    (@cai.command(name="compact", help="...")). the function is the handler,
    called with a CommandContext; the name defaults to the function name."""
    def decorator(target):
        cmd_name = name or target.__name__
        CommandsRegistry.register_global(cmd_name, target, help=help)
        return target
    if fn is not None:
        return decorator(fn)
    return decorator


class CommandContext:
    """what a command's fn receives: the text after the command name, the agent
    client to drive, and the screen to write back to."""

    def __init__(self, args, client, screen):
        self.args = args
        self.client = client
        self.screen = screen

    def write(self, text):
        from cai.screen import Screen
        self.screen.write(text, kind=Screen.META)
