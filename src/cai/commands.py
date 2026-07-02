"""commands: the `:`-commands registered via cai.command.

A Command is a named handler registered through the cai.command decorator onto
the current Environment; Environment.load() imports the extensions so theirs
land there too. The tui routes an unrecognized `:`-name to its env's commands(),
calling the fn with a CommandContext - the argument text after the name, the
agent client, and the screen to write back to."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cai.environment import Environment

if TYPE_CHECKING:
    from cai.screen import Screen
    from cai.tui import AgentClient


@dataclass
class Command:
    fn: object
    help: str = ""
    origin: str = None


def command(fn=None, *, name=None, help=""):
    """decorator: register a `:command` on the current Environment. use bare
    (@cai.command) or with args (@cai.command(name="compact", help="...")). the
    function is the handler, called with a CommandContext; the name defaults to
    the function name. it lands on the env being load()ed - else the process
    default - and a later registration of the same name wins (the user's init.py
    runs last)."""
    def decorator(target):
        cmd_name = name or target.__name__
        origin = None
        code = getattr(target, "__code__", None)
        if code is not None:
            origin = code.co_filename
        Environment.target().register_command(cmd_name,
                                              Command(fn=target, help=help, origin=origin))
        return target
    if fn is not None:
        return decorator(fn)
    return decorator


@dataclass
class CommandContext:
    """what a command's fn receives: the text after the command name, the agent
    client to drive (get_messages / set_messages / get_info / submit / steer),
    and the screen to write back to. the typed fields let an extension author's
    editor jump from ctx.client.<method> into its definition."""
    args: str
    client: AgentClient
    screen: Screen

    def write(self, text):
        from cai.screen import Screen
        self.screen.write(text, kind=Screen.META)
