"""commands: the types behind the user/extension `:`-commands.

A Command is a named handler an extension registers through cai.userconfig
(reg.add_command). The tui routes an unrecognized `:`-name to it, calling its fn
with a CommandContext - the argument text after the name, the agent client, and
the screen to write back to."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Command:
    fn: object
    help: str = ""


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
