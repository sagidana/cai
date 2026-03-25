"""
screen.py — terminal TUI via prompt_toolkit.

Layout is managed by prompt_toolkit:
  - scrollable output above the prompt (regular stdout)
  - status bar via bottom_toolbar (visible while prompting)
  - multi-line input with Enter-to-submit / Alt-Enter-for-newline
  - tools overlay via checkboxlist_dialog
"""

import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import has_focus


class Screen:
    """Sole owner of terminal drawing for --interactive mode."""

    # ANSI style constants used by cli.py
    _USER_STYLE  = "\033[1m"      # bold  — user messages
    _LLM_STYLE   = "\033[36m"     # cyan  — LLM responses
    _META_STYLE  = "\033[2;37m"   # dim gray — tool calls / metadata
    _ERROR_STYLE = "\033[1;31m"   # bold red — errors
    _RESET       = "\033[m"

    def __init__(self):
        self._status_text: str = ""
        self._cmd_completions: list[str] = []

        kb = KeyBindings()

        @kb.add("enter", filter=has_focus(DEFAULT_BUFFER), eager=True)
        def _submit(event):
            """Enter always submits (even in multiline mode)."""
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _newline(event):
            """Alt-Enter inserts a literal newline for multiline input."""
            event.current_buffer.insert_text("\n")

        self._session = PromptSession(
            history=InMemoryHistory(),
            bottom_toolbar=self._get_toolbar,
            key_bindings=kb,
            multiline=True,
            prompt_continuation="  ",   # matches original _CONT_PREFIX width
            style=Style.from_dict({
                "bottom-toolbar": "bg:#3a3a3a #00aaff bold",
            }),
        )

    def _get_toolbar(self):
        return HTML(f" {self._status_text} ")

    # ------------------------------------------------------------------ public API

    def write(self, text: str):
        """Write text into the scrollable output area."""
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()

    def set_status(self, text: str):
        """Update the status bar text (shown while prompting)."""
        self._status_text = text

    def show_prompt_placeholder(self, msg: str = "> "):
        """Cosmetic hint while the LLM is thinking — no-op with prompt_toolkit."""
        pass

    def set_cmd_completions(self, cmds: list[str]):
        """Register command names tab-completed after ':' (e.g. ':compact')."""
        self._cmd_completions = list(cmds)

    def prompt(self, msg: str = "> ") -> str:
        """
        Collect user input with full line-editing support. Returns the string.

        Blocking. Raises KeyboardInterrupt on Ctrl-C, EOFError on Ctrl-D with
        an empty buffer.

        Vim-style commands (:compact, :tools, :clear) are typed directly and
        tab-completed; cli.py detects them via the leading ':'.
        """
        completer = None
        if self._cmd_completions:
            completer = WordCompleter(
                [f":{c}" for c in self._cmd_completions],
                sentence=True,
                match_middle=False,
            )

        return self._session.prompt(msg, completer=completer)

    def prompt_tools_overlay(self, tool_names: list[str], enabled: set) -> set:
        """
        Show an interactive checkbox list for toggling tools.
        Returns the updated enabled set.
        """
        if not tool_names:
            return set(enabled)

        from prompt_toolkit.shortcuts import checkboxlist_dialog

        result = checkboxlist_dialog(
            title="Tools",
            text="Space/Tab to toggle   Enter to confirm   Ctrl-C to cancel",
            values=[(name, name) for name in tool_names],
            default_values=[name for name in tool_names if name in enabled],
        ).run()

        return set(result) if result is not None else set(enabled)

    def close(self):
        """Restore terminal to its normal state."""
        sys.stdout.write("\033[?25h\033[m")  # show cursor, reset attributes
        sys.stdout.flush()
