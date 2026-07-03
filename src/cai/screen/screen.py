"""Screen - alternate-screen TUI for --interactive mode.

all content is managed in an internal buffer with a scrollable viewport.
the screen uses vim-like modal keybindings:
  Normal  - navigate content (j/k, Ctrl-U/D, gg/G, /search)
  Insert  - edit the prompt (i to enter, Esc to exit)
  Visual  - select text (v/V to enter, y to yank)
  Command - enter slash commands (: to enter)

layout (top to bottom): content viewport | prompt | status line"""

import json
import os
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty

from .ansi import (
    SGR_RESET, SGR_BOLD, SGR_CYAN, SGR_DIM_GRAY, SGR_BOLD_RED,
    SGR_AZURE_ON_DGRAY,
    CUR_SHOW, CUR_HIDE,
    CURSOR_BAR, CURSOR_BLOCK, CURSOR_RESET,
    ERASE_SCREEN,
    ALT_ENTER, ALT_EXIT,
    MOUSE_ON, MOUSE_OFF,
    BRACKET_PASTE_ON, BRACKET_PASTE_OFF,
    SYNC_START, SYNC_END,
)
from .state import Mode, TUIState, SubmitException, CommandException
from .buffer import ContentBuffer
from .layout import Layout
from .modes import ModeHandler
from .input import read_key

# minimum time between streaming redraws (~60fps).
_RENDER_INTERVAL = 0.016


class Screen:
    """alternate-screen TUI for --interactive mode."""

    _PROMPT_PREFIX = '> '
    _CONT_PREFIX   = '  '

    # what's currently being written. callers pass one of these as `kind`
    # to Screen.write(); the screen owns the style and emits SGR
    # transitions on state change (or on a fresh segment, so a stream of
    # chunks that happens to break on '\n' keeps its color).
    USER      = 'user'
    LLM       = 'llm'
    REASONING = 'reasoning'
    META      = 'meta'
    TOOL      = 'tool'
    ERROR     = 'error'
    DEFAULT   = 'default'

    _KIND_STYLES = {
        USER:      SGR_BOLD,
        LLM:       SGR_CYAN,
        REASONING: SGR_DIM_GRAY,
        META:      SGR_DIM_GRAY,
        TOOL:      SGR_DIM_GRAY,
        ERROR:     SGR_BOLD_RED,
        DEFAULT:   '',
    }

    # back-compat aliases still referenced by cli.py and tests.
    _USER_STYLE   = SGR_BOLD
    _LLM_STYLE    = SGR_CYAN
    _META_STYLE   = SGR_DIM_GRAY
    _ERROR_STYLE  = SGR_BOLD_RED
    _RESET        = SGR_RESET
    _STATUS_STYLE = SGR_AZURE_ON_DGRAY

    _HISTORY_FILE = os.path.join(os.path.expanduser('~'), '.config', 'cai', '.prompts_history')
    _HISTORY_MAX = 1000

    def __init__(self):
        ts = shutil.get_terminal_size()
        self._rows = ts.lines
        self._cols = ts.columns

        self._input_buf = []
        self._cursor_pos = 0
        self._history = self._load_history()
        self._history_idx = -1
        self._in_prompt = False
        self._current_prompt_msg = '> '

        self._status_text = ''
        self._status_right = ''

        self._command_result = None

        # in-memory conversation-history tree for the :history view, bound to
        # the run it was built from so attaching elsewhere starts fresh.
        self._convo_history_tree = None
        self._convo_history_run_id = None

        # command-mode completions: top-level commands and sub-options
        # {cmd_name: [sub_options]} - empty list means no sub-options
        self._cmd_completions = {}

        # command palette (Ctrl-P): [(name, help)] entries plus the subset
        # of commands that expect an argument (picked -> pre-filled ':').
        self._palette_commands = []
        self._palette_arg_commands = set()

        # hover widgets: name -> pre-styled lines, painted over the content
        # viewport (top-right, stacked in insertion order) after every
        # content render so they always sit above the conversation.
        self._widgets = {}

        self._closed = False
        self._resize_pending = False

        self._render_lock = threading.RLock()
        self._interrupt_event = threading.Event()

        # cross-thread UI requests. a non-UI thread (the LLM worker) can ask
        # the main thread - which owns the terminal inside prompt() - to run
        # a blocking interaction (e.g. an allow/deny confirmation) and hand
        # back its result. one slot at a time; the worker serialises its own
        # calls.
        self._req_lock = threading.Lock()
        self._req_pending = None  # (request: dict, done: Event, box: dict) | None

        # set by abort_prompt() so a background thread can make a blocked
        # prompt() return without a keypress (the attach pump on host-exit).
        self._prompt_abort = False

        # focus stack: bottom is 'main' (the conversation view), each
        # overlay pushes its name on entry and pops on exit. only the
        # focused window emits ANSI to the terminal - write()/set_status()/
        # _handle_resize() still update the buffer and state when an overlay
        # owns the screen, but skip the redraw so they don't paint over the
        # overlay's frame.
        self._focus_stack = ['main']

        # optional callback invoked on Ctrl-C when the input buffer is
        # empty. returning True signals that the event was handled (e.g.
        # the LLM was interrupted) and the default double-tap quit logic
        # should be skipped.
        self._interrupt_handler = None

        # set by the caller (cli.py) so Ctrl-X can pull the last (interrupted)
        # user prompt back out of the conversation and into the input box.
        self._recall_handler = None

        # set by the caller (cli.py) while an LLM response is in flight so
        # the TUI can distinguish "interrupt LLM" from "quit" on Ctrl-C.
        self._busy = False

        # track whether new content arrived while user is scrolled up
        self._new_content_below = False

        # what's currently being written (see Screen.USER/LLM/...).
        # transitions emit SGR automatically; a fresh buffer segment in the
        # same kind re-emits the style so the color survives \n boundaries.
        self._current_kind = None

        # batch rendering for streaming: accumulate writes, flush periodically
        self._write_pending = False
        self._last_render_time = 0.0

        self._tty_file = open('/dev/tty', 'rb+', buffering=0)
        self._tty_fd = self._tty_file.fileno()
        self._cooked_attrs = termios.tcgetattr(self._tty_fd)

        # core TUI components
        self._buffer = ContentBuffer(self._cols)
        self._layout = Layout(self._rows, self._cols)
        self._state = TUIState()
        self._modes = ModeHandler()

        # enter alternate screen
        sys.stdout.write(ALT_ENTER + MOUSE_ON + BRACKET_PASTE_ON + ERASE_SCREEN + CUR_HIDE)
        sys.stdout.flush()

        signal.signal(signal.SIGWINCH, self._on_resize)

    @classmethod
    def _load_history(cls):
        """load prompt history from disk. returns most-recent-first list.
        each entry is one JSON-encoded string per file line, so multi-line
        prompts round-trip as a single entry. legacy plain-text lines (from
        before this format) are treated as raw entries."""
        try:
            with open(cls._HISTORY_FILE, 'r') as f:
                raw = [l.rstrip('\n') for l in f if l.strip()]
        except FileNotFoundError:
            return []
        except OSError:
            return []

        entries = []
        for line in raw:
            try:
                value = json.loads(line)
            except ValueError:
                entries.append(line)
                continue
            if isinstance(value, str):
                entries.append(value)
            else:
                entries.append(line)
        entries.reverse()
        return entries[:cls._HISTORY_MAX]

    def _save_history_entry(self, entry):
        """append a single entry to the history file on disk. the entry is
        JSON-encoded so embedded newlines do not split it across multiple
        physical lines."""
        try:
            os.makedirs(os.path.dirname(self._HISTORY_FILE), exist_ok=True)
            with open(self._HISTORY_FILE, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except OSError:
            pass

    def write(self, text, kind=None, block=False):
        """append text to the content buffer and refresh the viewport.

        when kind is one of the class kind constants (USER, LLM, ...), the
        screen owns styling: the caller writes plain text and the screen
        emits SGR transitions on state change, and re-emits the active style
        at the start of a fresh buffer segment so streamed chunks that
        happen to break on '\\n' keep their color.

        kind=None means the caller manages styling itself (raw ANSI in
        text); the screen leaves _current_kind untouched.

        block=True marks the start of a new logical block (a user prompt, an
        assistant message, a tool call, a notice): the screen lays down
        exactly one blank line between it and whatever is already on screen,
        so block separation is owned here rather than summed from per-caller
        newlines. streamed continuation chunks pass block=False."""
        if not text: return

        with self._render_lock:
            # handle resize that may have occurred outside the prompt loop
            if self._resize_pending:
                self._resize_pending = False
                self._handle_resize()

            sep = ''
            if block:
                sep = self._block_separator()
            if kind is not None:
                text = self._apply_kind(text, kind, separated=bool(sep))
            if sep:
                text = sep + text

            self._buffer.append_text(text)
            total = self._buffer.line_count()
            content_rows = self._layout.content_rows

            if self._state.auto_scroll:
                self._state.viewport_offset = max(0, total - content_rows)
                # pin the cursor to the tail while following the stream -
                # otherwise G drops cursor_row on the old bottom and each
                # new line pushes the viewport past it, stranding the block
                # cursor above the viewport.
                self._state.cursor_row = max(0, total - 1)
                self._new_content_below = False
            else:
                self._new_content_below = True

            # an overlay owns the screen - buffer the write but don't paint.
            # the eventual pop_focus -> _restore_after_overlay does a full
            # repaint with whatever accumulated.
            if self._focus_stack[-1] != 'main':
                self._write_pending = True
                return

            # batch rendering: don't redraw more than ~60fps
            now = time.monotonic()
            if now - self._last_render_time >= _RENDER_INTERVAL:
                sys.stdout.write(SYNC_START)
                self._refresh_content()
                self._refresh_status()
                # while the user is actively at the prompt, re-render the
                # input area so their cursor stays parked where they're
                # typing instead of jumping into the content viewport every
                # time the LLM streams another chunk. in NORMAL/VISUAL modes
                # position_cursor parks the block cursor in the content
                # area instead of the prompt.
                if self._in_prompt:
                    self._refresh_input()
                    self._layout.position_cursor(self._state.mode,
                                                 self._state.cursor_row,
                                                 self._state.viewport_offset,
                                                 self._state.cursor_col)
                sys.stdout.write(SYNC_END)
                sys.stdout.flush()
                self._last_render_time = now
                self._write_pending = False
            else:
                self._write_pending = True

    def _apply_kind(self, text, kind, separated=False):
        """derive the SGR/newline prefix required to render text as kind.

        three things the state transition takes care of:
          1. kind change mid-line - insert '\\n' so the new state starts on
             a fresh row instead of continuing the previous one.
          2. kind change - reset the previous style (if any) and open the
             new one.
          3. same kind but last write ended with '\\n' - wrap_ansi resets
             style tracking per segment, so re-open the style or the new
             segment would render in default color.

        separated=True means a block separator already terminated the line,
        so the mid-line '\\n' is not needed - the line is treated as already
        ended for the prefix decision."""
        style = self._KIND_STYLES.get(kind, '')
        prev = self._current_kind
        partial = self._buffer._partial and not separated
        prefix = ''
        if kind != prev:
            if partial:
                prefix += '\n'
            if prev is not None and self._KIND_STYLES.get(prev):
                prefix += SGR_RESET
            if style:
                prefix += style
        elif not partial and style:
            prefix += style
        self._current_kind = kind
        if not prefix:
            return text
        return prefix + text

    def _block_separator(self):
        """leading whitespace that leaves exactly one blank line before the
        next block. nothing when the buffer is empty or already ends on a
        blank line; a single '\\n' after a finished, non-blank line; '\\n\\n'
        to terminate a partial line and then drop one blank line."""
        if self._buffer.line_count() == 0:
            return ''
        if self._buffer._partial:
            return '\n\n'
        if self._buffer.ends_blank():
            return ''
        return '\n'

    def set_status(self, text, right=''):
        """update status bar text and refresh. right is rendered flush against
        the scroll percentage on the far right (e.g. context usage)."""
        self._status_text = text
        self._status_right = right
        with self._render_lock:
            if self._focus_stack[-1] != 'main': return
            if self._resize_pending:
                self._resize_pending = False
                self._handle_resize()
            else:
                self._refresh_status()
            sys.stdout.flush()

    def write_status_hint(self, hint):
        """temporarily show a hint on the status bar."""
        old = self._status_text
        self._status_text = hint
        with self._render_lock:
            if self._focus_stack[-1] == 'main':
                self._refresh_status()
                sys.stdout.flush()
        self._status_text = old

    def set_cmd_completions(self, completions):
        """set command-mode completions: a dict mapping command names to
        lists of sub-options, e.g. {"skill": ["off", "git"], "new": []}."""
        self._cmd_completions = dict(completions)

    def set_palette_commands(self, commands, arg_commands=None):
        """register the Ctrl-P palette entries. commands is a list of
        (name, help) pairs. arg_commands names the subset that takes an
        argument - picking one pre-fills COMMAND mode (":save ") instead of
        dispatching immediately."""
        self._palette_commands = list(commands)
        self._palette_arg_commands = set(arg_commands or ())

    def open_palette(self):
        """show the command palette. returns the picked command or None."""
        from .overlays.palette import prompt_palette_overlay

        if not self._palette_commands:
            return None
        self.push_focus('palette')
        try:
            return prompt_palette_overlay(self, self._palette_commands)
        finally:
            self.pop_focus()

    def add_widget(self, name, lines):
        """add or replace a hover widget: a block of pre-styled lines painted
        over the conversation, anchored top-right. widgets stack vertically in
        insertion order; replacing an existing name keeps its slot. lines carry
        their own SGR styling - the screen just places them."""
        with self._render_lock:
            self._widgets[name] = list(lines)
            if self._focus_stack[-1] != 'main': return
            self._refresh_all()

    def remove_widget(self, name):
        """remove a hover widget. unknown names are ignored."""
        with self._render_lock:
            if name not in self._widgets: return
            del self._widgets[name]
            if self._focus_stack[-1] != 'main': return
            self._refresh_all()

    def _widget_lines(self):
        """all widgets' lines flattened in insertion order, top to bottom,
        with one empty line between blocks. an empty line paints nothing, so
        the gap row shows the conversation through."""
        lines = []
        for name in self._widgets:
            if lines:
                lines.append('')
            lines.extend(self._widgets[name])
        return lines

    def set_interrupt_handler(self, fn):
        """register a Ctrl-C handler invoked when the input buffer is empty.
        the handler should return True if it consumed the interrupt (e.g.
        cancelled a running LLM response); in that case the TUI skips its
        double-tap quit logic."""
        self._interrupt_handler = fn

    def set_recall_handler(self, fn):
        """register a Ctrl-X handler that pulls the last user prompt back into
        the input box. the handler should return the prompt text to recall (it
        also removes that message from the conversation), or None to do nothing
        (mid-run, last message isn't a plain user prompt, etc.)."""
        self._recall_handler = fn

    def set_busy(self, busy):
        """mark whether an LLM response is currently in flight."""
        self._busy = busy

    def abort_prompt(self):
        """make a blocked prompt() return promptly from another thread (no
        keypress). used by the attach pump when the served host exits."""
        self._prompt_abort = True

    def clear_buffer(self):
        """clear the content buffer and refresh."""
        self._buffer.clear()
        self._state.viewport_offset = 0
        self._state.cursor_row = 0
        self._state.auto_scroll = True
        self._new_content_below = False
        self._current_kind = None
        with self._render_lock:
            if self._focus_stack[-1] != 'main': return
            self._refresh_all()
            sys.stdout.flush()

    def close(self):
        """exit alternate screen and restore the terminal."""
        if self._closed: return
        self._closed = True
        # wake a worker blocked in submit_request so it doesn't hang on a UI
        # request that will never be serviced now the prompt loop is gone.
        with self._req_lock:
            pend = self._req_pending
            self._req_pending = None
        if pend is not None:
            pend[1].set()
        sys.stdout.write(f'{MOUSE_OFF}{BRACKET_PASTE_OFF}{ALT_EXIT}{CUR_SHOW}{CURSOR_RESET}{SGR_RESET}\n')
        sys.stdout.flush()
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        try:
            self._tty_file.close()
        except Exception:
            pass

    def view_in_editor(self, text, *, suffix='.txt'):
        """open text in nvim read-only as a pager, then restore the screen.
        for inspecting large content (the system prompt) without dumping it
        into the conversation buffer. the next prompt() repaints; the brief
        cooked window between here and there reads no input."""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(text)
            tmp = f.name
        self.push_focus('editor')
        try:
            sys.stdout.write(f'{ALT_EXIT}{CUR_SHOW}')
            sys.stdout.flush()
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._cooked_attrs)
            subprocess.run(['nvim', '-R', tmp])
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            self.pop_focus()

    def edit_in_editor(self, text, *, suffix='.txt'):
        """open text in nvim (writable), restore the screen, and return the
        edited text - or None if the buffer was left unchanged. companion to
        view_in_editor, which is read-only. focus handling is identical: the
        pump's writes buffer behind the 'editor' focus and repaint on pop."""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(text)
            tmp = f.name
        new_text = text
        self.push_focus('editor')
        try:
            sys.stdout.write(f'{ALT_EXIT}{CUR_SHOW}')
            sys.stdout.flush()
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._cooked_attrs)
            subprocess.run(['nvim', tmp])
            with open(tmp) as f:
                new_text = f.read()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            self.pop_focus()
        if new_text == text:
            return None
        return new_text

    def prompt(self, msg='> '):
        """collect user input. blocking. raises KeyboardInterrupt / EOFError."""
        self._in_prompt = True
        self._current_prompt_msg = msg
        # the input buffer persists across prompt() calls: it is only emptied on
        # submit (below) or Ctrl-C (see modes.py). so a ':' command that exits
        # and re-enters prompt() keeps the half-typed input intact.
        self._history_idx = -1
        self._command_result = None

        # enter insert mode only if the user was following the conversation
        # (auto_scroll still on). otherwise stay in normal mode so we don't
        # jump the cursor while they're reading earlier content.
        if self._state.auto_scroll:
            self._state.mode = Mode.INSERT
            total = self._buffer.line_count()
            content_rows = self._layout.content_rows
            self._state.viewport_offset = max(0, total - content_rows)
            self._new_content_below = False
        else:
            self._state.mode = Mode.NORMAL

        self._layout.update_input_height(self._input_buf, self._PROMPT_PREFIX, self._CONT_PREFIX)

        old = termios.tcgetattr(self._tty_fd)
        result = None
        caught_exc = None
        cmd_result = None

        try:
            tty.setraw(self._tty_fd)
            termios.tcflush(self._tty_fd, termios.TCIFLUSH)
            self._refresh_all()
            if self._state.mode == Mode.INSERT:
                sys.stdout.write(f'{CUR_SHOW}{CURSOR_BAR}')
            else:
                sys.stdout.write(f'{CUR_SHOW}{CURSOR_BLOCK}')
            sys.stdout.flush()

            while True:
                if self._resize_pending:
                    self._resize_pending = False
                    self._handle_resize()

                # service any pending cross-thread UI request (e.g. an
                # allow/deny confirmation issued by the LLM worker) before
                # reading the next key.
                if self._req_pending is not None:
                    self._service_requests()

                # a background thread (e.g. the attach pump on host-exit) asked
                # prompt() to return without a keypress.
                if self._prompt_abort:
                    self._prompt_abort = False
                    break

                rlist, _, _ = select.select([self._tty_fd], [], [], 0.05)

                # flush any pending batched writes
                if self._write_pending:
                    with self._render_lock:
                        sys.stdout.write(SYNC_START)
                        self._refresh_content()
                        self._refresh_status()
                        self._refresh_input()
                        self._layout.position_cursor(self._state.mode,
                                                     self._state.cursor_row,
                                                     self._state.viewport_offset,
                                                     self._state.cursor_col)
                        sys.stdout.write(SYNC_END)
                        sys.stdout.flush()
                        self._write_pending = False
                        self._last_render_time = time.monotonic()

                if not rlist: continue

                key = read_key(self._tty_fd)
                try:
                    self._modes.handle_key(key, self._state, self)
                except SubmitException as e:
                    result = e.value
                    break
                except CommandException as e:
                    cmd_result = e.value
                    break

                sys.stdout.flush()

        except (KeyboardInterrupt, EOFError) as exc:
            caught_exc = exc
        finally:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, old)
            self._in_prompt = False
            if caught_exc is not None:
                raise caught_exc

        if cmd_result is not None:
            # command mode produced a command - store it for the caller
            self._command_result = cmd_result
            return ''

        # clear the input area on submit so the next prompt() call starts clean.
        # the caller is responsible for echoing the submitted text into the
        # content buffer at the right time - when prompts are queued the
        # echo should happen when the LLM actually starts processing it,
        # not at submit time. a ':' command exits via the cmd_result path above,
        # so it never reaches here and keeps the buffer.
        if result is not None:
            self._input_buf = []
            self._cursor_pos = 0
            with self._render_lock:
                self._layout.render_input([],
                                          0,
                                          Mode.NORMAL,
                                          self._PROMPT_PREFIX,
                                          self._CONT_PREFIX,
                                          self._cols)

        return result

    def push_focus(self, name):
        """mark name as the currently-focused window. while the focus stack
        is not at 'main', the conversation view's ANSI emit is suppressed.
        write()/set_status() still update the buffer so nothing is lost -
        they just don't paint."""
        with self._render_lock:
            self._focus_stack.append(name)

    def pop_focus(self):
        """pop the top window. when we land back on 'main', redraw fully."""
        with self._render_lock:
            if len(self._focus_stack) > 1:
                self._focus_stack.pop()
            if self._focus_stack[-1] == 'main':
                self._restore_after_overlay()

    def current_focus(self):
        return self._focus_stack[-1]

    def _restore_after_overlay(self):
        """restore the main TUI after an overlay exits: sync dimensions (the
        overlay may have handled resize), re-enter alternate screen, rewrap
        content, and redraw."""
        # the overlay's resize handler updated self._rows/self._cols
        self._layout.resize(self._rows, self._cols)
        self._buffer.rewrap(self._cols)
        total = self._buffer.line_count()
        content_rows = self._layout.content_rows
        if self._state.auto_scroll:
            self._state.viewport_offset = max(0, total - content_rows)
        sys.stdout.write(ALT_ENTER + MOUSE_ON + BRACKET_PASTE_ON + ERASE_SCREEN)
        self._refresh_all()
        sys.stdout.flush()
        self._write_pending = False
        self._last_render_time = time.monotonic()

    def prompt_tools_overlay(self, tool_entries, enabled):
        """interactive tools toggle overlay. see overlays/tools.py for docs."""
        from .overlays.tools import prompt_tools_overlay

        self.push_focus('tools')
        try:
            return prompt_tools_overlay(self, tool_entries, enabled)
        finally:
            self.pop_focus()

    def prompt_skills_overlay(self, skill_names, active):
        """interactive skills toggle overlay (same UI as the tools toggle).
        skill_names is the list of available skills; active the names
        currently enabled. returns the new active set."""
        from .overlays.tools import prompt_tools_overlay

        entries = [(name, 'skill') for name in sorted(skill_names)]
        self.push_focus('skills')
        try:
            return prompt_tools_overlay(self, entries, set(active))
        finally:
            self.pop_focus()

    def prompt_model_overlay(self, models, prices=None, favorites=False):
        """interactive model picker. returns selected model name or None.
        prices maps model id -> price record for a right-aligned label;
        favorites=True enables the favorites section (the :models view)."""
        from .overlays.model import prompt_model_overlay

        self.push_focus('model')
        try:
            return prompt_model_overlay(self, models, prices=prices, favorites=favorites)
        finally:
            self.pop_focus()

    def prompt_config_overlay(self, settings):
        """interactive session-config menu. see overlays/config.py for docs.
        returns True if any setting changed."""
        from .overlays.config import prompt_config_overlay

        self.push_focus('config')
        try:
            return prompt_config_overlay(self, settings)
        finally:
            self.pop_focus()

    def prompt_session_overlay(self, sessions, preview_fn=None):
        """interactive session picker. returns the selected label or None on
        cancel. sessions is kept in the caller's order (newest-first). the
        list is navigated tools/skills-style (j/k, gg/G, /search) so the
        preview only follows the cursor. preview_fn(label, width, max_lines)
        renders the preview pane for the selected session."""
        from .overlays.model import prompt_model_overlay

        self.push_focus('model')
        try:
            return prompt_model_overlay(self,
                                        sessions,
                                        presorted=True,
                                        noun="sessions",
                                        preview_fn=preview_fn,
                                        navigate=True)
        finally:
            self.pop_focus()

    def prompt_agents_overlay(self, fetch_fn, preview_fn, stop_fn=None, self_id=None):
        """live agents tree (tree(1)-style hierarchy + auto-refreshing preview).
        fetch_fn() returns the node snapshot (see cli._agent_tree_nodes);
        preview_fn(node, width, max_lines) renders the selected agent; stop_fn(id)
        interrupts one live agent (Ctrl-K); self_id marks the agent being viewed
        from. returns the selected run id or None."""
        from .overlays.agents import prompt_agents_overlay

        self.push_focus('agents')
        try:
            return prompt_agents_overlay(self, fetch_fn, preview_fn, stop_fn, self_id)
        finally:
            self.pop_focus()

    def prompt_tree_overlay(self, fetch_fn, **kwargs):
        """generic tree picker (see overlays.tree.prompt_tree_overlay). exposed
        for future hierarchical views; the agents view uses it via
        prompt_agents_overlay."""
        from .overlays.tree import prompt_tree_overlay

        self.push_focus('tree')
        try:
            return prompt_tree_overlay(self, fetch_fn, **kwargs)
        finally:
            self.pop_focus()

    def prompt_messages_overlay(self, messages, *,
                                context_size=0, prompt_tokens=0, sample_chars=0,
                                refetch=None, revision=None):
        """interactive messages overlay. see overlays/messages.py for docs."""
        from .overlays.messages import prompt_messages_overlay

        self.push_focus('messages')
        try:
            return prompt_messages_overlay(self,
                                           messages,
                                           context_size=context_size,
                                           prompt_tokens=prompt_tokens,
                                           sample_chars=sample_chars,
                                           refetch=refetch,
                                           revision=revision)
        finally:
            self.pop_focus()

    def prompt_approval_overlay(self, title, body):
        """blocking allow/deny confirmation. see overlays/approval.py."""
        from .overlays.approval import prompt_approval_overlay

        self.push_focus('approval')
        try:
            return prompt_approval_overlay(self, title, body)
        finally:
            self.pop_focus()

    def prompt_select_overlay(self, options, message=""):
        """blocking single-choice picker (reuses the fuzzy list overlay). returns
        the chosen option or None on cancel. backs a hook's ctx.ui.select."""
        from .overlays.model import prompt_model_overlay

        self.push_focus('model')
        try:
            return prompt_model_overlay(self, list(options), presorted=True,
                                        noun=message or "options", navigate=True)
        finally:
            self.pop_focus()

    def prompt_text_overlay(self, title, default="", secret=False):
        """blocking single-line text input. returns the entered string (an empty
        entry yields default) or None on cancel. backs a hook's ctx.ui.text."""
        from .overlays.textinput import prompt_text_overlay

        self.push_focus('textinput')
        try:
            return prompt_text_overlay(self, title, default=default, secret=secret)
        finally:
            self.pop_focus()

    def submit_request(self, request):
        """hand a blocking UI request to the main thread and wait for its
        result. safe to call from a non-UI thread (the LLM worker): the main
        thread services it from inside prompt() (see _service_requests),
        runs the matching overlay, and wakes us with the result. returns the
        handler's result (e.g. a bool for a 'confirm' request), or None if
        the screen is closed / no UI is available."""
        if self._closed:
            return None
        done = threading.Event()
        box = {}
        with self._req_lock:
            self._req_pending = (request, done, box)
        done.wait()
        return box.get('result')

    def _service_requests(self):
        """run any pending cross-thread UI request on the main thread.
        called from prompt()'s input loop. runs the matching overlay, stores
        the result, wakes the waiting thread, then restores the prompt
        cursor so input resumes cleanly."""
        with self._req_lock:
            pend = self._req_pending
            self._req_pending = None
        if pend is None: return

        request, done, box = pend
        try:
            kind = request.get('kind')
            if kind == 'confirm':
                box['result'] = self.prompt_approval_overlay(
                    request.get('title', 'Allow this action?'),
                    request.get('body', ''),
                )
            elif kind == 'select':
                box['result'] = self.prompt_select_overlay(
                    request.get('options', []),
                    message=request.get('title', ''),
                )
            elif kind == 'text':
                box['result'] = self.prompt_text_overlay(
                    request.get('title', ''),
                    default=request.get('default', ''),
                    secret=request.get('secret', False),
                )
            else:
                box['result'] = None
        finally:
            done.set()

        # the overlay hid the cursor; bring it back for the live prompt.
        if self._in_prompt:
            cursor = CURSOR_BLOCK
            if self._state.mode == Mode.INSERT:
                cursor = CURSOR_BAR
            sys.stdout.write(f'{CUR_SHOW}{cursor}')
            sys.stdout.flush()

    def _refresh_content(self):
        """re-render the content viewport area (and the widgets above it)."""
        self._layout.render_content(self._buffer._lines,
                                    self._layout.content_rows,
                                    self._cols,
                                    cursor_row=self._state.cursor_row,
                                    selection=self._build_selection(),
                                    search_spans=self._build_search_spans(),
                                    viewport_offset=self._state.viewport_offset,
                                    widget_lines=self._widget_lines())

    def _refresh_status(self):
        """re-render the status line."""
        search_buf = None
        if self._state.mode == Mode.SEARCH:
            search_buf = self._state.search_buf
        command_buf = None
        if self._state.mode == Mode.COMMAND:
            command_buf = self._state.command_buf

        self._layout.render_status(self._state.mode,
                                   self._status_text,
                                   status_right=self._status_right,
                                   search_buf=search_buf,
                                   search_direction=self._state.search_direction,
                                   viewport_offset=self._state.viewport_offset,
                                   total_lines=self._buffer.line_count(),
                                   cols=self._cols,
                                   auto_scroll=self._state.auto_scroll,
                                   new_content_below=self._new_content_below,
                                   command_buf=command_buf,
                                   cursor_row=self._state.cursor_row)
        sys.stdout.flush()

    def _refresh_input(self):
        """re-render the input/prompt area."""
        prev_height = self._layout.input_height
        self._layout.update_input_height(self._input_buf, self._PROMPT_PREFIX, self._CONT_PREFIX)
        if self._layout.input_height != prev_height:
            # input area resized - the content viewport grew or shrank.
            # rebalance viewport_offset so the buffer stays anchored, and
            # redraw content + status so rows that transitioned between
            # input and content don't keep stale text.
            total = self._buffer.line_count()
            content_rows = self._layout.content_rows
            max_offset = max(0, total - content_rows)
            if self._state.auto_scroll:
                self._state.viewport_offset = max_offset
            else:
                self._state.viewport_offset = min(self._state.viewport_offset, max_offset)
            self._refresh_content()
            self._refresh_status()
        self._layout.render_input(self._input_buf,
                                  self._cursor_pos,
                                  self._state.mode,
                                  self._PROMPT_PREFIX,
                                  self._CONT_PREFIX,
                                  self._cols,
                                  command_buf=self._state.command_buf)
        sys.stdout.flush()

    def _build_search_spans(self):
        """group the flat match list into per-line (start, end) spans."""
        if not self._state.search_matches:
            return None
        spans = {}
        for row, s, e in self._state.search_matches:
            spans.setdefault(row, []).append((s, e))
        return spans

    def _build_selection(self):
        """build the selection tuple for visual mode, or None."""
        if self._state.mode not in (Mode.VISUAL, Mode.VISUAL_LINE):
            return None
        line_mode = self._state.mode == Mode.VISUAL_LINE
        sr = self._state.visual_anchor_row
        er = self._state.cursor_row
        sc = self._state.visual_anchor_col
        ec = self._state.cursor_col
        if sr > er or (sr == er and sc > ec):
            sr, sc, er, ec = er, ec, sr, sc
        return (sr, er, line_mode, sc, ec)

    def _refresh_all(self):
        """full screen redraw, presented as one synchronized frame."""
        sys.stdout.write(SYNC_START)
        self._layout.update_input_height(self._input_buf, self._PROMPT_PREFIX, self._CONT_PREFIX)
        search_buf = None
        if self._state.mode == Mode.SEARCH:
            search_buf = self._state.search_buf

        self._layout.render_all(self._buffer._lines,
                                self._state.viewport_offset,
                                self._state.mode,
                                self._status_text,
                                self._input_buf,
                                self._cursor_pos,
                                self._PROMPT_PREFIX,
                                self._CONT_PREFIX,
                                status_right=self._status_right,
                                search_buf=search_buf,
                                search_direction=self._state.search_direction,
                                search_spans=self._build_search_spans(),
                                selection=self._build_selection(),
                                total_lines=self._buffer.line_count(),
                                command_buf=self._state.command_buf,
                                auto_scroll=self._state.auto_scroll,
                                new_content_below=self._new_content_below,
                                cursor_row=self._state.cursor_row,
                                cursor_col=self._state.cursor_col,
                                widget_lines=self._widget_lines())
        sys.stdout.write(SYNC_END)
        sys.stdout.flush()

    def _on_resize(self, signum, frame):
        ts = shutil.get_terminal_size()
        self._rows = ts.lines
        self._cols = ts.columns
        self._resize_pending = True

    def _handle_resize(self):
        """handle terminal resize: rewrap buffer and full redraw. while an
        overlay owns the screen, just rewrap the buffer/layout - the overlay
        has its own SIGWINCH handler and will repaint itself; drawing the
        conversation view here would punch through the overlay."""
        self._buffer.rewrap(self._cols)
        self._layout.resize(self._rows, self._cols)
        total = self._buffer.line_count()
        content_rows = self._layout.content_rows
        max_offset = max(0, total - content_rows)
        if self._state.auto_scroll:
            self._state.viewport_offset = max(0, total - content_rows)
        else:
            self._state.viewport_offset = min(self._state.viewport_offset, max_offset)

        # clamp cursor to valid range and ensure it's visible
        self._state.cursor_row = max(0, min(self._state.cursor_row, max(0, total - 1)))
        if self._state.cursor_row < self._state.viewport_offset:
            self._state.viewport_offset = self._state.cursor_row
        elif self._state.cursor_row >= self._state.viewport_offset + content_rows:
            self._state.viewport_offset = self._state.cursor_row - content_rows + 1

        if self._focus_stack[-1] != 'main': return
        sys.stdout.write(ERASE_SCREEN)
        self._refresh_all()
        sys.stdout.flush()
