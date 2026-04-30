"""Screen — alternate-screen TUI for --interactive mode.

All content is managed in an internal buffer with a scrollable viewport.
The screen uses vim-like modal keybindings:
  Normal  — navigate content (j/k, Ctrl-U/D, gg/G, /search)
  Insert  — edit the prompt (i to enter, Esc to exit)
  Visual  — select text (v/V to enter, y to yank)
  Command — enter slash commands (: to enter)

Layout (top to bottom): content viewport | prompt | status line
"""

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
    ERASE_SCREEN, ERASE_LINE,
    ALT_ENTER, ALT_EXIT,
    cur_move,
)
from .state import Mode, TUIState, _SubmitException, _CommandException
from .buffer import ContentBuffer
from .layout import Layout
from .modes import ModeHandler
from .input import read_key
from .overlays.tools import prompt_tools_overlay as _tools_overlay
from .overlays.context import prompt_context_overlay as _ctx_overlay
from .overlays.model import prompt_model_overlay as _model_overlay
from .overlays.messages import prompt_messages_overlay as _msg_overlay
from .overlays.history import prompt_history_overlay as _history_overlay


class Screen:
    """Alternate-screen TUI for --interactive mode."""

    _PROMPT_PREFIX = '> '
    _CONT_PREFIX   = '  '

    # What's currently being written. Callers pass one of these as `kind`
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

    # Back-compat aliases still referenced by cli.py and tests.
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
        self._rows: int = ts.lines
        self._cols: int = ts.columns

        self._input_buf: list[str] = []
        self._cursor_pos: int = 0
        self._history: list[str] = self._load_history()
        self._history_idx: int = -1
        self._in_prompt: bool = False
        self._current_prompt_msg: str = '> '

        self._status_text: str = ''

        self._command_result: str | None = None

        # Command-mode completions: top-level commands and sub-options
        # {cmd_name: [sub_options]} — empty list means no sub-options
        self._cmd_completions: dict[str, list[str]] = {}

        self._closed: bool = False
        self._resize_pending: bool = False

        self._render_lock = threading.RLock()
        self._interrupt_event = threading.Event()

        # Optional callback invoked on Ctrl-C when the input buffer is empty.
        # Returning True signals that the event was handled (e.g. the LLM was
        # interrupted) and the default double-tap quit logic should be skipped.
        self._interrupt_handler = None

        # Set by the caller (cli.py) while an LLM response is in flight so
        # the TUI can distinguish "interrupt LLM" from "quit" on Ctrl-C.
        self._busy: bool = False

        # Track whether new content arrived while user is scrolled up
        self._new_content_below: bool = False

        # What's currently being written (see Screen.USER/LLM/...).
        # Transitions emit SGR automatically; a fresh buffer segment in
        # the same kind re-emits the style so the color survives \n boundaries.
        self._current_kind: str | None = None

        # Batch rendering for streaming: accumulate writes and flush periodically
        self._write_pending: bool = False
        self._last_render_time: float = 0.0
        _RENDER_INTERVAL = 0.016  # ~60fps

        self._tty_file = open('/dev/tty', 'rb+', buffering=0)
        self._tty_fd = self._tty_file.fileno()
        self._cooked_attrs = termios.tcgetattr(self._tty_fd)

        # Core TUI components
        self._buffer = ContentBuffer(self._cols)
        self._layout = Layout(self._rows, self._cols)
        self._state = TUIState()
        self._modes = ModeHandler()

        # Enter alternate screen
        sys.stdout.write(ALT_ENTER + ERASE_SCREEN + CUR_HIDE)
        sys.stdout.flush()

        signal.signal(signal.SIGWINCH, self._on_resize)

    # ── History persistence ────────────────────────────────────────────────────

    @classmethod
    def _load_history(cls) -> list[str]:
        """Load prompt history from disk. Returns most-recent-first list."""
        try:
            with open(cls._HISTORY_FILE, 'r') as f:
                lines = [l.rstrip('\n') for l in f if l.strip()]
            lines.reverse()
            return lines[:cls._HISTORY_MAX]
        except FileNotFoundError:
            return []
        except OSError:
            return []

    def _save_history_entry(self, entry: str) -> None:
        """Append a single entry to the history file on disk."""
        try:
            os.makedirs(os.path.dirname(self._HISTORY_FILE), exist_ok=True)
            with open(self._HISTORY_FILE, 'a') as f:
                f.write(entry + '\n')
        except OSError:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def write(self, text: str, kind: str | None = None) -> None:
        """Append *text* to the content buffer and refresh the viewport.

        When *kind* is one of the class kind constants (USER, LLM, ...),
        the screen owns styling: the caller writes plain text and the
        screen emits SGR transitions on state change, and re-emits the
        active style at the start of a fresh buffer segment so streamed
        chunks that happen to break on '\\n' keep their color.

        kind=None means the caller manages styling itself (raw ANSI in
        *text*); the screen leaves _current_kind untouched.
        """
        if not text:
            return
        if kind is not None:
            text = self._apply_kind(text, kind)
        with self._render_lock:
            # Handle resize that may have occurred outside the prompt loop
            if self._resize_pending:
                self._resize_pending = False
                self._handle_resize()

            self._buffer.append_text(text)
            total = self._buffer.line_count()
            content_rows = self._layout.content_rows

            if self._state.auto_scroll:
                self._state.viewport_offset = max(0, total - content_rows)
                # Pin the cursor to the tail while following the stream —
                # otherwise G drops cursor_row on the old bottom and each
                # new line pushes the viewport past it, stranding the
                # block cursor above the viewport.
                self._state.cursor_row = max(0, total - 1)
                self._new_content_below = False
            else:
                self._new_content_below = True

            # Batch rendering: don't redraw more than ~60fps
            now = time.monotonic()
            if now - self._last_render_time >= 0.016:
                self._refresh_content()
                self._refresh_status()
                # While the user is actively at the prompt, re-render the
                # input area so their cursor stays parked where they're
                # typing instead of jumping into the content viewport
                # every time the LLM streams another chunk. In NORMAL/VISUAL
                # modes position_cursor parks the block cursor in the
                # content area instead of the prompt.
                if self._in_prompt:
                    self._refresh_input()
                    self._layout.position_cursor(
                        self._state.mode,
                        self._state.cursor_row,
                        self._state.viewport_offset,
                        self._state.cursor_col,
                    )
                sys.stdout.flush()
                self._last_render_time = now
                self._write_pending = False
            else:
                self._write_pending = True

    def _apply_kind(self, text: str, kind: str) -> str:
        """Derive the SGR/newline prefix required to render *text* as *kind*.

        Three things the state transition takes care of:
          1. Kind change mid-line — insert '\\n' so the new state starts
             on a fresh row instead of continuing the previous one.
          2. Kind change — reset the previous style (if any) and open
             the new one.
          3. Same kind but last write ended with '\\n' — wrap_ansi resets
             style tracking per segment, so re-open the style or the new
             segment would render in default color.
        """
        style = self._KIND_STYLES.get(kind, '')
        prev = self._current_kind
        prefix = ''
        if kind != prev:
            if self._buffer._partial:
                prefix += '\n'
            if prev is not None and self._KIND_STYLES.get(prev):
                prefix += SGR_RESET
            if style:
                prefix += style
        elif not self._buffer._partial and style:
            prefix += style
        self._current_kind = kind
        return prefix + text if prefix else text

    def set_status(self, text: str) -> None:
        """Update status bar text and refresh."""
        self._status_text = text
        with self._render_lock:
            if self._resize_pending:
                self._resize_pending = False
                self._handle_resize()
            else:
                self._refresh_status()
            sys.stdout.flush()

    def write_status_hint(self, hint: str) -> None:
        """Temporarily show a hint on the status bar."""
        old = self._status_text
        self._status_text = hint
        with self._render_lock:
            self._refresh_status()
            sys.stdout.flush()
        self._status_text = old

    def set_cmd_completions(self, completions: dict[str, list[str]]) -> None:
        """Set command-mode completions.

        *completions* maps command names to lists of sub-options.
        E.g. {"skill": ["off", "frida", "web"], "compact": [], ...}
        """
        self._cmd_completions = dict(completions)

    def set_interrupt_handler(self, fn) -> None:
        """Register a Ctrl-C handler invoked when the input buffer is empty.

        The handler should return True if it consumed the interrupt (e.g.
        cancelled a running LLM response); in that case the TUI skips its
        double-tap quit logic. Returning False falls through to the normal
        press-again-to-quit flow.
        """
        self._interrupt_handler = fn

    def set_busy(self, busy: bool) -> None:
        """Mark whether an LLM response is currently in flight."""
        self._busy = busy

    def clear_buffer(self) -> None:
        """Clear the content buffer and refresh."""
        self._buffer.clear()
        self._state.viewport_offset = 0
        self._state.cursor_row = 0
        self._state.auto_scroll = True
        self._new_content_below = False
        self._current_kind = None
        with self._render_lock:
            self._refresh_all()
            sys.stdout.flush()

    def close(self) -> None:
        """Exit alternate screen and restore the terminal."""
        if self._closed:
            return
        self._closed = True
        sys.stdout.write(f'{ALT_EXIT}{CUR_SHOW}{CURSOR_RESET}{SGR_RESET}\n')
        sys.stdout.flush()
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        try:
            self._tty_file.close()
        except Exception:
            pass

    def prompt(self, msg: str = '> ') -> str:
        """Collect user input. Blocking. Raises KeyboardInterrupt / EOFError."""
        self._in_prompt = True
        self._current_prompt_msg = msg
        self._input_buf = []
        self._cursor_pos = 0
        self._history_idx = -1
        self._command_result = None

        # Enter insert mode only if the user was following the conversation
        # (auto_scroll still on). Otherwise stay in normal mode so we don't
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

                rlist, _, _ = select.select([self._tty_fd], [], [], 0.05)

                # Flush any pending batched writes
                if self._write_pending:
                    with self._render_lock:
                        self._refresh_content()
                        self._refresh_status()
                        self._refresh_input()
                        self._layout.position_cursor(
                            self._state.mode,
                            self._state.cursor_row,
                            self._state.viewport_offset,
                            self._state.cursor_col,
                        )
                        sys.stdout.flush()
                        self._write_pending = False
                        self._last_render_time = time.monotonic()

                if not rlist:
                    continue

                key = read_key(self._tty_fd)
                try:
                    self._modes.handle_key(key, self._state, self)
                except _SubmitException as e:
                    result = e.value
                    break
                except _CommandException as e:
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
            # Command mode produced a command — store it for the caller
            self._command_result = cmd_result
            return ''

        # Clear the input area so the next prompt() call starts clean.
        # The caller is responsible for echoing the submitted text into
        # the content buffer at the right time — when prompts are queued
        # the echo should happen when the LLM actually starts processing
        # it, not at submit time.
        if result is not None and result.strip():
            self._input_buf = []
            self._cursor_pos = 0
            with self._render_lock:
                self._layout.render_input(
                    [], 0, Mode.NORMAL,
                    self._PROMPT_PREFIX, self._CONT_PREFIX, self._cols,
                )

        return result  # type: ignore[return-value]

    # ── Overlay entry points ──────────────────────────────────────────────────

    def _restore_after_overlay(self) -> None:
        """Restore the main TUI after an overlay exits.

        Syncs dimensions (the overlay may have handled resize),
        re-enters alternate screen, rewraps content, and redraws.
        """
        # The overlay's resize handler updated self._rows/self._cols
        self._layout.resize(self._rows, self._cols)
        self._buffer.rewrap(self._cols)
        total = self._buffer.line_count()
        content_rows = self._layout.content_rows
        if self._state.auto_scroll:
            self._state.viewport_offset = max(0, total - content_rows)
        sys.stdout.write(ALT_ENTER + ERASE_SCREEN)
        self._refresh_all()
        sys.stdout.flush()

    def prompt_tools_overlay(self, tool_entries: list, enabled: set) -> set:
        """Interactive tools toggle overlay. See overlays/tools.py for docs."""
        result = _tools_overlay(self, tool_entries, enabled)
        self._restore_after_overlay()
        return result

    def prompt_context_overlay(
        self, messages: list, context_size: int = 0, prompt_tokens: int = 0
    ) -> tuple:
        """Interactive context viewer/editor. See overlays/context.py for docs."""
        result = _ctx_overlay(self, messages, context_size, prompt_tokens)
        self._restore_after_overlay()
        return result

    def prompt_model_overlay(self, models: list) -> 'str | None':
        """Interactive model picker. Returns selected model name or None."""
        result = _model_overlay(self, models)
        self._restore_after_overlay()
        return result

    def prompt_messages_overlay(
        self, messages: list, tracker, *,
        model: str = '', context_size: int = 0, prompt_tokens: int = 0,
    ) -> tuple:
        """Interactive messages overlay. See overlays/messages.py for docs."""
        result = _msg_overlay(self, messages, tracker,
                              model=model,
                              context_size=context_size,
                              prompt_tokens=prompt_tokens)
        self._restore_after_overlay()
        return result

    def prompt_history_overlay(self, tracker, *, context_size: int = 0) -> bool:
        """Interactive undo-tree viewer. See overlays/history.py for docs.

        Returns True if the user pressed F to fork at a node.
        """
        result = _history_overlay(self, tracker, context_size=context_size)
        self._restore_after_overlay()
        return bool(result)

    # ── Refresh helpers ───────────────────────────────────────────────────────

    def _refresh_content(self) -> None:
        """Re-render the content viewport area."""
        search_spans = self._build_search_spans()
        selection = self._build_selection()

        self._layout.render_content(
            self._buffer._lines,
            self._layout.content_rows,
            self._cols,
            cursor_row=self._state.cursor_row,
            selection=selection,
            search_spans=search_spans,
            viewport_offset=self._state.viewport_offset,
        )

    def _refresh_status(self) -> None:
        """Re-render the status line."""
        self._layout.render_status(
            self._state.mode,
            self._status_text,
            search_buf=self._state.search_buf if self._state.mode == Mode.SEARCH else None,
            search_direction=self._state.search_direction,
            viewport_offset=self._state.viewport_offset,
            total_lines=self._buffer.line_count(),
            cols=self._cols,
            auto_scroll=self._state.auto_scroll,
            new_content_below=self._new_content_below,
            command_buf=self._state.command_buf if self._state.mode == Mode.COMMAND else None,
            cursor_row=self._state.cursor_row,
        )
        sys.stdout.flush()

    def _refresh_input(self) -> None:
        """Re-render the input/prompt area."""
        prev_height = self._layout.input_height
        self._layout.update_input_height(self._input_buf, self._PROMPT_PREFIX, self._CONT_PREFIX)
        if self._layout.input_height != prev_height:
            # Input area resized — the content viewport grew or shrank.
            # Rebalance viewport_offset so the buffer stays anchored, and
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
        self._layout.render_input(
            self._input_buf, self._cursor_pos, self._state.mode,
            self._PROMPT_PREFIX, self._CONT_PREFIX, self._cols,
            command_buf=self._state.command_buf,
        )
        sys.stdout.flush()

    def _build_search_spans(self) -> 'dict[int, list[tuple[int, int]]] | None':
        """Group the flat match list into per-line (start, end) spans."""
        if not self._state.search_matches:
            return None
        spans: dict[int, list[tuple[int, int]]] = {}
        for row, s, e in self._state.search_matches:
            spans.setdefault(row, []).append((s, e))
        return spans

    def _build_selection(self):
        """Build the selection tuple for visual mode, or None."""
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

    def _refresh_all(self) -> None:
        """Full screen redraw."""
        self._layout.update_input_height(self._input_buf, self._PROMPT_PREFIX, self._CONT_PREFIX)
        search_spans = self._build_search_spans()
        selection = self._build_selection()
        self._layout.render_all(
            self._buffer._lines,
            self._state.viewport_offset,
            self._state.mode,
            self._status_text,
            self._input_buf,
            self._cursor_pos,
            self._PROMPT_PREFIX,
            self._CONT_PREFIX,
            search_buf=self._state.search_buf if self._state.mode == Mode.SEARCH else None,
            search_direction=self._state.search_direction,
            search_spans=search_spans,
            selection=selection,
            total_lines=self._buffer.line_count(),
            command_buf=self._state.command_buf,
            auto_scroll=self._state.auto_scroll,
            new_content_below=self._new_content_below,
            cursor_row=self._state.cursor_row,
            cursor_col=self._state.cursor_col,
        )
        sys.stdout.flush()

    # ── Resize ────────────────────────────────────────────────────────────────

    def _on_resize(self, signum, frame) -> None:
        ts = shutil.get_terminal_size()
        self._rows, self._cols = ts.lines, ts.columns
        self._resize_pending = True

    def _handle_resize(self) -> None:
        """Handle terminal resize: rewrap buffer and full redraw."""
        self._buffer.rewrap(self._cols)
        self._layout.resize(self._rows, self._cols)
        total = self._buffer.line_count()
        content_rows = self._layout.content_rows
        max_offset = max(0, total - content_rows)
        if self._state.auto_scroll:
            self._state.viewport_offset = max(0, total - content_rows)
        else:
            self._state.viewport_offset = min(self._state.viewport_offset, max_offset)
        # Clamp cursor to valid range and ensure it's visible
        self._state.cursor_row = max(0, min(self._state.cursor_row, max(0, total - 1)))
        if self._state.cursor_row < self._state.viewport_offset:
            self._state.viewport_offset = self._state.cursor_row
        elif self._state.cursor_row >= self._state.viewport_offset + content_rows:
            self._state.viewport_offset = self._state.cursor_row - content_rows + 1
        sys.stdout.write(ERASE_SCREEN)
        self._refresh_all()
        sys.stdout.flush()
