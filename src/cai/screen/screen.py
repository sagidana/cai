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
    KEY_CTRL_C,
)
from .state import Mode, TUIState, _SubmitException, _CommandException
from .buffer import ContentBuffer
from .layout import Layout
from .modes import ModeHandler
from .input import read_key, handle_listener_key, get_overlay_matches
from .overlays.tools import prompt_tools_overlay as _tools_overlay
from .overlays.context import prompt_context_overlay as _ctx_overlay
from .overlays.model import prompt_model_overlay as _model_overlay


class Screen:
    """Alternate-screen TUI for --interactive mode."""

    _PROMPT_PREFIX = '> '
    _CONT_PREFIX   = '  '

    # Style constants — class-level so cli.py can access as Screen._LLM_STYLE
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

        self._cmd_completions: list[str] = []
        self._cmd_overlay_idx: int = -1

        self._closed: bool = False
        self._resize_pending: bool = False

        self._render_lock = threading.RLock()
        self._interrupt_event = threading.Event()
        self._listener_active = False
        self._listener_thread: 'threading.Thread | None' = None

        # Track whether new content arrived while user is scrolled up
        self._new_content_below: bool = False

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

    def write(self, text: str) -> None:
        """Append *text* to the content buffer and refresh the viewport."""
        if not text:
            return
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
                self._new_content_below = False
            else:
                self._new_content_below = True

            # Batch rendering: don't redraw more than ~60fps
            now = time.monotonic()
            if now - self._last_render_time >= 0.016:
                self._refresh_content()
                self._refresh_status()
                sys.stdout.flush()
                self._last_render_time = now
                self._write_pending = False
            else:
                self._write_pending = True

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

    def show_prompt_placeholder(self, msg: str = '> ') -> None:
        """Show a dimmed placeholder in the input area during streaming."""
        with self._render_lock:
            self._layout.render_input(
                [], 0, Mode.NORMAL,
                self._PROMPT_PREFIX, self._CONT_PREFIX, self._cols,
            )
            sys.stdout.flush()

    def set_cmd_completions(self, cmds: list[str]) -> None:
        """Set the command names available for tab/command completion."""
        self._cmd_completions = list(cmds)

    def clear_buffer(self) -> None:
        """Clear the content buffer and refresh."""
        self._buffer.clear()
        self._state.viewport_offset = 0
        self._state.cursor_row = 0
        self._state.auto_scroll = True
        self._new_content_below = False
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
        self._cmd_overlay_idx = -1

        # Enter insert mode, pin to bottom
        self._state.mode = Mode.INSERT
        self._state.auto_scroll = True
        total = self._buffer.line_count()
        content_rows = self._layout.content_rows
        self._state.viewport_offset = max(0, total - content_rows)
        self._new_content_below = False

        self._layout.update_input_height(self._input_buf, self._PROMPT_PREFIX, self._CONT_PREFIX)

        old = termios.tcgetattr(self._tty_fd)
        result = None
        caught_exc = None
        cmd_result = None

        try:
            tty.setraw(self._tty_fd)
            termios.tcflush(self._tty_fd, termios.TCIFLUSH)
            self._refresh_all()
            sys.stdout.write(f'{CUR_SHOW}{CURSOR_BAR}')
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
            # Command mode produced a command — return it as a /command
            return f'/{cmd_result}'

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

    # ── Streaming listener ────────────────────────────────────────────────────

    def start_input_listener(self) -> None:
        """Start a daemon thread that reads interrupt keys during LLM streaming."""
        self._interrupt_event.clear()
        self._listener_active = True
        self._listener_thread = threading.Thread(
            target=self._input_listener_loop, daemon=True, name='cai-input-listener'
        )
        self._listener_thread.start()

    def stop_input_listener(self) -> None:
        """Stop the input listener thread and wait for it to exit."""
        self._listener_active = False
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=0.5)
            self._listener_thread = None

    def _input_listener_loop(self) -> None:
        old = termios.tcgetattr(self._tty_fd)
        try:
            tty.setraw(self._tty_fd)
            while self._listener_active:
                rlist, _, _ = select.select([self._tty_fd], [], [], 0.05)
                if not rlist:
                    continue
                key = read_key(self._tty_fd)
                handle_listener_key(key, self._interrupt_event)
        finally:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, old)

    # ── Refresh helpers ───────────────────────────────────────────────────────

    def _refresh_content(self) -> None:
        """Re-render the content viewport area."""
        search_set = set(self._state.search_matches) if self._state.search_matches else None
        selection = self._build_selection()

        self._layout.render_content(
            self._buffer._lines,
            self._layout.content_rows,
            self._cols,
            cursor_row=self._state.cursor_row,
            selection=selection,
            search_matches=search_set,
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
        )
        sys.stdout.flush()

    def _refresh_input(self) -> None:
        """Re-render the input/prompt area."""
        self._layout.update_input_height(self._input_buf, self._PROMPT_PREFIX, self._CONT_PREFIX)
        overlay_matches = get_overlay_matches(
            ''.join(self._input_buf), self._cmd_completions
        )
        self._layout.render_input(
            self._input_buf, self._cursor_pos, self._state.mode,
            self._PROMPT_PREFIX, self._CONT_PREFIX, self._cols,
            cmd_overlay=overlay_matches,
            overlay_idx=self._cmd_overlay_idx,
            command_buf=self._state.command_buf,
        )
        sys.stdout.flush()

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
        overlay_matches = get_overlay_matches(
            ''.join(self._input_buf), self._cmd_completions
        )
        search_set = set(self._state.search_matches) if self._state.search_matches else None
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
            search_matches=search_set,
            selection=selection,
            total_lines=self._buffer.line_count(),
            cmd_overlay=overlay_matches,
            overlay_idx=self._cmd_overlay_idx,
            command_buf=self._state.command_buf,
            auto_scroll=self._state.auto_scroll,
            new_content_below=self._new_content_below,
            cursor_row=self._state.cursor_row,
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
        # Adjust viewport proportionally
        total = self._buffer.line_count()
        content_rows = self._layout.content_rows
        if self._state.auto_scroll:
            self._state.viewport_offset = max(0, total - content_rows)
        else:
            self._state.viewport_offset = min(
                self._state.viewport_offset,
                max(0, total - content_rows)
            )
        sys.stdout.write(ERASE_SCREEN)
        self._refresh_all()
        sys.stdout.flush()
