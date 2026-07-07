"""watch: run a one-shot agent each time a byte stream settles.

`cai --watch` turns the CLI into a trigger on stdin: while bytes keep coming
nothing happens; once the stream has been quiet for --watch-threshold seconds,
a one-shot run is spawned over the tail of the stream (the last --watch-window
bytes) plus the usual prompt and flags. bytes arriving while that run is in
flight kill it at once - its input is stale - and the wait starts over, the
new bytes still inside the window for the next settle. EOF (the feeding
process exited) gives any unprocessed data one final run, which nothing can
kill anymore, and exits with its status.

the CLI owns flag parsing and run construction; run() here takes a
make_run(text) factory and a drive(run) consumer, so this module is only the
settle loop - testable with a pipe and fakes, no LLM anywhere. the contract
with a spawned run: run.interrupt is the kill switch (a threading.Event the
agentic loop checks at its boundaries) and run.close() the teardown, called
once per run on its worker thread however the run ends."""
from __future__ import annotations

import os
import select
import sys
import threading


_DIM = "\033[2m"
_RESET = "\033[0m"


def _note(text):
    """a dim progress line to stderr, only when a human is watching there -
    the agent's own output owns stdout."""
    if not sys.stderr.isatty():
        return
    sys.stderr.write(_DIM + text + _RESET + "\n")
    sys.stderr.flush()


def _spawn(make_run, drive, text):
    """start one run over `text` on a worker thread; returns the worker handle
    the loop tracks - its thread, its run (for the kill switch), and the box
    drive's exit code lands in."""
    run = make_run(text)
    _note(f"[watch] settled - running over {len(text)} chars")

    box = {}
    def _target():
        try:
            box["code"] = drive(run)
        finally:
            run.close()

    thread = threading.Thread(target=_target, daemon=True, name="cai-watch-run")
    worker = {}
    worker["thread"] = thread
    worker["run"] = run
    worker["box"] = box
    thread.start()
    return worker


def _kill(worker):
    """interrupt an in-flight run (new bytes made its input stale) and wait it
    out. returns None: the worker slot is free."""
    if worker is None:
        return None
    if worker["thread"].is_alive():
        worker["run"].interrupt.set()
        _note("[watch] new data - run killed")
    worker["thread"].join()
    return None


def _reap(worker):
    """drop a worker whose run finished on its own; an in-flight one is kept."""
    if worker is None:
        return None
    if worker["thread"].is_alive():
        return worker
    worker["thread"].join()
    return None


def run(make_run, drive, *, threshold=2.0, window=65536, stdin_fd=None):
    """the settle loop: watch stdin_fd, fire drive(make_run(text)) on each
    settle, kill an in-flight run the moment new bytes land. blocking; returns
    the process exit code - the final EOF run's, or 0."""
    if stdin_fd is None:
        stdin_fd = sys.stdin.fileno()
    buf = bytearray()
    pending = False           # bytes arrived since the last completed run
    worker = None
    try:
        while True:
            readable, _, _ = select.select([stdin_fd], [], [], threshold)
            if not readable:
                # quiet for a full threshold: reap a naturally-finished run,
                # then fire on unprocessed data. a run still in flight just
                # keeps running - only new bytes kill it.
                worker = _reap(worker)
                if pending and worker is None:
                    pending = False
                    worker = _spawn(make_run, drive, buf.decode("utf-8", errors="replace"))
                continue
            chunk = os.read(stdin_fd, 65536)
            if not chunk:
                break                                        # EOF
            buf += chunk
            if len(buf) > window:
                del buf[:len(buf) - window]
            pending = True
            worker = _kill(worker)
    except KeyboardInterrupt:
        _kill(worker)
        return 130
    # EOF. a run in flight already covers the latest data (newer bytes would
    # have killed it): wait for it and exit with its status. otherwise any
    # unprocessed data gets one final run, on this thread - nothing can kill
    # it now.
    if worker is not None:
        worker["thread"].join()
        return worker["box"].get("code", 0)
    if pending:
        run_ = make_run(buf.decode("utf-8", errors="replace"))
        _note(f"[watch] stream ended - final run over {len(buf)} bytes")
        try:
            return drive(run_)
        finally:
            run_.close()
    return 0
