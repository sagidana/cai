"""watch: run a one-shot agent each time a byte stream settles.

`cai --watch` turns the CLI into a trigger on stdin: while bytes keep coming
nothing happens; once the stream has been quiet for --watch-settle-after
seconds,
a one-shot run is spawned over the tail of the stream (the last --watch-window
bytes) plus the usual prompt and flags. up to --watch-max-concurrents runs
(default 1) may be in flight at once; spawning past that limit kills the
oldest run first - its input is the stalest. EOF (the feeding process exited)
gives any unprocessed data one final run and exits with the newest run's
status once every run in flight has been waited out.

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
    """interrupt an in-flight run (the concurrency limit needs its slot) and
    wait it out."""
    if worker["thread"].is_alive():
        worker["run"].interrupt.set()
        _note("[watch] limit reached - oldest run killed")
    worker["thread"].join()


def _reap(workers):
    """drop workers whose runs finished on their own; in-flight ones are
    kept, oldest first."""
    alive = []
    for worker in workers:
        if worker["thread"].is_alive():
            alive.append(worker)
            continue
        worker["thread"].join()
    return alive


def _spawn_capped(workers, max_concurrents, make_run, drive, text):
    """spawn a run, first killing oldest workers until the limit has room."""
    while len(workers) >= max_concurrents:
        _kill(workers[0])
        del workers[0]
    workers.append(_spawn(make_run, drive, text))


def run(make_run, drive, *, settle_after=2.0, window=65536, max_concurrents=1,
        stdin_fd=None):
    """the settle loop: watch stdin_fd, fire drive(make_run(text)) on each
    settle, at most max_concurrents runs in flight - spawning past the limit
    kills the oldest run. blocking; returns the process exit code - the
    newest run's, or 0."""
    if stdin_fd is None:
        stdin_fd = sys.stdin.fileno()
    buf = bytearray()
    pending = False           # bytes arrived since the last spawned run
    workers = []              # in-flight runs, oldest first
    try:
        while True:
            readable, _, _ = select.select([stdin_fd], [], [], settle_after)
            if not readable:
                # quiet for a full settle-after: reap naturally-finished runs,
                # then fire on unprocessed data. runs still in flight keep
                # running - only spawning past the limit kills one.
                workers = _reap(workers)
                if pending:
                    pending = False
                    _spawn_capped(workers, max_concurrents, make_run, drive,
                                  buf.decode("utf-8", errors="replace"))
                continue
            chunk = os.read(stdin_fd, 65536)
            if not chunk:
                break                                        # EOF
            buf += chunk
            if len(buf) > window:
                del buf[:len(buf) - window]
            pending = True
    except KeyboardInterrupt:
        for worker in workers:
            _kill(worker)
        return 130
    # EOF. any unprocessed data gets one final run - a worker like any other,
    # subject to the limit - then every run in flight is waited out; the exit
    # code is the newest run's. no reap here: a finished worker may still own
    # the exit code.
    if pending:
        _note(f"[watch] stream ended - final run over {len(buf)} bytes")
        _spawn_capped(workers, max_concurrents, make_run, drive,
                      buf.decode("utf-8", errors="replace"))
    for worker in workers:
        worker["thread"].join()
    if workers:
        return workers[-1]["box"].get("code", 0)
    return 0
