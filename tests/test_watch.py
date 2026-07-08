"""Tests for cai.watch - the `cai --watch` settle loop.

Pure loop mechanics: stdin is a real os.pipe a feeder thread writes to, the
runs are fakes carrying only the contract watch.run relies on (an interrupt
Event and close()), and drive is a recording function. No LLM anywhere."""
import os
import threading
import time

import pytest

from cai import watch


SETTLE_AFTER = 0.05


class FakeRun:
    def __init__(self, data):
        self.data = data
        self.interrupt = threading.Event()
        self.closed = False

    def close(self):
        self.closed = True


def _make_run(runs):
    def make_run(text):
        run = FakeRun(text)
        runs.append(run)
        return run
    return make_run


def _wait_for(predicate, message, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail(f"timed out waiting for {message}")


def _pipe():
    r, w = os.pipe()
    return r, w


def test_settle_triggers_one_run_over_the_buffer():
    r, w = _pipe()
    runs = []
    log = []
    def drive(run):
        log.append(run.data)
        return 0
    def feeder():
        os.write(w, b"abc")
        _wait_for(lambda: log, "the settled run")
        os.close(w)
    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    code = watch.run(_make_run(runs), drive, settle_after=SETTLE_AFTER, window=100, stdin_fd=r)

    feeder_thread.join(timeout=5)
    os.close(r)
    assert code == 0
    assert log == ["abc"]                  # one run; no new data, no final run
    assert runs[0].closed


def test_new_data_kills_the_inflight_run_and_retriggers():
    r, w = _pipe()
    runs = []
    log = []
    def drive(run):
        log.append(("start", run.data))
        if run.data == "a":
            run.interrupt.wait(5)          # hang until killed
        log.append(("end", run.data, run.interrupt.is_set()))
        return 0
    def feeder():
        os.write(w, b"a")
        _wait_for(lambda: ("start", "a") in log, "the first run")
        os.write(w, b"b")                  # lands mid-run: the retrigger kills it
        _wait_for(lambda: ("start", "ab") in log, "the retriggered run")
        os.close(w)
    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    code = watch.run(_make_run(runs), drive, settle_after=SETTLE_AFTER, window=100, stdin_fd=r)

    feeder_thread.join(timeout=5)
    os.close(r)
    assert code == 0
    assert ("end", "a", True) in log       # killed, interrupt seen
    assert ("end", "ab", False) in log     # the retrigger ran out its course
    assert runs[0].closed
    assert runs[1].closed


def test_window_slides_over_the_stream():
    r, w = _pipe()
    runs = []
    log = []
    def drive(run):
        log.append(run.data)
        return 0
    def feeder():
        os.write(w, b"abcdefgh")
        _wait_for(lambda: log, "the settled run")
        os.close(w)
    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    watch.run(_make_run(runs), drive, settle_after=SETTLE_AFTER, window=4, stdin_fd=r)

    feeder_thread.join(timeout=5)
    os.close(r)
    assert log == ["efgh"]                 # only the last window bytes


def test_a_silent_stream_does_not_retrigger():
    r, w = _pipe()
    runs = []
    log = []
    def drive(run):
        log.append(run.data)
        return 0
    def feeder():
        os.write(w, b"x")
        _wait_for(lambda: log, "the settled run")
        time.sleep(SETTLE_AFTER * 6)       # several quiet settle windows
        os.close(w)
    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    code = watch.run(_make_run(runs), drive, settle_after=SETTLE_AFTER, window=100, stdin_fd=r)

    feeder_thread.join(timeout=5)
    os.close(r)
    assert code == 0
    assert log == ["x"]                    # once, not once per quiet window


def test_eof_gives_unprocessed_data_a_final_run_and_its_exit_code():
    r, w = _pipe()
    runs = []
    log = []
    def drive(run):
        log.append(run.data)
        return 7
    os.write(w, b"tail")
    os.close(w)                            # EOF before any settle

    code = watch.run(_make_run(runs), drive, settle_after=SETTLE_AFTER, window=100, stdin_fd=r)

    os.close(r)
    assert code == 7
    assert log == ["tail"]
    assert runs[0].closed


def test_eof_with_nothing_pending_exits_clean():
    r, w = _pipe()
    runs = []
    os.close(w)                            # empty stream

    code = watch.run(_make_run(runs), lambda run: 0, settle_after=SETTLE_AFTER, window=100, stdin_fd=r)

    os.close(r)
    assert code == 0
    assert runs == []


def test_max_concurrents_runs_in_parallel_and_kills_the_oldest_at_the_limit():
    r, w = _pipe()
    runs = []
    log = []
    release = threading.Event()
    def drive(run):
        log.append(("start", run.data))
        while not release.is_set() and not run.interrupt.is_set():
            time.sleep(0.005)
        log.append(("end", run.data, run.interrupt.is_set()))
        return len(run.data)
    def feeder():
        os.write(w, b"a")
        _wait_for(lambda: ("start", "a") in log, "the first run")
        os.write(w, b"b")
        _wait_for(lambda: ("start", "ab") in log, "the second run")
        os.write(w, b"c")
        _wait_for(lambda: ("start", "abc") in log, "the third run")
        release.set()
        os.close(w)
    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    code = watch.run(_make_run(runs), drive, settle_after=SETTLE_AFTER, window=100,
                     max_concurrents=2, stdin_fd=r)

    feeder_thread.join(timeout=5)
    os.close(r)
    assert code == 3                                     # the newest run's code
    assert log.index(("start", "ab")) < log.index(("end", "a", True))
    assert log.index(("end", "a", True)) < log.index(("start", "abc"))
    assert ("end", "ab", False) in log                   # never killed
    assert ("end", "abc", False) in log
    assert runs[0].closed
    assert runs[1].closed
    assert runs[2].closed


def test_eof_final_run_kills_the_oldest_when_at_the_limit():
    r, w = _pipe()
    runs = []
    log = []
    def drive(run):
        log.append(("start", run.data))
        if run.data == "a":
            run.interrupt.wait(5)          # hang until killed
        log.append(("end", run.data, run.interrupt.is_set()))
        return 9
    def feeder():
        os.write(w, b"a")
        _wait_for(lambda: ("start", "a") in log, "the first run")
        os.write(w, b"b")
        os.close(w)                        # EOF while the first run hangs
    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    code = watch.run(_make_run(runs), drive, settle_after=SETTLE_AFTER, window=100, stdin_fd=r)

    feeder_thread.join(timeout=5)
    os.close(r)
    assert code == 9
    assert ("end", "a", True) in log       # killed to make room
    assert ("end", "ab", False) in log     # the final run, never killed
    assert runs[0].closed
    assert runs[1].closed


def test_eof_waits_out_an_inflight_run_and_returns_its_code():
    r, w = _pipe()
    runs = []
    log = []
    release = threading.Event()
    def drive(run):
        log.append(("start", run.data))
        release.wait(5)                    # still running when EOF lands
        log.append(("end", run.data))
        return 3
    def feeder():
        os.write(w, b"data")
        _wait_for(lambda: ("start", "data") in log, "the settled run")
        os.close(w)                        # EOF mid-run
        release.set()
    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    code = watch.run(_make_run(runs), drive, settle_after=SETTLE_AFTER, window=100, stdin_fd=r)

    feeder_thread.join(timeout=5)
    os.close(r)
    assert code == 3
    assert ("end", "data") in log          # waited out, never killed
    assert len(runs) == 1
    assert runs[0].closed
