"""lines: run a one-shot agent over every line of an input, N at a time.

`cai --line-by-line` maps the prompt over its input: each non-blank line of
--file (or piped stdin) gets its own one-shot run - the line as context, the
prompt as the task, every other flag applying as usual - and the answers
print to stdout in input order, each prefixed by the input line that produced
it (tab-separated) so every output carries its own reference back to the
source. --cores bounds how many runs are in flight at once (worker
threads: a turn is IO-bound, and every run owns its agent, tools and MCP
servers - the isolation sub-agents already rely on).

the input is consumed as a stream, not slurped: a reader thread pulls lines
off the source as they arrive and queues them, so runs start while the
producer is still writing - `tail -f access.log | cai --line-by-line ...`
begins working on the first line, not at EOF. the answers still print in
input order (early finishers wait in the reorder buffer).

the CLI owns flag parsing, input opening and run construction; run() here is
only the scheduler - a reader, a worker pool, and the in-order drain -
testable with fakes, no LLM anywhere. the contract with a spawned run
matches watch.py's: iterate it for its events (discarded - parallel streams
would interleave), read .text, run.interrupt kills it, run.close() tears it
down, once, on its worker."""
from __future__ import annotations

import io
import queue
import sys
import threading

from cai.events import EventType
from cai.tail import _Printer


_DIM = "\033[2m"
_RESET = "\033[0m"


def _note(text):
    """a dim progress line to stderr, only when a human is watching there -
    the answers own stdout."""
    if not sys.stderr.isatty():
        return
    sys.stderr.write(_DIM + text + _RESET + "\n")
    sys.stderr.flush()


def _emit(index, ok, answer, line, trace):
    """print one line's answer to stdout, prefixed by the input line that
    produced it (tab-separated, so the output cuts/joins back onto its
    source). the run's trace - its tool calls, and its reasoning when the
    settings allow - goes to stderr with the progress note: buffered during
    the run and flushed here as one block, so parallel lines never
    interleave."""
    status = "ok"
    if not ok:
        status = "failed"
    _note(f"[{index + 1}] {status}")
    if trace:
        _note(trace.rstrip("\n"))
    if answer is None:
        answer = ""
    if not answer.endswith("\n"):
        answer = answer + "\n"
    sys.stdout.write(line + "\t" + answer)
    sys.stdout.flush()


def run(make_run, line_source, cores=1, show_reasoning=True):
    """the scheduler: run make_run(line) for every line of `line_source` (any
    iterable - a generator over live stdin included), `cores` at a time,
    printing each answer to stdout in input order. blocking; returns the exit
    code - 0 when every line succeeded, 1 when any failed, 130 on Ctrl-C."""
    from cai.api import ApiError
    from cai.llm import LLMError

    cores = max(1, cores)
    todo = queue.Queue()                # (index, line), then one None per worker
    results = {}                        # index -> (ok, answer, line, trace)
    done = threading.Condition()        # guards results and state
    state = {}
    state["total"] = None               # set once the source is exhausted
    stop = threading.Event()
    active = []                         # in-flight runs, for Ctrl-C
    active_lock = threading.Lock()

    def _read():
        """pull lines off the source as they arrive; each is work at once."""
        count = 0
        for line in line_source:
            if stop.is_set():
                break
            todo.put((count, line))
            count += 1
        with done:
            state["total"] = count
            done.notify_all()
        for _ in range(cores):
            todo.put(None)

    def _drive_line(line):
        """drive one line's run to completion; (ok, answer, trace). the run's
        events render into a per-line buffer (its tool calls, plus reasoning
        when the settings allow - never its content, which IS the answer) so
        the trace can flush grouped instead of interleaving with other lines.
        a run that fails becomes a per-line 'Error:' answer - one bad line
        must not take the batch (or its worker, leaving the drain hanging)
        down."""
        try:
            run_ = make_run(line)
        except Exception as e:
            return False, f"Error: {type(e).__name__}: {e}", ""
        with active_lock:
            active.append(run_)
        if stop.is_set():
            run_.interrupt.set()        # scheduled after Ctrl-C: wind down now
        printer = _Printer(io.StringIO(), show_reasoning=show_reasoning)
        try:
            for event in run_:
                if event.type == EventType.CONTENT: continue
                if event.type == EventType.USER: continue
                printer.event(event)
            return True, run_.text, printer.out.getvalue()
        except (ApiError, LLMError) as e:
            return False, f"Error: {e}", printer.out.getvalue()
        except Exception as e:
            return False, f"Error: {type(e).__name__}: {e}", printer.out.getvalue()
        finally:
            with active_lock:
                active.remove(run_)
            run_.close()

    def _worker():
        while not stop.is_set():
            item = todo.get()
            if item is None:
                return
            index, line = item
            ok, answer, trace = _drive_line(line)
            with done:
                results[index] = (ok, answer, line, trace)
                done.notify_all()

    reader = threading.Thread(target=_read, daemon=True, name="cai-line-reader")
    reader.start()
    workers = []
    for _ in range(cores):
        thread = threading.Thread(target=_worker, daemon=True, name="cai-line-run")
        workers.append(thread)
        thread.start()

    failed = False
    emitted = 0
    try:
        while True:
            with done:
                while emitted not in results:
                    if state["total"] is not None and emitted >= state["total"]:
                        break
                    done.wait()
                finished = emitted not in results
                if not finished:
                    ok, answer, line, trace = results.pop(emitted)
            if finished:
                break
            if not ok:
                failed = True
            _emit(emitted, ok, answer, line, trace)
            emitted += 1
    except KeyboardInterrupt:
        stop.set()
        with active_lock:
            for run_ in active:
                run_.interrupt.set()
        for _ in workers:               # wake any worker parked on the queue
            todo.put(None)
        for thread in workers:
            thread.join()
        # the reader may sit in a blocking read on the source; it is a daemon
        # and dies with the process rather than being joined.
        return 130
    for thread in workers:
        thread.join()
    if failed:
        return 1
    return 0
