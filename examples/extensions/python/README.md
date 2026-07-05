# python — an example cai extension

Run Python: **`python__run(code)`** executes a snippet in a fresh subprocess
of cai's own interpreter, inside the same sandbox the fs tools live in —
writes are confined to the working directory plus the session scratch dir
(`$CAI_SCRATCH`), and the easy ways out of a python-level jail (subprocesses,
sockets, ctypes) are blocked.

## Layout

```
python/
├── README.md
└── mcps/
    └── python.py      # FastMCP stdio server → tool python__run
```

A cai extension is a self-contained bundle. Each file under `mcps/` is
spawned as an MCP stdio server; its tools are surfaced prefixed with the
file name, so `run` becomes `python__run`.

## Install

```sh
cp -r examples/extensions/python ~/.config/cai/extensions/
```

## How it works

The tool spawns `sys.executable -c <bootstrap>` and pipes the snippet in over
stdin. Before executing it, the bootstrap installs a `sys.addaudithook` jail —
audit hooks cannot be removed once installed, so the snippet runs entirely
inside it. The jail mirrors `cai.safe_path`:

- **writes** (open for write, remove, rename, mkdir, …) — allowed only under
  the working directory and `$CAI_SCRATCH`, exactly the fs-tool jail.
- **reads** — additionally allow the interpreter's own install prefixes,
  so importing stdlib and site-packages keeps working.
- **subprocess / exec / fork, sockets, ctypes** — blocked outright; each is a
  one-liner around a python-level jail otherwise.

A denied operation raises `PermissionError` inside the snippet and the
traceback comes back as the tool result, so the model sees exactly what was
blocked and can route the work through scratch instead.

## Notes

- **A guardrail, not a hard security boundary** — the same posture as
  `safe_path` itself. A hostile snippet with a compiled escape hatch already
  installed in site-packages (e.g. cffi) could slip a python-level jail; for
  untrusted use, run cai itself inside a container.
- Each call is one-shot: variables don't survive between runs — state carries
  via files in the working directory or `$CAI_SCRATCH`
  (`os.environ['CAI_SCRATCH']` from inside a snippet).
- The snippet arrives over stdin, so `input()` sees EOF.
- Runs are killed after `timeout` seconds (default 60); long output is
  truncated to its head and tail.
