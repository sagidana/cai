name: python
tools: python
---
# Skill: Python

Run Python to compute, transform data, or orchestrate your other tools from a
script — reach for it when a task is awkward as a chain of tool calls but easy as
a few lines of code.

`python(code="...", timeout=60)` runs the snippet in a fresh interpreter and
returns its combined stdout/stderr. Each call is one-shot — variables do NOT
survive between calls; persist state via files. `print()` what you want to see.

## Calling your other tools

The snippet gets a `call(name, **kwargs) -> str` builtin that runs one of your
own currently-available tools and returns its result as a string:

```python
text = call("fs__read_file", file_path="big.log")
print(text.count("ERROR"))
```

The tool call runs inside cai (with its normal confinement and gates); only what
you `print()` becomes the result. So read a large result, reduce it in Python,
and return just the answer — the intermediate data never enters the conversation.
`call` reaches the tools you already have selected, by their exact schema names
(not `python` itself). Use it for map/reduce over files, filtering a big
listing, or fanning out to sub-agents (`call("launch_agent", ...)`).

## Sandbox

- Writing files is confined to the working directory and the session scratch dir
  — address scratch as `os.environ['CAI_SCRATCH']`.
- `subprocess`, network, `ctypes` and `cffi` are blocked; a denied op raises
  `PermissionError` (its traceback comes back as the result).
- The interpreter is a managed virtualenv with the **standard library only** — no
  third-party packages. `input()` sees EOF.
