name: python
tools: python
---
# Skill: Python

`python(code="...", timeout=60)` runs a snippet and returns its stdout/stderr.
Use it when a task is easier as a few lines of code than as a chain of tool
calls. `print()` what you want back.

## Rules

- One-shot: fresh interpreter every call, no variables survive between calls.
- Standard library only. No network. `input()` sees EOF.
- Read/list the working dir and scratch (`os.environ['CAI_SCRATCH']`);
  every other path does not exist.
- Writes land under scratch ONLY - everywhere else is read-only. Use scratch to
  pass state between calls. Writes elsewhere, `subprocess`, `ctypes`, `cffi`
  raise `PermissionError`. To change a project file, `tool_call()` a write tool.

## tool_call() — drive your other tools

`tool_call(name, **kwargs) -> str` runs one of your selected tools (by its exact
name, not `python`) and returns its result. Reduce big results in Python; only
what you `print()` reaches you:

```python
text = tool_call("fs__read_file", file_path="big.log")
print(text.count("ERROR"))
```

Arguments are the SAME JSON a direct MCP tool call takes — text only, NEVER
Python `bytes`. Encode binary as text first:

```python
tool_call("fs__create_file", file_path="C", content=data.hex(), encoding="hex")   # right
tool_call("fs__create_file", file_path="C", content=data)                         # WRONG: bytes
```

Do writes inside the call — never `print()` a payload and re-type it into a
write tool. To write 1638 bytes of `0x36`:

```python
print(tool_call("fs__create_file", file_path="C", content="36" * 1638, encoding="hex"))
```

`fs__create_file` takes `encoding="hex"`, so build any binary in Python and
hand off `bytes.hex()` — a PNG, WAV, zip, fixed header.
