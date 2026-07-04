name: fs-read-only
tools: fs__search, fs__read_file, fs__list_files
---
# Skill: File System (Read-Only)

Inspect files only — nothing here can create, edit, move, or delete. If a task needs changes, ask to switch to the `fs` skill instead of attempting them.

Workflow:
- `fs__list_files` to orient in an unfamiliar tree.
- `fs__search` to locate symbols, strings, or patterns — prefer specific patterns over broad ones.
- `fs__read_file` with `line_start`/`line_end` for targeted ranges — avoid loading large files whole.

All paths must stay inside the working directory. Cite every finding as `path/to/file.py:42`.
