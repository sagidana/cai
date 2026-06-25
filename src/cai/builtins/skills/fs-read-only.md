name: fs-read-only
tools: fs__search, fs__read_file, fs__list_files
---
# Skill: File System (Read-Only)

This is a read-only skill. It exposes only tools that inspect the file system - it cannot create, edit, move, or delete anything. If a task requires modifying files, ask to switch to the `fs` skill rather than attempting changes here.

Workflow:
- Use `fs__list_files` to orient yourself in an unfamiliar directory tree before reading individual files.
- Use `fs__search` to locate symbols, strings, or patterns across the tree before opening files.
- Use `fs__read_file` with `line_start`/`line_end` to read targeted ranges - avoid loading large files in full.

Discipline:
- Always use absolute paths.
- When searching, prefer specific patterns over broad globs - narrow results are faster to reason about.
- Cite every finding with file path and line number: `path/to/file.py:42`
