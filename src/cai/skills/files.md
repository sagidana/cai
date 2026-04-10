tools: search, read_file, list_files
---
## Skill: File System (Read-Only)

This is a read-only skill. Do not create, edit, move, or delete files unless explicitly asked.

Workflow:
- Use `list_files` to orient yourself in an unfamiliar directory tree before reading individual files.
- Use `search` to locate symbols, strings, or patterns across the tree before opening files.
- Use `read_file` with `line_start`/`line_end` to read targeted ranges — avoid loading large files in full.

Discipline:
- Always use absolute paths.
- When searching, prefer specific patterns over broad globs — narrow results are faster to reason about.
- Cite every finding with file path and line number: `path/to/file.py:42`
