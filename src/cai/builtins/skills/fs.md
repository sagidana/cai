name: fs
skills: fs-read-only
tools: fs__create_file, fs__edit_file, fs__rename_file, fs__move_file, fs__copy_file, fs__copy_bytes, fs__remove_file, fs__create_directory, fs__move_directory
---
# Skill: File System (Read-Write)

Adds the tools that modify the tree to `fs-read-only`. Always inspect with the read-only tools before you mutate — never edit blind.

Modifying tools:
- `fs__create_file` — write a new file (parents created automatically).
- `fs__edit_file` — replace `old_text` with `new_text`; quote enough context to make the match unique.
- `fs__rename_file` / `fs__move_file` — rename or relocate a file (`fs__move_file` accepts a destination directory).
- `fs__copy_file` — copy a file (accepts a destination directory).
- `fs__copy_bytes` — copy a byte range between files without the bytes passing through you: extract, append, or patch in place.
- `fs__remove_file` — delete a regular file (never a directory).
- `fs__create_directory` — `mkdir -p`.
- `fs__move_directory` — rename or relocate a directory.

Scratch directory:
- Use `$CAI_SCRATCH` for intermediates (bulky outputs, binaries, working files) — it expands in any path and in `bash`. It is deleted at session end, so move anything worth keeping into the project tree.
- Prefer it over the working directory for anything the user did not ask to have in their tree.

Discipline:
- Read a file before editing; confirm the edit landed.
- Make the smallest change that does the task; don't touch files that weren't part of the request.
- All paths stay inside the working directory. Cite as `path/to/file.py:42`.
