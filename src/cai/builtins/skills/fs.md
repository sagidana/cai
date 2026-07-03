name: fs
skills: fs-read-only
tools: fs__create_file, fs__edit_file, fs__rename_file, fs__move_file, fs__copy_file, fs__copy_bytes, fs__remove_file, fs__create_directory, fs__move_directory
---
# Skill: File System (Read-Write)

This skill extends `fs-read-only` with the tools that modify the tree. The read-only inspection tools (`fs__list_files`, `fs__search`, `fs__read_file`) come from that skill; use them to inspect before you mutate - never edit a file blind. Use this skill when a task requires creating, editing, moving, or deleting files and directories.

Modifying tools:
- `fs__create_file` - write a new file (parent directories are created automatically).
- `fs__edit_file` - replace the first occurrence of `old_text` with `new_text`; quote enough surrounding context to make the match unique.
- `fs__rename_file` / `fs__move_file` - rename in place or relocate a file; `fs__move_file` accepts a destination directory.
- `fs__copy_file` - copy a file to a new path; accepts a destination directory and preserves metadata.
- `fs__copy_bytes` - copy a byte range between files without the bytes passing through you: extract a slice to a new file, append, or patch a byte range in place (src offsets default to the whole file; dst offsets choose append/create vs replace).
- `fs__remove_file` - delete a regular file (directories are never removed).
- `fs__create_directory` - make a directory tree (`mkdir -p` semantics).
- `fs__move_directory` - rename or relocate a directory.

Scratch directory:
- A session scratch directory exists for intermediates - bulky tool outputs, binary artifacts, working files that are not part of the project. Address it as `$CAI_SCRATCH` in any path (e.g. `fs__copy_bytes(src, "$CAI_SCRATCH/head.bin", ...)`); it also expands in `bash` commands.
- It lives for this session only and is deleted afterwards - move anything worth keeping into the project tree.
- Prefer it over the working directory for anything the user did not ask to have in their tree.

Discipline:
- Always use absolute paths (or the `$CAI_SCRATCH` prefix for scratch).
- Read a file before editing it, and confirm the edit landed where you intended.
- Make the smallest change that accomplishes the task; do not create, move, or delete files that were not part of the request.
- All paths must stay inside the working directory - traversal outside it is rejected.
- Cite findings and changes with file path and line number: `path/to/file.py:42`
