"""extend: the `cai extend` subcommand - manage extension bundles.

An extension bundle is a self-contained directory carrying any of skills/*.md,
tools/*.py (function tools), mcps/*.py (MCP servers), init.py, hooks/init.py,
commands/init.py and an optional README.md.
Installing one means dropping the whole directory under
~/.config/cai/extensions/<name>/, where UserConfig.load() discovers it. This is
the inverse of the legacy `cai extension`, which flattened skills/ and mcps/ into
shared dirs; here the bundle stays whole so the new loader can attribute hooks
and commands back to it.

Three modes, dispatched by run():
  cai extend <folder|.zip|http(s) url> [--replace]   install a bundle
  cai extend --list                                  list installed bundles
  cai extend --remove <name>                          uninstall a bundle"""
from __future__ import annotations

import os
import shutil
import zipfile
import tempfile

from cai.userconfig import UserConfig


# the files/dirs that mark a directory as a real bundle. used both to validate a
# source and to descend through a single wrapper folder a zip often introduces.
BUNDLE_MARKERS = ("skills", "tools", "mcps", "init.py", "hooks", "commands", "README.md")


def _extension_completer(prefix, **kwargs):
    """tab completion for `--remove <name>`: the installed extension names."""
    matching = []
    for extension in UserConfig.list_extensions():
        if not extension.name.startswith(prefix): continue
        matching.append(extension.name)
    return matching


def _bundle_root(root):
    """if a bundle's files sit inside a single wrapper subdirectory (common when
    a zip wraps everything in one top-level folder), descend into it; otherwise
    use root as-is."""
    for name in BUNDLE_MARKERS:
        if os.path.exists(os.path.join(root, name)):
            return root

    entries = []
    for name in os.listdir(root):
        if name.startswith("__MACOSX"): continue
        entries.append(name)
    if len(entries) != 1:
        return root

    sub = os.path.join(root, entries[0])
    if not os.path.isdir(sub):
        return root
    return sub


def _is_valid_bundle(bundle):
    """a directory is a bundle if it carries at least one marker."""
    for name in BUNDLE_MARKERS:
        if name == "README.md": continue
        if os.path.exists(os.path.join(bundle, name)):
            return True
    return False


def _extract_zip(zip_path, workdir, source):
    """unzip into workdir/extracted and return its bundle root, or None after
    printing an error."""
    extract_dir = os.path.join(workdir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        print(f"[!] extend: not a valid zip archive: {source}")
        return None
    return _bundle_root(extract_dir)


def _download(url, workdir):
    """fetch a zip url into workdir/bundle.zip, or None after printing an error."""
    import requests

    zip_path = os.path.join(workdir, "bundle.zip")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk: continue
                f.write(chunk)
    except requests.RequestException as e:
        print(f"[!] extend: download failed: {e}")
        return None
    return zip_path


def _archive_name(source):
    """the extension name implied by a zip/url path: its basename without the
    .zip suffix. used when the archive has no single wrapper folder to name it."""
    base = os.path.basename(source.rstrip("/"))
    if base.endswith(".zip"):
        base = base[:-len(".zip")]
    return base


def _resolve_bundle(source, workdir):
    """turn a folder / .zip / http(s) url into a local (bundle_dir, name), or
    (None, None) after printing an error. name is the wrapper folder's basename
    when present, else the folder/archive basename."""
    if source.startswith("http://") or source.startswith("https://"):
        zip_path = _download(source, workdir)
        if zip_path is None:
            return None, None
        root = _extract_zip(zip_path, workdir, source)
        if root is None:
            return None, None
        return root, _resolved_name(root, _archive_name(source), workdir)

    if not os.path.exists(source):
        print(f"[!] extend: no such path: {source}")
        return None, None

    if os.path.isdir(source):
        root = _bundle_root(source)
        return root, os.path.basename(os.path.abspath(root))

    if zipfile.is_zipfile(source):
        root = _extract_zip(source, workdir, source)
        if root is None:
            return None, None
        return root, _resolved_name(root, _archive_name(source), workdir)

    print(f"[!] extend: {source} is not a folder or a .zip file.")
    return None, None


def _resolved_name(root, archive_name, workdir):
    """name an extracted bundle: the wrapper folder's basename when extractall
    produced one (root descended below workdir/extracted), else the archive's
    own name (a flat zip)."""
    extracted = os.path.abspath(os.path.join(workdir, "extracted"))
    if os.path.abspath(root) != extracted:
        return os.path.basename(os.path.abspath(root))
    return archive_name


def _cmd_install(source, replace):
    """install a bundle into ~/.config/cai/extensions/<name>/. aborts the whole
    bundle if that directory already exists, unless replace is set."""
    tmp = tempfile.TemporaryDirectory(prefix="cai-extend-")
    try:
        bundle, name = _resolve_bundle(source, tmp.name)
        if bundle is None:
            return 1
        if not name:
            print(f"[!] extend: could not determine an extension name for {source}")
            return 1
        if not _is_valid_bundle(bundle):
            print("[!] extend: not a bundle - no skills/, tools/, init.py, "
                  "hooks/ or commands/ found.")
            return 1

        dest = os.path.join(UserConfig.extensions_dir(), name)
        if os.path.exists(dest):
            if not replace:
                print(f"[!] extend: extension '{name}' is already installed at {dest}")
                print("[!] extend: pass --replace to overwrite it.")
                return 1
            shutil.rmtree(dest)

        os.makedirs(UserConfig.extensions_dir(), exist_ok=True)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(bundle, dest, ignore=ignore)
        print(f"[*] installed extension '{name}' into {dest}")
        _print_readme(dest)
        return 0
    finally:
        tmp.cleanup()


def _print_readme(dest):
    readme = os.path.join(dest, "README.md")
    if not os.path.isfile(readme):
        return
    print()
    with open(readme) as f:
        print(f.read())


def _cmd_list():
    """list installed extensions, each with the components it carries."""
    extensions = UserConfig.list_extensions()
    if not extensions:
        print(f"no extensions installed under {UserConfig.extensions_dir()}")
        return 0
    for extension in extensions:
        print(f"{extension.name}  ({', '.join(_components(extension))})")
    return 0


def _components(extension):
    """the human-readable list of what an installed bundle contributes."""
    parts = []
    if os.path.isdir(extension.skills_dir):
        parts.append("skills")
    if os.path.isdir(extension.tools_dir):
        parts.append("tools")
    if os.path.isdir(extension.mcps_dir):
        parts.append("mcps")
    if os.path.isfile(extension.init_file):
        parts.append("init")
    if os.path.isfile(extension.hooks_file):
        parts.append("hooks")
    if os.path.isfile(extension.commands_file):
        parts.append("commands")
    if not parts:
        parts.append("empty")
    return parts


def _cmd_remove(name):
    """uninstall an extension by deleting its directory."""
    dest = os.path.join(UserConfig.extensions_dir(), name)
    if not os.path.isdir(dest):
        print(f"[!] extend: no extension named '{name}' under {UserConfig.extensions_dir()}")
        return 1
    shutil.rmtree(dest)
    print(f"[*] removed extension '{name}'")
    return 0


def run(args, parser):
    """dispatch `cai extend` to exactly one of install / --list / --remove."""
    modes = 0
    if args.source is not None:
        modes += 1
    if args.list:
        modes += 1
    if args.remove is not None:
        modes += 1
    if modes == 0:
        parser.error("cai extend: give a source to install, --list, or --remove <name>")
    if modes > 1:
        parser.error("cai extend: source, --list and --remove are mutually exclusive")

    if args.list:
        return _cmd_list()
    if args.remove is not None:
        return _cmd_remove(args.remove)
    return _cmd_install(args.source, args.replace)
