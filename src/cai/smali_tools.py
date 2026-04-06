import json
import os
import subprocess

from cai.utils import safe_path


_RUNNABLE_PREFIXES = (
    'Ljava/lang/Runnable;->run()',
    'Ljava/util/concurrent/Callable;->call()',
    'Landroid/os/Handler;->post(',
    'Landroid/os/Handler;->postDelayed(',
    'Ljava/util/concurrent/ExecutorService;->submit(',
    'Ljava/util/concurrent/ExecutorService;->execute(',
    'Ljava/lang/Thread;->start()',
    'Landroid/os/AsyncTask;->execute(',
    'Landroid/os/AsyncTask;->executeOnExecutor(',
    'Ljava/util/concurrent/ScheduledExecutorService;->schedule(',
)


def _smali_parse(fpath: str):
    """Parse a smali file. Returns (root_node, lines_list)."""
    from tree_sitter_language_pack import get_parser
    with open(fpath, 'rb') as _f:
        _src = _f.read()
    _root = get_parser('smali').parse(_src).root_node
    return _root, _src.decode('utf-8', errors='replace').splitlines()


def _smali_walk(node, node_type: str):
    """Yield all descendant nodes of node_type (depth-first)."""
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == node_type:
            yield n
        stack.extend(reversed(n.children))


def _smali_class_info(root):
    """Return (class_descriptor, modifiers_set) from a parsed smali root."""
    class_dir = next((c for c in root.children if c.type == 'class_directive'), None)
    if not class_dir:
        return '', set()
    class_id = next((c for c in class_dir.children if c.type == 'class_identifier'), None)
    class_desc = class_id.text.decode() if class_id else ''
    access = next((c for c in class_dir.children if c.type == 'access_modifiers'), None)
    mods = set()
    if access:
        for mod in access.children:
            if mod.type == 'access_modifier':
                mods.add(mod.text.decode())
    return class_desc, mods


def _smali_method_sig(method_node):
    """Return the method_signature text from a method_definition node."""
    sig = next((c for c in method_node.children if c.type == 'method_signature'), None)
    return sig.text.decode() if sig else ''


def _smali_enclosing_method(node):
    """Walk up the AST to find the nearest method_definition ancestor."""
    n = node.parent
    while n is not None:
        if n.type == 'method_definition':
            return n
        n = n.parent
    return None


def _smali_rg_files(needle: str, safe_path: str) -> list:
    """Return smali files whose text contains needle (fixed-string, via rg)."""
    r = subprocess.run(
        ['rg', '--fixed-strings', '-l', needle, '--glob', '*.smali', safe_path],
        capture_output=True, text=True
    )
    return [f.strip() for f in r.stdout.splitlines() if f.strip()] if r.returncode in (0, 1) else []


def register(mcp):
    @mcp.tool()
    def smali_find_callers(descriptor: str, path: str = ".") -> str:
        """Find all smali methods that invoke the given method descriptor via any invoke-* opcode.

        Searches the smali codebase for every call site that targets descriptor.
        Uses ripgrep for fast pre-filtering, then tree-sitter for precise structural parsing
        so caller context (enclosing class and method) is always correct.

        Args:
            descriptor: Full or partial smali method descriptor.
                        Full: "Lcom/example/Foo;->bar(I)V"
                        Partial filter: "->encrypt(" or "Lcom/example/Crypto;"
            path:       Directory to search. Defaults to "." (working directory).

        Returns:
            JSON array of caller objects:
              caller_file       — relative path to the smali file
              caller_class      — class descriptor of the caller ("Lcom/example/Bar;")
              caller_method     — method signature of the caller ("doWork(I)V")
              caller_descriptor — full caller descriptor
              line              — 1-based line number of the invoke instruction
              invoke_line       — raw invoke instruction text
              invoke_kind       — "virtual", "static", "interface", "direct", or "super"
            Returns "[]" if no callers found. Returns "Error: ..." on path violations.
        """
        try:
            safe = safe_path(path)
        except ValueError as e:
            return str(e)

        callers = []
        for fpath in _smali_rg_files(descriptor, safe):
            try:
                root, lines = _smali_parse(fpath)
            except Exception:
                continue
            class_desc, _ = _smali_class_info(root)
            for expr in _smali_walk(root, 'expression'):
                opcode_node = next((c for c in expr.children if c.type == 'opcode'), None)
                if opcode_node is None:
                    continue
                opcode_text = opcode_node.text.decode()
                if not opcode_text.startswith('invoke-'):
                    continue
                body_node = next((c for c in expr.children if c.type == 'body'), None)
                if body_node is None:
                    continue
                full_sig = next((c for c in body_node.children if c.type == 'full_method_signature'), None)
                if full_sig is None or descriptor not in full_sig.text.decode():
                    continue
                lineno = expr.start_point[0] + 1
                invoke_line = lines[lineno - 1].strip() if lineno <= len(lines) else ''
                method_node = _smali_enclosing_method(expr)
                method_sig = _smali_method_sig(method_node) if method_node else ''
                kind = opcode_text[len('invoke-'):]
                callers.append({
                    'caller_file': os.path.relpath(fpath, safe),
                    'caller_class': class_desc,
                    'caller_method': method_sig,
                    'caller_descriptor': f'{class_desc}->{method_sig}',
                    'line': lineno,
                    'invoke_line': invoke_line,
                    'invoke_kind': kind,
                })
        return json.dumps(callers)

    @mcp.tool()
    def smali_find_callees(descriptor: str, path: str = ".") -> str:
        """Find all methods invoked from within the body of the given smali method.

        Args:
            descriptor: Full smali method descriptor: "Lcom/example/Foo;->bar(I)V".
                        The class part is used to locate the smali file.
            path:       Directory to search. Defaults to "." (working directory).

        Returns:
            JSON array of callee objects:
              opcode           — the full invoke opcode (e.g. "invoke-virtual")
              callee_descriptor — full callee descriptor
              callee_class      — class part of the callee
              callee_method     — method signature of the callee
              line              — 1-based line number of the invoke instruction
              is_runnable_like  — true if callee looks like a deferred/async execution target
                                  (Runnable.run, Handler.post, ExecutorService.submit, etc.)
            Returns "Error: ..." on failure or if method not found.
        """
        try:
            safe = safe_path(path)
        except ValueError as e:
            return str(e)

        if '->' not in descriptor:
            return f'Error: descriptor must contain "->": {descriptor}'
        class_part, method_part = descriptor.split('->', 1)
        if not class_part.endswith(';'):
            class_part += ';'

        callees = []
        found_method = False
        for fpath in _smali_rg_files(class_part, safe):
            try:
                root, lines = _smali_parse(fpath)
            except Exception:
                continue
            class_desc, _ = _smali_class_info(root)
            if class_desc != class_part:
                continue
            target_method = None
            for m in _smali_walk(root, 'method_definition'):
                sig = _smali_method_sig(m)
                if sig == method_part or sig.startswith(method_part.split('(')[0] + '('):
                    target_method = m
                    break
            if target_method is None:
                continue
            found_method = True
            for expr in _smali_walk(target_method, 'expression'):
                opcode_node = next((c for c in expr.children if c.type == 'opcode'), None)
                if opcode_node is None:
                    continue
                opcode_text = opcode_node.text.decode()
                if not opcode_text.startswith('invoke-'):
                    continue
                body_node = next((c for c in expr.children if c.type == 'body'), None)
                if body_node is None:
                    continue
                full_sig = next((c for c in body_node.children if c.type == 'full_method_signature'), None)
                if full_sig is None:
                    continue
                callee_full = full_sig.text.decode()
                callee_class_node = next((c for c in full_sig.children if c.type == 'class_identifier'), None)
                callee_sig_node = next((c for c in full_sig.children if c.type == 'method_signature'), None)
                callee_class = callee_class_node.text.decode() if callee_class_node else ''
                callee_method = callee_sig_node.text.decode() if callee_sig_node else ''
                lineno = expr.start_point[0] + 1
                is_runnable = any(callee_full.startswith(p) for p in _RUNNABLE_PREFIXES)
                callees.append({
                    'opcode': opcode_text,
                    'callee_descriptor': callee_full,
                    'callee_class': callee_class,
                    'callee_method': callee_method,
                    'line': lineno,
                    'is_runnable_like': is_runnable,
                })
        if not found_method:
            return f'Error: method not found: {descriptor}'
        return json.dumps(callees)

    @mcp.tool()
    def smali_resolve_descriptor(query: str, path: str = ".") -> str:
        """Resolve a partial class or method name to full smali descriptors.

        Useful when you know "MainActivity" or "encrypt" but need the full descriptor
        "Lcom/example/app/MainActivity;->encrypt(I)V". Handles obfuscated names
        (e.g. "La;") by returning all candidates — use signature to disambiguate.

        Args:
            query: Partial class name, method name, package path, or full/partial descriptor.
                   If already a full smali descriptor (starts with "L", contains "->"),
                   returned as-is. Case-insensitive matching.
            path:  Directory to search. Defaults to "." (working directory).

        Returns:
            JSON array (up to 20 results):
              descriptor     — full smali descriptor ("Lclass;->method(sig)ret" or "Lclass;")
              file           — relative path to the smali file
              kind           — "class" or "method"
              class_modifier — "concrete", "abstract", or "interface"
            Returns "[]" if nothing matches. Returns "Error: ..." on path violations.
        """
        try:
            safe = safe_path(path)
        except ValueError as e:
            return str(e)

        # Already a full descriptor — return immediately
        if query.startswith('L') and '->' in query and '(' in query:
            return json.dumps([{'descriptor': query, 'file': '', 'kind': 'method', 'class_modifier': 'unknown'}])

        results = []
        seen: set = set()

        rg_r = subprocess.run(
            ['rg', '-i', '-l', query, '--glob', '*.smali', safe],
            capture_output=True, text=True
        )
        candidate_files = [f.strip() for f in rg_r.stdout.splitlines() if f.strip()] if rg_r.returncode in (0, 1) else []

        for fpath in candidate_files:
            try:
                root, _ = _smali_parse(fpath)
            except Exception:
                continue
            class_desc, mods = _smali_class_info(root)
            if not class_desc:
                continue
            if 'interface' in mods:
                modifier = 'interface'
            elif 'abstract' in mods:
                modifier = 'abstract'
            else:
                modifier = 'concrete'
            rel_file = os.path.relpath(fpath, safe)

            if query.lower() in class_desc.lower() and class_desc not in seen:
                seen.add(class_desc)
                results.append({'descriptor': class_desc, 'file': rel_file, 'kind': 'class', 'class_modifier': modifier})

            for m in _smali_walk(root, 'method_definition'):
                sig = _smali_method_sig(m)
                if query.lower() in sig.lower():
                    full = f'{class_desc}->{sig}'
                    if full not in seen:
                        seen.add(full)
                        results.append({'descriptor': full, 'file': rel_file, 'kind': 'method', 'class_modifier': modifier})
            if len(results) >= 20:
                break

        return json.dumps(results[:20])

    @mcp.tool()
    def smali_find_implementations(interface_descriptor: str, method_name: str = "", path: str = ".", max_depth: int = 4) -> str:
        """Find all concrete classes implementing an interface or extending an abstract class.

        Recursively resolves through abstract intermediate classes so only
        instantiable (concrete) leaf implementations are returned.
        Essential for resolving invoke-interface call sites and tracking
        Runnable/Callable implementations passed as callbacks.

        Args:
            interface_descriptor: Full smali class descriptor of the interface or abstract class.
                                  e.g. "Ljava/lang/Runnable;" or "Lcom/example/IProcessor;"
            method_name:          Optional method name or signature to filter results.
                                  e.g. "run()V" or just "run". Leave empty to return all.
            path:                 Directory to search. Defaults to "."
            max_depth:            Maximum inheritance depth to recurse (default 4).

        Returns:
            JSON array of concrete implementation objects:
              kind              — "implementation" (implements interface) or "override" (extends class)
              implementor_class — full class descriptor of the concrete implementor
              implementor_file  — relative path to the smali file
              method_descriptor — full descriptor of the matching method (empty if method_name omitted)
              depth             — depth in the hierarchy (0 = direct implementor)
            Returns "[]" if no concrete implementations found. Returns "Error: ..." on path violations.
        """
        try:
            safe = safe_path(path)
        except ValueError as e:
            return str(e)

        results: list = []
        seen_classes: set = set()

        # Determine kind from the target class itself
        base_kind = 'implementation'
        for fpath in _smali_rg_files(interface_descriptor, safe):
            try:
                root, _ = _smali_parse(fpath)
            except Exception:
                continue
            cd, mods = _smali_class_info(root)
            if cd == interface_descriptor:
                base_kind = 'implementation' if 'interface' in mods else 'override'
                break

        def _recurse(target: str, depth: int):
            if depth > max_depth or target in seen_classes:
                return
            seen_classes.add(target)
            for fpath in _smali_rg_files(target, safe):
                try:
                    root, _ = _smali_parse(fpath)
                except Exception:
                    continue
                class_desc, mods = _smali_class_info(root)
                if not class_desc or class_desc == target:
                    continue
                has_target = any(
                    next((c for c in d.children if c.type == 'class_identifier' and c.text.decode() == target), None)
                    for d in root.children if d.type in ('implements_directive', 'super_directive')
                )
                if not has_target:
                    continue
                rel_file = os.path.relpath(fpath, safe)
                is_abstract = 'abstract' in mods or 'interface' in mods
                if is_abstract:
                    _recurse(class_desc, depth + 1)
                else:
                    if method_name:
                        for m in _smali_walk(root, 'method_definition'):
                            sig = _smali_method_sig(m)
                            if method_name.lower() in sig.lower():
                                results.append({
                                    'kind': base_kind,
                                    'implementor_class': class_desc,
                                    'implementor_file': rel_file,
                                    'method_descriptor': f'{class_desc}->{sig}',
                                    'depth': depth,
                                })
                    else:
                        results.append({
                            'kind': base_kind,
                            'implementor_class': class_desc,
                            'implementor_file': rel_file,
                            'method_descriptor': '',
                            'depth': depth,
                        })

        _recurse(interface_descriptor, 0)
        return json.dumps(results)
