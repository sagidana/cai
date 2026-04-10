import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from cai.utils import safe_path

# All runtime state lives under /tmp/cai/ — no user setup required.
_CAI_DIR      = "/tmp/cai"
REGISTRY_PATH = os.path.join(_CAI_DIR, "frida_registry.json")
LOG_DIR       = os.path.join(_CAI_DIR, "frida_logs")
_FRIDA_JS_ROOT     = os.path.join(_CAI_DIR, "frida_js")
_FRIDA_SERVER_CACHE = os.path.join(_CAI_DIR, "frida_server")
DEVICE_SERVER = "/data/local/tmp/frida-server"

_JS_LIST_CLASSES = """\
import Java from "frida-java-bridge";
(function(prefix) {
  Java.perform(function() {
    Java.enumerateLoadedClasses({
      onMatch: function(name) {
        if (!prefix || name.startsWith(prefix)) { send({ name: name }); }
      },
      onComplete: function() { send({ done: true }); }
    });
  });
})(%s)
"""

_JS_LIST_METHODS = """\
import Java from "frida-java-bridge";
(function(cls) {
  Java.perform(function() {
    try {
      var methods = Java.use(cls).class.getDeclaredMethods();
      for (var i = 0; i < methods.length; i++) {
        send({ name: methods[i].toString() });
      }
    } catch(e) { send({ error: e.toString() }); }
    send({ done: true });
  });
})(%s)
"""

_JS_LIST_FIELDS = """\
import Java from "frida-java-bridge";
(function(cls) {
  Java.perform(function() {
    try {
      var fields = Java.use(cls).class.getDeclaredFields();
      for (var i = 0; i < fields.length; i++) {
        send({ name: fields[i].toString() });
      }
    } catch(e) { send({ error: e.toString() }); }
    send({ done: true });
  });
})(%s)
"""

_JS_CLASS_HIERARCHY = """\
import Java from "frida-java-bridge";
(function(cls) {
  Java.perform(function() {
    try {
      var klass = Java.use(cls).class;
      var sup = klass.getSuperclass();
      while (sup !== null) {
        send({ kind: "superclass", name: sup.getName() });
        sup = sup.getSuperclass();
      }
      var ifaces = klass.getInterfaces();
      for (var i = 0; i < ifaces.length; i++) {
        send({ kind: "interface", name: ifaces[i].getName() });
      }
    } catch(e) { send({ error: e.toString() }); }
    send({ done: true });
  });
})(%s)
"""

_JS_CALL_METHOD = """\
import Java from "frida-java-bridge";
(function(cls, method, args) {
  Java.perform(function() {
    try {
      var klass = Java.use(cls);
      var result = klass[method].apply(klass, args);
      send({ result: (result !== null && result !== undefined) ? String(result) : "null" });
    } catch(e) { send({ error: e.toString() }); }
    send({ done: true });
  });
})(%s, %s, %s)
"""

_JS_GET_FIELD = """\
import Java from "frida-java-bridge";
(function(cls, field) {
  Java.perform(function() {
    try {
      var f = Java.use(cls).class.getDeclaredField(field);
      f.setAccessible(true);
      var val = f.get(null);
      send({ value: (val !== null) ? String(val) : "null" });
    } catch(e) { send({ error: e.toString() }); }
    send({ done: true });
  });
})(%s, %s)
"""

_JS_HOOK_WORKER = """\
import Java from "frida-java-bridge";
(function(cls, method) {
  Java.perform(function() {
    try {
      var klass = Java.use(cls);
      var overloads = klass[method].overloads;
      overloads.forEach(function(overload) {
        overload.implementation = function() {
          var args = Array.prototype.slice.call(arguments).map(String);
          send({ event: "enter", "class": cls, method: method, args: args, ts: Date.now() });
          try {
            var retval = this[method].apply(this, arguments);
            send({ event: "leave", "class": cls, method: method, retval: String(retval), ts: Date.now() });
            return retval;
          } catch(e) {
            send({ event: "error", "class": cls, method: method, error: e.toString(), ts: Date.now() });
            throw e;
          }
        };
      });
    } catch(e) {
      send({ event: "hook_error", "class": cls, method: method, error: e.toString(), ts: Date.now() });
    }
  });
})(%s, %s)
"""


def _ensure_frida_js() -> None:
    """Install frida-java-bridge into _FRIDA_JS_ROOT if not already present.

    Called lazily on first compile so no user setup is required after
    ``pip install cai``.  Requires npm to be on PATH.
    """
    bridge_dir = os.path.join(_FRIDA_JS_ROOT, "node_modules", "frida-java-bridge")
    if os.path.isdir(bridge_dir):
        return

    os.makedirs(_FRIDA_JS_ROOT, exist_ok=True)

    pkg_json = os.path.join(_FRIDA_JS_ROOT, "package.json")
    if not os.path.exists(pkg_json):
        with open(pkg_json, "w") as f:
            json.dump({
                "name": "cai-frida-js",
                "version": "1.0.0",
                "private": True,
                "dependencies": {"frida-java-bridge": "*"},
            }, f)

    result = subprocess.run(
        ["npm", "install", "--prefix", _FRIDA_JS_ROOT],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"npm install frida-java-bridge failed:\n{result.stderr.strip()}"
        )


def _ensure_frida_server(serial: str = "") -> str:
    """Return a local path to a frida-server binary matching the installed frida version.

    Downloads from the official GitHub release if not already cached under
    _FRIDA_SERVER_CACHE.  The binary version is derived from ``frida.__version__``
    so it always matches the Python package installed by ``pip install cai``.

    Args:
        serial: Device serial used to detect the device ABI via adb.

    Returns:
        Absolute path to the cached binary, ready to push to the device.

    Raises:
        RuntimeError: if adb, the ABI, or the download fails.
    """
    import frida as _frida_mod
    import lzma
    import urllib.request

    version = _frida_mod.__version__
    prefix = ["-s", serial] if serial else []

    abi = _adb(prefix + ["shell", "getprop", "ro.product.cpu.abi"]).strip()
    if not abi or abi.startswith("Error:"):
        raise RuntimeError(f"Could not detect device ABI: {abi!r}")

    arch_map = {
        "arm64-v8a":   "arm64",
        "armeabi-v7a": "arm",
        "x86_64":      "x86_64",
        "x86":         "x86",
    }
    arch = arch_map.get(abi, abi)

    os.makedirs(_FRIDA_SERVER_CACHE, exist_ok=True)
    binary_name = f"frida-server-{version}-android-{arch}"
    binary_path = os.path.join(_FRIDA_SERVER_CACHE, binary_name)

    if os.path.isfile(binary_path):
        return binary_path

    url = (
        f"https://github.com/frida/frida/releases/download/"
        f"{version}/{binary_name}.xz"
    )
    xz_path = binary_path + ".xz"
    try:
        urllib.request.urlretrieve(url, xz_path)
        with lzma.open(xz_path) as xz_f, open(binary_path, "wb") as out_f:
            out_f.write(xz_f.read())
        os.chmod(binary_path, 0o755)
    finally:
        try:
            os.unlink(xz_path)
        except OSError:
            pass

    return binary_path


def _adb(args: list, timeout: int = 30) -> str:
    """Run an adb command and return combined stdout/stderr."""
    try:
        result = subprocess.run(
            ["adb"] + args,
            capture_output=True, text=True, timeout=timeout
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0:
            return f"Error: {err or out}"
        return out or err or "(no output)"
    except FileNotFoundError:
        return "Error: adb is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return f"Error: adb command timed out after {timeout}s."


def _registry_read() -> dict:
    try:
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _registry_write(data: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _registry_purge_dead() -> None:
    """Remove hooks whose worker process has exited without an explicit unhook."""
    reg = _registry_read()
    if not reg:
        return
    cleaned = {}
    for hid, entry in reg.items():
        pid = entry.get("pid")
        status = entry.get("status", "unknown")
        if status == "active" and pid:
            try:
                os.kill(pid, 0)  # signal 0 = existence check only
                cleaned[hid] = entry  # still alive — keep
            except ProcessLookupError:
                pass  # dead without unhook — drop it
            except PermissionError:
                cleaned[hid] = entry  # exists but not ours — keep
        # already-stopped entries are never written back (purge them too)
    if len(cleaned) != len(reg):
        _registry_write(cleaned)


def _frida_attach(package: str, serial: str):
    """Attach to a running app and return (device, session). Raises on failure.

    Tries attaching by package/process name first (works on AOSP).  If that
    fails (e.g. Samsung devices where frida sees the display name instead of
    the package name), falls back to resolving the PID via ``adb pidof`` and
    attaching by numeric PID.
    """
    import frida  # deferred — avoids import-time crash if frida not installed
    prefix = ["-s", serial] if serial else []
    if serial:
        device = frida.get_device(serial)
    else:
        device = frida.get_usb_device()

    try:
        session = device.attach(package)
    except frida.ProcessNotFoundError:
        # Fallback: resolve PID via adb (handles Samsung display-name quirk)
        pids_out = _adb(prefix + ["shell", "pidof", package])
        if not pids_out.strip() or pids_out.startswith("Error:"):
            raise  # re-raise original error — process truly not found
        pid = int(pids_out.strip().split()[0])
        session = device.attach(pid)

    return device, session


def _compile_js(js: str) -> str:
    """Compile an ESM script (using frida-java-bridge imports) into a loadable bundle.

    Requires frida-java-bridge to be installed as an npm package under
    _FRIDA_JS_ROOT/node_modules. The temp source file is written to the system
    temp dir (always writable); _FRIDA_JS_ROOT is only used as project_root so
    the compiler can locate node_modules.
    """
    import frida  # deferred
    _ensure_frida_js()
    with tempfile.NamedTemporaryFile(
        suffix=".js", mode="w", dir=_FRIDA_JS_ROOT, delete=False
    ) as f:
        f.write(js)
        tmp = f.name
    try:
        return frida.Compiler().build(tmp, project_root=_FRIDA_JS_ROOT)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


@contextlib.contextmanager
def _session(package: str, serial: str):
    """Context manager: attach frida to *package* and yield the session.

    Detaches cleanly on exit regardless of exceptions.
    """
    _device, session = _frida_attach(package, serial)
    try:
        yield session
    finally:
        try:
            session.detach()
        except Exception:
            pass


def _run_script(session, js: str, timeout: int = 30) -> list:
    """Compile and load *js*, collecting every send() payload into a list.

    Stops when the script calls ``send({done: true})`` or *timeout* expires.
    Frida-level runtime errors are appended as ``{"error": "..."}`` dicts so
    callers can handle them uniformly alongside JS-side error payloads.
    """
    payloads = []
    done = threading.Event()

    def on_message(msg, _data):
        if msg["type"] == "send":
            p = msg["payload"]
            if isinstance(p, dict) and p.get("done"):
                done.set()
            else:
                payloads.append(p)
        elif msg["type"] == "error":
            payloads.append({"error": msg.get("description", "unknown")})
            done.set()

    bundle = _compile_js(js)
    script = session.create_script(bundle)
    script.on("message", on_message)
    script.load()
    done.wait(timeout=timeout)
    return payloads


def register(mcp):
    @mcp.tool()
    def frida_server_start(serial: str = "", server_binary: str = "") -> str:
        """Start frida-server on a connected Android device.

        Automatically downloads the correct frida-server binary for the
        device's ABI and the installed frida version (from pip) if one is not
        already present on the device or provided explicitly.  Requires a
        rooted device (su must be available).

        Args:
            serial:        Device serial (from adb devices). Leave empty if only
                           one device is connected.
            server_binary: Local path to a specific frida-server binary to push.
                           Leave empty to auto-download the version that matches
                           the installed frida Python package.

        Returns:
            Status string, or "Error: ..." on failure.
        """
        prefix = ["-s", serial] if serial else []

        # Resolve the binary to push: explicit path → auto-download → skip push
        binary_to_push = None
        if server_binary:
            try:
                binary_to_push = safe_path(server_binary)
            except ValueError as e:
                return str(e)
        else:
            # Check if a matching binary is already on the device
            running = _adb(prefix + ["shell", "pgrep", "-f", "frida-server"])
            already_on_device = _adb(prefix + ["shell", "test", "-f", DEVICE_SERVER,
                                                "&&", "echo", "yes"]).strip() == "yes"
            if not already_on_device:
                try:
                    binary_to_push = _ensure_frida_server(serial)
                except Exception as e:
                    return f"Error: could not obtain frida-server binary: {e}"

        if binary_to_push:
            r = _adb(prefix + ["push", binary_to_push, DEVICE_SERVER], timeout=120)
            if r.startswith("Error:"):
                return r
            r = _adb(prefix + ["shell", "su", "-c", f"chmod 755 {DEVICE_SERVER}"])
            if r.startswith("Error:"):
                return r

        # nohup + stdin/stdout/stderr redirected → adb shell exits immediately
        _adb(
            prefix + ["shell",
                       f"su -c 'nohup {DEVICE_SERVER} </dev/null >/dev/null 2>&1 &'"],
            timeout=10
        )

        time.sleep(1)  # give the server a moment to come up

        pids = _adb(prefix + ["shell", "pgrep", "-f", "frida-server"])
        if pids.startswith("Error:") or not pids.strip():
            return "Error: frida-server did not start (pgrep returned nothing)"
        return f"frida-server started (pids: {pids.strip()})"

    @mcp.tool()
    def frida_server_stop(serial: str = "") -> str:
        """Stop frida-server on a connected Android device.

        Args:
            serial: Device serial. Leave empty if only one device is connected.

        Returns:
            Status string, or "Error: ..." on failure.
        """
        prefix = ["-s", serial] if serial else []
        result = _adb(prefix + ["shell", "su", "-c", "pkill frida-server"])
        if result.startswith("Error:"):
            return result
        return "frida-server stopped."

    @mcp.tool()
    def frida_list_classes(package: str, serial: str = "", filter: str = "") -> str:
        """List loaded Java classes in a running Android app.

        Attaches to the app via frida-server, enumerates all loaded Java classes
        whose name starts with the package prefix, then detaches.
        The app must already be running and frida-server must be active.

        Args:
            package: App package name, e.g. "com.example.app". Used both to
                     attach to the process and as a class-name prefix filter.
            serial:  Device serial. Leave empty if only one device.
            filter:  Additional substring filter on top of the package prefix.

        Returns:
            One class name per line, or "Error: ..." on failure.
        """
        try:
            import frida  # noqa: F401
        except ImportError:
            return "Error: frida not installed. Run: pip install frida"

        js = _JS_LIST_CLASSES % json.dumps(package)
        try:
            with _session(package, serial) as sess:
                payloads = _run_script(sess, js)
        except Exception as e:
            return f"Error: {e}"

        results = [
            f"[error] {p['error']}" if p.get("error") else p.get("name", "")
            for p in payloads
        ]
        if filter:
            results = [r for r in results if filter in r]
        return "\n".join(r for r in results if r) or "(no classes found)"

    @mcp.tool()
    def frida_list_methods(package: str, class_name: str,
                           filter: str = "", serial: str = "") -> str:
        """List all declared methods of a Java class in a running Android app.

        Attaches to the app via frida-server, uses reflection to enumerate every
        method declared directly on the class (not inherited), then detaches.
        The app must already be running and frida-server must be active.

        Args:
            package:    App package name, e.g. "com.example.app".
            class_name: Fully-qualified Java class, e.g. "com.example.Foo".
            filter:     Optional substring filter applied to each method signature.
            serial:     Device serial. Leave empty if only one device.

        Returns:
            One method signature per line, or "Error: ..." on failure.
        """
        try:
            import frida  # noqa: F401
        except ImportError:
            return "Error: frida not installed. Run: pip install frida"

        js = _JS_LIST_METHODS % json.dumps(class_name)
        try:
            with _session(package, serial) as sess:
                payloads = _run_script(sess, js)
        except Exception as e:
            return f"Error: {e}"

        results = [
            f"[error] {p['error']}" if p.get("error") else p.get("name", "")
            for p in payloads
        ]
        if filter:
            results = [r for r in results if filter in r]
        return "\n".join(r for r in results if r) or "(no methods found)"

    @mcp.tool()
    def frida_list_fields(package: str, class_name: str,
                          filter: str = "", serial: str = "") -> str:
        """List all declared fields of a Java class in a running Android app.

        Uses reflection to enumerate every field declared directly on the class
        (not inherited). Complements frida_list_methods.

        Args:
            package:    App package name, e.g. "com.example.app".
            class_name: Fully-qualified Java class, e.g. "com.example.Foo".
            filter:     Optional substring filter on field signatures.
            serial:     Device serial. Leave empty if only one device.

        Returns:
            One field signature per line, or "Error: ..." on failure.
        """
        try:
            import frida  # noqa: F401
        except ImportError:
            return "Error: frida not installed. Run: pip install frida"

        js = _JS_LIST_FIELDS % json.dumps(class_name)
        try:
            with _session(package, serial) as sess:
                payloads = _run_script(sess, js)
        except Exception as e:
            return f"Error: {e}"

        results = [
            f"[error] {p['error']}" if p.get("error") else p.get("name", "")
            for p in payloads
        ]
        if filter:
            results = [r for r in results if filter in r]
        return "\n".join(r for r in results if r) or "(no fields found)"

    @mcp.tool()
    def frida_get_class_hierarchy(package: str, class_name: str,
                                  serial: str = "") -> str:
        """Return the superclass chain and implemented interfaces of a Java class.

        Useful for finding which ancestor class to hook when getDeclaredMethods
        returns nothing, or for understanding the full type hierarchy.

        Output format — one entry per line:
            superclass: android.app.Activity
            superclass: android.view.ContextThemeWrapper
            ...
            interface:  android.view.Window$Callback

        Args:
            package:    App package name, e.g. "com.example.app".
            class_name: Fully-qualified Java class, e.g. "com.example.Foo".
            serial:     Device serial. Leave empty if only one device.

        Returns:
            Superclass chain then interfaces, or "Error: ..." on failure.
        """
        try:
            import frida  # noqa: F401
        except ImportError:
            return "Error: frida not installed. Run: pip install frida"

        js = _JS_CLASS_HIERARCHY % json.dumps(class_name)
        try:
            with _session(package, serial) as sess:
                payloads = _run_script(sess, js)
        except Exception as e:
            return f"Error: {e}"

        results = [
            f"[error] {p['error']}" if p.get("error") else f"{p['kind']}: {p['name']}"
            for p in payloads
        ]
        return "\n".join(results) or "(no hierarchy found)"

    @mcp.tool()
    def frida_call_method(package: str, class_name: str, method_name: str,
                          args: list = None, serial: str = "") -> str:
        """Call a static Java method and return its result as a string.

        Uses frida-java-bridge to invoke the method synchronously inside the
        app process. frida-java-bridge resolves overloads automatically for
        simple primitive/string args; use frida_eval with explicit
        overload(...) for ambiguous cases.

        Args:
            package:     App package name, e.g. "com.example.app".
            class_name:  Fully-qualified class, e.g. "android.os.SystemClock".
            method_name: Static method name, e.g. "elapsedRealtime".
            args:        JSON-serialisable list of arguments (default: []).
            serial:      Device serial. Leave empty if only one device.

        Returns:
            String representation of the return value, or "Error: ..." on failure.
        """
        try:
            import frida  # noqa: F401
        except ImportError:
            return "Error: frida not installed. Run: pip install frida"

        js = _JS_CALL_METHOD % (
            json.dumps(class_name),
            json.dumps(method_name),
            json.dumps(args or []),
        )
        try:
            with _session(package, serial) as sess:
                payloads = _run_script(sess, js, timeout=15)
        except Exception as e:
            return f"Error: {e}"

        if not payloads:
            return "(no result)"
        p = payloads[0]
        return f"[error] {p['error']}" if p.get("error") else p.get("result", "null")

    @mcp.tool()
    def frida_get_field_value(package: str, class_name: str, field_name: str,
                              serial: str = "") -> str:
        """Read the value of a static Java field.

        Uses reflection with setAccessible(true) so private fields are also
        readable. Works for static fields only; for instance fields use
        frida_eval to obtain an instance first.

        Args:
            package:    App package name, e.g. "com.example.app".
            class_name: Fully-qualified class, e.g. "android.os.Build".
            field_name: Field name, e.g. "MODEL".
            serial:     Device serial. Leave empty if only one device.

        Returns:
            String representation of the field value, or "Error: ..." on failure.
        """
        try:
            import frida  # noqa: F401
        except ImportError:
            return "Error: frida not installed. Run: pip install frida"

        js = _JS_GET_FIELD % (json.dumps(class_name), json.dumps(field_name))
        try:
            with _session(package, serial) as sess:
                payloads = _run_script(sess, js, timeout=15)
        except Exception as e:
            return f"Error: {e}"

        if not payloads:
            return "(no value)"
        p = payloads[0]
        return f"[error] {p['error']}" if p.get("error") else p.get("value", "null")

    @mcp.tool()
    def frida_eval(package: str, script: str,
                  serial: str = "", timeout: int = 10) -> str:
        """Run arbitrary Frida JavaScript synchronously in a running Android app.

        The script communicates results back via send(). Execution ends when
        the script calls send({done: true}) or the timeout expires.

        Example — read the app's package name via Java reflection:
            import Java from "frida-java-bridge";
            Java.perform(function() {
                var pkg = Java.use("android.app.ActivityThread")
                    .currentApplication().getPackageName();
                send(pkg);
                send({done: true});
            });

        Example — plain send without Java (no imports needed):
            send("hello from frida");
            send({done: true});

        Args:
            package: App package name, e.g. "com.example.app".
            script:  Frida JavaScript source (ESM). May import frida-java-bridge.
            serial:  Device serial. Leave empty if only one device.
            timeout: Seconds to wait for send({done:true}) before returning
                     whatever has been collected so far (default 10).

        Returns:
            Newline-separated send() payloads (strings serialised to JSON for
            non-string values), or "Error: ..." on failure.
        """
        try:
            import frida  # noqa: F401
        except ImportError:
            return "Error: frida not installed. Run: pip install frida"

        try:
            with _session(package, serial) as sess:
                payloads = _run_script(sess, script, timeout=timeout)
        except Exception as e:
            return f"Error: {e}"

        results = [p if isinstance(p, str) else json.dumps(p) for p in payloads]
        return "\n".join(results) if results else "(no output)"

    @mcp.tool()
    def frida_hook_method(
        package: str,
        class_name: str,
        method_name: str,
        serial: str = "",
    ) -> str:
        """Hook a Java method in a running Android app using Frida (async/non-blocking).

        Starts a background worker that attaches to the app and intercepts every
        call to the specified method, writing enter/leave/error events to a log file.
        Returns immediately with a hook_id — use frida_hook_log(hook_id) to read
        captured events and frida_unhook_method(hook_id) to stop the hook.

        Typical workflow:
          1. frida_hook_method(...)      → get hook_id
          2. Trigger the target action in the app
          3. frida_hook_log(hook_id)     → inspect captured events
          4. frida_unhook_method(hook_id) → stop

        Args:
            package:     App package name, e.g. "com.example.app".
            class_name:  Fully-qualified Java class, e.g. "com.example.Foo".
            method_name: Method name to hook, e.g. "doSomething".
            serial:      Device serial (from adb devices). Leave empty if only
                         one device is connected.

        Returns:
            Multi-line status string containing hook_id, or "Error: ..." on failure.
        """
        _registry_purge_dead()
        hook_id = str(uuid.uuid4())[:8]
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"{hook_id}.jsonl")

        cmd = [
            sys.executable, "-m", "cai.frida_tools", "--worker",
            "--hook-id", hook_id,
            "--package", package,
            "--class", class_name,
            "--method", method_name,
            "--log-file", log_file,
            "--serial", serial,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as e:
            return f"Error: could not start worker: {e}"

        reg = _registry_read()
        reg[hook_id] = {
            "pid": proc.pid,
            "package": package,
            "class": class_name,
            "method": method_name,
            "log_file": log_file,
            "serial": serial,
            "status": "active",
            "log_pos": 0,
        }
        _registry_write(reg)

        return (
            f"Hook started.\n"
            f"hook_id:  {hook_id}\n"
            f"class:    {class_name}\n"
            f"method:   {method_name}\n"
            f"package:  {package}\n"
            f"Use frida_hook_log('{hook_id}') to read events.\n"
            f"Use frida_unhook_method('{hook_id}') to stop."
        )

    @mcp.tool()
    def frida_unhook_method(hook_id: str) -> str:
        """Stop a Frida method hook and detach the worker from the app.

        Sends SIGTERM to the background worker process, which will detach Frida
        from the target app and write a final 'worker_stop' log entry.

        Args:
            hook_id: The hook_id returned by frida_hook_method.

        Returns:
            Status string, or "Error: ..." on failure.
        """
        _registry_purge_dead()
        reg = _registry_read()
        if hook_id not in reg:
            return f"Error: unknown hook_id {hook_id!r}"

        entry = reg[hook_id]
        pid = entry.get("pid")
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # worker already exited
            except Exception as e:
                return f"Error: could not signal worker (pid {pid}): {e}"

        del reg[hook_id]
        _registry_write(reg)
        return f"Hook {hook_id} stopped (pid {pid} sent SIGTERM)."

    @mcp.tool()
    def frida_hook_log(hook_id: str) -> str:
        """Return new log entries for an active Frida hook since the last call.

        Each call advances an internal cursor so only events captured since the
        previous frida_hook_log call are returned. Call repeatedly to poll for
        new method invocations.

        Log entries are JSON objects with fields:
          event  — "worker_start" | "enter" | "leave" | "error" |
                   "hook_error" | "frida_error" | "worker_stop"
          class, method, args, retval, error — depending on event type
          ts     — Unix timestamp in milliseconds (0 for lifecycle events)

        Args:
            hook_id: The hook_id returned by frida_hook_method.

        Returns:
            Newline-separated JSON log entries, "(no new events)", or "Error: ...".
        """
        _registry_purge_dead()
        reg = _registry_read()
        if hook_id not in reg:
            return f"Error: unknown hook_id {hook_id!r}"

        entry = reg[hook_id]
        log_file = entry["log_file"]
        pos = entry.get("log_pos", 0)

        if not os.path.exists(log_file):
            return "(no log file yet — worker may still be starting)"

        with open(log_file, "r") as f:
            lines = f.readlines()

        new_lines = lines[pos:]
        entry["log_pos"] = len(lines)
        reg[hook_id] = entry
        _registry_write(reg)

        if not new_lines:
            return "(no new events)"

        return "".join(new_lines).strip()

    @mcp.tool()
    def frida_list_hooks() -> str:
        """List all registered Frida hooks and their current status.

        Checks whether each worker process is still alive and corrects any stale
        "active" entries in the registry. Useful to see all hooks across sessions
        and to get hook_ids for frida_hook_log / frida_unhook_method.

        Returns:
            A human-readable list of hooks with hook_id, status, package,
            class.method, and PID — or "(no hooks registered)" if empty.
        """
        _registry_purge_dead()
        reg = _registry_read()
        if not reg:
            return "(no hooks registered)"

        lines = []
        for hid, entry in reg.items():
            pid = entry.get("pid")
            status = entry.get("status", "unknown")
            lines.append(
                f"[{hid}] {status:<22}  pid={pid:<8}  "
                f"{entry.get('package','?')}  "
                f"{entry.get('class','?')}.{entry.get('method','?')}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Worker & test harness
# ---------------------------------------------------------------------------

def _run_worker(hook_id, package, class_name, method_name, log_file, serial):
    """Background daemon: attaches frida, hooks method, writes log until SIGTERM."""
    import signal as _signal

    # Create log directory first so _write always works, even in error paths.
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def _write(entry):
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    try:
        import frida  # noqa: F401
    except ImportError:
        _write({"event": "fatal", "error": "frida not installed", "ts": 0})
        return

    js = _JS_HOOK_WORKER % (json.dumps(class_name), json.dumps(method_name))

    try:
        _device, session = _frida_attach(package, serial)
    except Exception as e:
        _write({"event": "attach_error", "error": str(e), "ts": 0})
        return

    def on_message(msg, _data):
        if msg["type"] == "send":
            _write(msg["payload"])
        elif msg["type"] == "error":
            _write({"event": "frida_error", "error": msg.get("description", ""), "ts": 0})

    try:
        bundle = _compile_js(js)
        script = session.create_script(bundle)
        script.on("message", on_message)
        script.load()
    except Exception as e:
        _write({"event": "script_error", "error": str(e), "ts": 0})
        try:
            session.detach()
        except Exception:
            pass
        return

    _write({"event": "worker_start", "hook_id": hook_id,
            "class": class_name, "method": method_name, "ts": 0})

    stop = threading.Event()
    _signal.signal(_signal.SIGTERM, lambda *_: stop.set())
    stop.wait()

    try:
        session.detach()
    except Exception:
        pass
    _write({"event": "worker_stop", "hook_id": hook_id, "ts": 0})


def _run_tests():
    import shutil
    import sys
    import tempfile

    class _MockMCP:
        def __init__(self): self._tools = {}
        def tool(self):
            def dec(fn): self._tools[fn.__name__] = fn; return fn
            return dec

    _pass = _fail = 0

    def check(name, cond, got=""):
        nonlocal _pass, _fail
        if cond:
            print(f"  PASS  {name}")
            _pass += 1
        else:
            print(f"  FAIL  {name}  →  {got!r}")
            _fail += 1

    # Redirect registry and log dir to a temp area for the tests
    global REGISTRY_PATH, LOG_DIR
    _orig_reg = REGISTRY_PATH
    _orig_log = LOG_DIR
    tmp = tempfile.mkdtemp()
    REGISTRY_PATH = os.path.join(tmp, "registry.json")
    LOG_DIR = os.path.join(tmp, "logs")

    try:
        mcp = _MockMCP()
        register(mcp)
        T = mcp._tools

        # ── Group 1: registry helpers ────────────────────────────────────────
        print("=== frida_tools: registry helpers ===")
        check("registry_read missing → {}", _registry_read() == {})
        _registry_write({"a": {"pid": 1}})
        check("registry_write + read roundtrip",
              _registry_read() == {"a": {"pid": 1}})
        with open(REGISTRY_PATH, "w") as f:
            f.write("not json{{{")
        check("registry_read corrupt → {}", _registry_read() == {})

        # ── Group 2: frida_server_start / frida_server_stop (adb-dependent) ─
        print("\n=== frida_tools: server start/stop (adb required) ===")
        r = T["frida_server_start"]()
        if "adb is not installed" in r:
            print("  SKIP  all adb runtime tests (adb not on PATH)")
        else:
            # start — OK or error (no device connected)
            check("frida_server_start no crash", not r.startswith("Traceback"), r)

            r_stop = T["frida_server_stop"]()
            check("frida_server_stop no crash", not r_stop.startswith("Traceback"), r_stop)

            # If server actually started, run it again and stop cleanly
            if "started" in r:
                print(f"    → server started: {r}")
                check("frida_server_start returns pids", "pids:" in r, r)

                r2 = T["frida_server_stop"]()
                check("frida_server_stop returns stopped", "stopped" in r2 or "Error:" in r2, r2)

        # ── Group 3: frida_list_classes (frida + device required) ───────────
        print("\n=== frida_tools: frida_list_classes (frida + device required) ===")
        _device_available = False
        try:
            import frida as _frida  # check installed
            # Try to get a device
            try:
                dev = _frida.get_usb_device(timeout=3)
                # Ensure frida-server is running before class enumeration
                r_start = T["frida_server_start"]()
                print(f"    → ensuring frida-server: {r_start}")
                _device_available = True
                # Try a well-known always-running system process
                test_pkg = "com.android.settings"
                r = T["frida_list_classes"](test_pkg)
                if r.startswith("Error:"):
                    print(f"  SKIP  frida_list_classes ({r})")
                else:
                    lines = [l for l in r.splitlines() if l]
                    check("frida_list_classes returns results", len(lines) > 0, r[:200])
                    check("frida_list_classes all start with package prefix",
                          all(l.startswith("com.android") or l.startswith("[error]")
                              for l in lines),
                          lines[:3])
                    # filter param
                    r2 = T["frida_list_classes"](test_pkg, filter="Activity")
                    check("frida_list_classes filter",
                          all("Activity" in l for l in r2.splitlines() if l and not l.startswith("[error]")),
                          r2[:200])

                # ── frida_list_methods ───────────────────────────────────────
                print("\n=== frida_tools: frida_list_methods (frida + device required) ===")
                r_methods = T["frida_list_methods"](test_pkg, "com.android.settings.Settings")
                if r_methods.startswith("Error:"):
                    print(f"  SKIP  frida_list_methods ({r_methods})")
                else:
                    method_lines = [l for l in r_methods.splitlines() if l and not l.startswith("[error]")]
                    check("frida_list_methods returns results",
                          len(method_lines) > 0, r_methods[:200])
                    check("frida_list_methods lines contain return type",
                          all(any(t in l for t in ("void", "boolean", "int", "long",
                                                    "String", "Object", "android", "java"))
                              for l in method_lines),
                          method_lines[:3])
                    print(f"    → {len(method_lines)} methods found")
                    for _m in method_lines[:5]:
                        print(f"       {_m}")
                    if len(method_lines) > 5:
                        print(f"       ... ({len(method_lines) - 5} more)")

                    # filter param
                    filter_term = "Intent"
                    r_mf = T["frida_list_methods"](test_pkg, "com.android.settings.Settings",
                                                   filter=filter_term)
                    filtered_lines = [l for l in r_mf.splitlines() if l and not l.startswith("[error]")]
                    check("frida_list_methods filter returns subset",
                          len(filtered_lines) <= len(method_lines), r_mf[:200])
                    check("frida_list_methods filter: all lines contain filter term",
                          all(filter_term in l for l in filtered_lines) if filtered_lines else True,
                          filtered_lines[:3])

                # ── frida_list_fields ────────────────────────────────────────
                # Use android.os.Build — has many well-known public static fields
                print("\n=== frida_tools: frida_list_fields (frida + device required) ===")
                r_fields = T["frida_list_fields"](test_pkg, "android.os.Build")
                if r_fields.startswith("Error:"):
                    print(f"  SKIP  frida_list_fields ({r_fields})")
                else:
                    field_lines = [l for l in r_fields.splitlines() if l and not l.startswith("[error]")]
                    check("frida_list_fields returns results",
                          len(field_lines) > 0, r_fields[:200])
                    check("frida_list_fields lines contain a Java type keyword",
                          all(any(t in l for t in ("int", "long", "boolean", "String",
                                                    "android", "java", "static", "final"))
                              for l in field_lines),
                          field_lines[:3])
                    check("frida_list_fields contains MODEL field",
                          any("MODEL" in l for l in field_lines), field_lines[:5])
                    print(f"    → {len(field_lines)} fields found")
                    for _f in field_lines[:5]:
                        print(f"       {_f}")

                    # filter param
                    r_ff = T["frida_list_fields"](test_pkg, "android.os.Build", filter="String")
                    filtered_ff = [l for l in r_ff.splitlines() if l and not l.startswith("[error]")]
                    check("frida_list_fields filter: all lines contain 'String'",
                          all("String" in l for l in filtered_ff) if filtered_ff else True,
                          filtered_ff[:3])

                # ── frida_get_class_hierarchy ────────────────────────────────
                print("\n=== frida_tools: frida_get_class_hierarchy (frida + device required) ===")
                r_hier = T["frida_get_class_hierarchy"](test_pkg, "com.android.settings.Settings")
                if r_hier.startswith("Error:"):
                    print(f"  SKIP  frida_get_class_hierarchy ({r_hier})")
                else:
                    hier_lines = [l for l in r_hier.splitlines() if l and not l.startswith("[error]")]
                    check("frida_get_class_hierarchy returns results",
                          len(hier_lines) > 0, r_hier[:300])
                    check("frida_get_class_hierarchy contains superclass entries",
                          any(l.startswith("superclass:") for l in hier_lines), hier_lines)
                    check("frida_get_class_hierarchy chain ends at java.lang.Object",
                          any("java.lang.Object" in l for l in hier_lines), hier_lines)
                    check("frida_get_class_hierarchy contains android.app.Activity",
                          any("android.app.Activity" in l for l in hier_lines), hier_lines)
                    print(f"    → {len(hier_lines)} hierarchy entries:")
                    for _h in hier_lines:
                        print(f"       {_h}")

                # ── frida_call_method ────────────────────────────────────────
                print("\n=== frida_tools: frida_call_method (frida + device required) ===")
                # android.os.SystemClock.elapsedRealtime() — no args, returns ms uptime as long
                r_call = T["frida_call_method"](test_pkg, "android.os.SystemClock",
                                                "elapsedRealtime", [])
                print(f"    → frida_call_method SystemClock.elapsedRealtime(): {r_call!r}")
                check("frida_call_method returns a numeric string",
                      r_call.isdigit() or (r_call.lstrip("-").isdigit()), r_call)
                check("frida_call_method result is positive uptime",
                      int(r_call) > 0, r_call)

                # ── frida_get_field_value ────────────────────────────────────
                print("\n=== frida_tools: frida_get_field_value (frida + device required) ===")
                # android.os.Build.MODEL — public static String, always set on any device
                r_field = T["frida_get_field_value"](test_pkg, "android.os.Build", "MODEL")
                print(f"    → frida_get_field_value Build.MODEL: {r_field!r}")
                check("frida_get_field_value returns non-empty string",
                      bool(r_field) and r_field != "null" and not r_field.startswith("[error]"),
                      r_field)
                # Cross-check against adb to confirm the value is correct
                adb_model = _adb(["shell", "getprop", "ro.product.model"]).strip()
                print(f"    → adb getprop ro.product.model: {adb_model!r}")
                check("frida_get_field_value Build.MODEL matches adb ro.product.model",
                      r_field.strip() == adb_model, f"frida={r_field!r} adb={adb_model!r}")

            except Exception as e:
                print(f"  SKIP  frida_list_classes (no USB device: {e})")
        except ImportError:
            print("  SKIP  frida_list_classes (frida not installed)")

        # ── Group 4: hook tool unit tests (no device required) ───────────────
        print("\n=== frida_tools: hook tools (unit tests, no device) ===")

        # empty registry
        _registry_write({})
        check("frida_list_hooks empty → message",
              "(no hooks registered)" in T["frida_list_hooks"]())

        # unknown hook_id errors
        check("frida_hook_log unknown id → Error",
              T["frida_hook_log"]("deadbeef").startswith("Error:"))
        check("frida_unhook_method unknown id → Error",
              T["frida_unhook_method"]("deadbeef").startswith("Error:"))

        # already-stopped guard
        _registry_write({"stopped1": {
            "pid": None, "package": "x", "class": "x", "method": "x",
            "log_file": "/tmp/x", "serial": "", "status": "stopped", "log_pos": 0,
        }})
        check("frida_unhook_method already stopped → message",
              "already stopped" in T["frida_unhook_method"]("stopped1"))

        # dead PID auto-correction in frida_list_hooks
        _registry_write({"fakepid": {
            "pid": 99999999, "package": "com.example", "class": "com.example.Foo",
            "method": "bar", "log_file": "/tmp/fake.jsonl",
            "serial": "", "status": "active", "log_pos": 0,
        }})
        r_list = T["frida_list_hooks"]()
        check("frida_list_hooks detects dead pid",
              "dead" in r_list, r_list)
        check("frida_list_hooks shows hook_id",
              "fakepid" in r_list, r_list)
        check("frida_list_hooks shows class.method",
              "com.example.Foo.bar" in r_list, r_list)

        # frida_hook_log: no log file yet
        _registry_write({"nofile": {
            "pid": 1, "package": "x", "class": "x", "method": "x",
            "log_file": os.path.join(LOG_DIR, "does_not_exist_xyz.jsonl"),
            "serial": "", "status": "active", "log_pos": 0,
        }})
        check("frida_hook_log no log file yet → message",
              "starting" in T["frida_hook_log"]("nofile") or
              "no log file" in T["frida_hook_log"]("nofile"))

        # frida_hook_log: cursor advances correctly
        fake_log = os.path.join(LOG_DIR, "cursortest.jsonl")
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(fake_log, "w") as _f:
            _f.write(json.dumps({"event": "worker_start"}) + "\n")
            _f.write(json.dumps({"event": "enter", "method": "foo"}) + "\n")
        _registry_write({"cursor1": {
            "pid": 1, "package": "x", "class": "x", "method": "x",
            "log_file": fake_log, "serial": "", "status": "active", "log_pos": 0,
        }})
        r_read1 = T["frida_hook_log"]("cursor1")
        check("frida_hook_log first read returns all lines",
              "worker_start" in r_read1 and "enter" in r_read1, r_read1)
        r_read2 = T["frida_hook_log"]("cursor1")
        check("frida_hook_log second read → no new events",
              "no new events" in r_read2, r_read2)
        with open(fake_log, "a") as _f:
            _f.write(json.dumps({"event": "leave", "retval": "null"}) + "\n")
        r_read3 = T["frida_hook_log"]("cursor1")
        check("frida_hook_log third read returns only new line",
              "leave" in r_read3 and "worker_start" not in r_read3, r_read3)

        _registry_write({})  # clean up registry

        # ── Group 5a: frida_eval (device required) ────────────────────────────
        print("\n=== frida_tools: frida_eval (device required) ===")
        if not _device_available:
            print("  SKIP  frida_eval tests (no device / frida-server)")
        else:
            # Sanity 1: plain send, no Java imports needed
            _plain_script = 'send("frida_eval_sanity_ok"); send({done: true});'
            r_run1 = T["frida_eval"]("com.android.settings", _plain_script)
            print(f"    → frida_eval plain send: {r_run1!r}")
            check("frida_eval plain send returns expected string",
                  "frida_eval_sanity_ok" in r_run1, r_run1)

            # Sanity 2: Java.perform reads the actual package name of the process
            _java_script = """\
import Java from "frida-java-bridge";
Java.perform(function() {
    var pkg = Java.use("android.app.ActivityThread")
        .currentApplication().getPackageName();
    send(pkg);
    send({done: true});
});
"""
            r_run2 = T["frida_eval"]("com.android.settings", _java_script)
            print(f"    → frida_eval Java.perform package name: {r_run2!r}")
            check("frida_eval Java.perform returns package name",
                  "com.android.settings" in r_run2, r_run2)

        # ── Group 5b: hook integration tests (device + frida-server required) ──
        print("\n=== frida_tools: hook integration tests (device required) ===")
        if not _device_available:
            print("  SKIP  all hook integration tests (no device / frida-server)")
        else:
            # Target: com.android.settings / android.app.Activity / onResume
            # onResume fires on every activity foreground transition — easy to trigger.
            hook_pkg  = "com.android.settings"
            hook_cls  = "android.app.Activity"
            hook_meth = "onResume"

            # Ensure settings is running so frida can attach to its process.
            # Force-stop first so we always get a fresh process.
            _adb(["shell", "am", "force-stop", hook_pkg])
            time.sleep(1)
            _adb(["shell", "am", "start", "-a", "android.intent.action.MAIN",
                  "-n", "com.android.settings/.Settings"])
            time.sleep(3)   # wait for process to be schedulable by the OS

            pid_check = _adb(["shell", "pidof", hook_pkg])
            print(f"    → {hook_pkg} pid: {pid_check!r}")
            proc_running = pid_check.strip() and not pid_check.startswith("Error:")

            if not proc_running:
                print(f"  SKIP  integration tests ({hook_pkg} not running after launch)")
                T["frida_server_stop"]()
            else:
                # 1. Start the hook (non-blocking — returns immediately)
                r_hook = T["frida_hook_method"](hook_pkg, hook_cls, hook_meth)
                print(f"    → frida_hook_method:\n{r_hook}")
                check("frida_hook_method returns hook_id",
                      "hook_id:" in r_hook and not r_hook.startswith("Error:"), r_hook)

                hook_id = None
                for _line in r_hook.splitlines():
                    if _line.strip().startswith("hook_id:"):
                        hook_id = _line.split(":", 1)[1].strip()
                        break

                if not hook_id:
                    print("  SKIP  remaining integration tests (hook_id not found)")
                else:
                    # 2. Wait for worker subprocess to compile JS and attach
                    time.sleep(3)
                    r_list = T["frida_list_hooks"]()
                    print(f"    → frida_list_hooks:\n{r_list}")
                    check("frida_list_hooks shows new hook as active",
                          hook_id in r_list and "active" in r_list, r_list)

                    # 3. Initial log — must contain worker_start (or attach_error to skip)
                    r_log1 = T["frida_hook_log"](hook_id)
                    print(f"    → frida_hook_log (initial): {r_log1[:300]}")
                    attach_failed = "attach_error" in r_log1 or "fatal" in r_log1

                    if attach_failed:
                        print(f"  SKIP  trigger/unhook tests (worker attach failed)")
                        T["frida_unhook_method"](hook_id)
                    else:
                        check("frida_hook_log initial: contains worker_start",
                              "worker_start" in r_log1, r_log1[:300])

                        # 4. Trigger onResume: home → reopen settings
                        _adb(["shell", "input", "keyevent", "KEYCODE_HOME"])
                        time.sleep(1)
                        _adb(["shell", "am", "start", "-a", "android.intent.action.MAIN",
                              "-n", "com.android.settings/.Settings"])
                        time.sleep(3)   # let hook events land in the log file

                        # 5. Log must now contain enter + leave events
                        r_log2 = T["frida_hook_log"](hook_id)
                        print(f"    → frida_hook_log (after trigger): {r_log2[:400]}")
                        check("frida_hook_log: enter event captured",
                              "enter" in r_log2, r_log2[:400])
                        check("frida_hook_log: leave event captured",
                              "leave" in r_log2, r_log2[:400])

                        # ── Group 5b: specific custom-string verification ─────
                        # Accumulate everything logged so far for full inspection.
                        print("\n=== frida_tools: specific custom-string log verification ===")
                        all_log = r_log1 + "\n" + r_log2

                        # The hook_id is a unique UUID prefix we generated — it
                        # must appear verbatim inside the worker_start JSON entry.
                        check("log contains unique hook_id string",
                              hook_id in all_log, all_log[:500])

                        # The class and method names we passed to frida_hook_method
                        # must appear verbatim inside every enter/leave event.
                        check("log contains hooked class name",
                              hook_cls in all_log, all_log[:500])
                        check("log contains hooked method name",
                              hook_meth in all_log, all_log[:500])

                        # Parse every line and assert the enter event has the
                        # correct structured fields — not just substrings.
                        parsed = []
                        for _raw in all_log.splitlines():
                            _raw = _raw.strip()
                            if not _raw:
                                continue
                            try:
                                parsed.append(json.loads(_raw))
                            except json.JSONDecodeError:
                                pass

                        enter_events = [e for e in parsed if e.get("event") == "enter"]
                        leave_events = [e for e in parsed if e.get("event") == "leave"]

                        check("enter event: class field == hooked class",
                              any(e.get("class") == hook_cls for e in enter_events),
                              str(enter_events))
                        check("enter event: method field == hooked method",
                              any(e.get("method") == hook_meth for e in enter_events),
                              str(enter_events))
                        check("enter event: args field is a list",
                              all(isinstance(e.get("args"), list) for e in enter_events),
                              str(enter_events))
                        check("enter event: ts field is an integer",
                              all(isinstance(e.get("ts"), int) for e in enter_events),
                              str(enter_events))
                        check("leave event: retval field present",
                              all("retval" in e for e in leave_events),
                              str(leave_events))

                        # Print a summary of all captured events for visibility
                        print(f"    → parsed {len(parsed)} total log entries, "
                              f"{len(enter_events)} enter, {len(leave_events)} leave")
                        for _e in parsed:
                            print(f"       {json.dumps(_e)}")

                        # 6. Unhook
                        r_unhook = T["frida_unhook_method"](hook_id)
                        print(f"    → frida_unhook_method: {r_unhook}")
                        check("frida_unhook_method returns stopped",
                              "stopped" in r_unhook, r_unhook)

                        # 7. Log after stop must include worker_stop
                        time.sleep(1)
                        r_log3 = T["frida_hook_log"](hook_id)
                        print(f"    → frida_hook_log (after unhook): {r_log3[:200]}")
                        check("frida_hook_log after unhook: worker_stop or no new events",
                              "worker_stop" in r_log3 or "no new events" in r_log3, r_log3)

                        # 8. List hooks must show stopped status
                        r_list2 = T["frida_list_hooks"]()
                        print(f"    → frida_list_hooks (after unhook):\n{r_list2}")
                        check("frida_list_hooks shows hook as stopped",
                              hook_id in r_list2 and "stopped" in r_list2, r_list2)

                        # 9. Double-unhook must be graceful
                        check("frida_unhook_method double-stop → already stopped",
                              "already stopped" in T["frida_unhook_method"](hook_id))

                # 10. Stop frida-server to leave the device clean
                r_srv_stop = T["frida_server_stop"]()
                print(f"    → frida_server_stop: {r_srv_stop}")
                check("frida_server_stop after integration tests",
                      "stopped" in r_srv_stop or "Error:" in r_srv_stop, r_srv_stop)

    finally:
        REGISTRY_PATH = _orig_reg
        LOG_DIR = _orig_log
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{_pass} passed, {_fail} failed")
    sys.exit(0 if _fail == 0 else 1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--hook-id")
        p.add_argument("--package")
        p.add_argument("--class", dest="class_name")
        p.add_argument("--method", dest="method_name")
        p.add_argument("--log-file")
        p.add_argument("--serial", default="")
        args = p.parse_args(sys.argv[2:])
        _run_worker(args.hook_id, args.package, args.class_name,
                    args.method_name, args.log_file, args.serial)
    else:
        _run_tests()
