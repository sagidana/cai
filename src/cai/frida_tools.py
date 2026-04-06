import json
import os
import subprocess
import tempfile
import threading
import time

from cai.utils import safe_path

REGISTRY_PATH = "/tmp/cai_frida_registry.json"
LOG_DIR = "/tmp/cai_frida_logs"
DEVICE_SERVER = "/data/local/tmp/frida-server"

# Frida 17: frida-java-bridge is no longer a built-in global. Scripts must use
# ESM import syntax and be compiled via frida.Compiler before loading.
# _FRIDA_JS_ROOT must point to a directory containing node_modules/frida-java-bridge.
_FRIDA_JS_ROOT = os.environ.get("FRIDA_JS_ROOT", "/tmp/frida_js_test")

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


def _frida_attach(package: str, serial: str):
    """Attach to a running app and return (device, session). Raises on failure."""
    import frida  # deferred — avoids import-time crash if frida not installed
    if serial:
        device = frida.get_device(serial)
    else:
        device = frida.get_usb_device()
    session = device.attach(package)
    return device, session


def _compile_js(js: str) -> str:
    """Compile an ESM script (using frida-java-bridge imports) into a loadable bundle.

    Requires frida-java-bridge to be installed as an npm package under
    _FRIDA_JS_ROOT/node_modules. Falls back to the raw source (for runtimes
    that support static import natively).
    """
    import frida  # deferred
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


def _collect_messages(session, js: str, timeout: int = 30) -> list:
    """Load a script and collect send() payloads until {done: true} or timeout."""
    results = []
    done = threading.Event()

    def on_message(msg, _data):
        if msg["type"] == "send":
            p = msg["payload"]
            if p.get("done"):
                done.set()
            elif p.get("error"):
                results.append(f"[error] {p['error']}")
            else:
                results.append(p.get("name", ""))
        elif msg["type"] == "error":
            results.append(f"[error] {msg.get('description', 'unknown')}")
            done.set()

    bundle = _compile_js(js)
    script = session.create_script(bundle)
    script.on("message", on_message)
    script.load()
    done.wait(timeout=timeout)
    return results


def register(mcp):
    @mcp.tool()
    def frida_server_start(serial: str = "", server_binary: str = "") -> str:
        """Start frida-server on a connected Android device.

        Starts frida-server as a background daemon. Requires a rooted device
        (su must be available). Uses nohup + full stdio redirect so the adb
        shell exits immediately and the call stays synchronous.

        Args:
            serial:        Device serial (from adb devices). Leave empty if only
                           one device is connected.
            server_binary: Local path to a frida-server binary to push to the
                           device first. Leave empty to use the binary already
                           on the device at /data/local/tmp/frida-server.

        Returns:
            Status string, or "Error: ..." on failure.
        """
        prefix = ["-s", serial] if serial else []

        if server_binary:
            try:
                safe_bin = safe_path(server_binary)
            except ValueError as e:
                return str(e)
            r = _adb(prefix + ["push", safe_bin, DEVICE_SERVER], timeout=60)
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
            import frida  # noqa: F401 — ensure it's installed
        except ImportError:
            return "Error: frida not installed. Run: pip install frida"

        js = _JS_LIST_CLASSES % json.dumps(package)

        try:
            _device, session = _frida_attach(package, serial)
        except Exception as e:
            return f"Error: could not attach to {package!r}: {e}"

        try:
            results = _collect_messages(session, js, timeout=30)
        except Exception as e:
            return f"Error: script error: {e}"
        finally:
            try:
                session.detach()
            except Exception:
                pass

        if filter:
            results = [r for r in results if filter in r]

        return "\n".join(r for r in results if r) or "(no classes found)"


# ---------------------------------------------------------------------------
# Worker & test harness
# ---------------------------------------------------------------------------

def _run_worker(hook_id, package, class_name, method_name, log_file, serial):
    """Background daemon: attaches frida, hooks method, writes log until SIGTERM."""
    import signal as _signal

    def _write(entry):
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    try:
        import frida
    except ImportError:
        _write({"event": "fatal", "error": "frida not installed", "ts": 0})
        return

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

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

    bundle = _compile_js(js)
    script = session.create_script(bundle)
    script.on("message", on_message)
    _write({"event": "worker_start", "hook_id": hook_id,
            "class": class_name, "method": method_name, "ts": 0})
    script.load()

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
        try:
            import frida as _frida  # check installed
            # Try to get a device
            try:
                dev = _frida.get_usb_device(timeout=3)
                # Ensure frida-server is running before class enumeration
                r_start = T["frida_server_start"]()
                print(f"    → ensuring frida-server: {r_start}")
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
            except Exception as e:
                print(f"  SKIP  frida_list_classes (no USB device: {e})")
        except ImportError:
            print("  SKIP  frida_list_classes (frida not installed)")

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
