import subprocess

from cai.utils import safe_path


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


def register(mcp):
    @mcp.tool()
    def adb_devices() -> str:
        """List all Android devices/emulators currently connected via ADB.

        Returns one device per line: <serial>  <state>
        State is usually 'device' (ready), 'offline', or 'unauthorized'.
        Returns an error string if adb is unavailable.
        """
        return _adb(["devices", "-l"])

    @mcp.tool()
    def adb_shell(command: str, serial: str = "") -> str:
        """Run a shell command on an Android device via 'adb shell'.

        Args:
            command: Shell command to execute on the device, e.g. "ls /sdcard",
                     "getprop ro.build.version.release", "pm list packages".
            serial:  Device serial number (from adb_devices). Required when
                     more than one device is connected; leave empty otherwise.

        Returns:
            Command output, or an error string prefixed with "Error:".
        """
        prefix = ["-s", serial] if serial else []
        return _adb(prefix + ["shell", command], timeout=60)

    @mcp.tool()
    def adb_root_shell(command: str, serial: str = "") -> str:
        """Run a shell command as root on an Android device via 'adb shell su -c'.

        Requires the device to be rooted (su binary present). Does not restart
        adbd — instead runs the command through su, so it works on both
        production and eng builds that have su available.

        Args:
            command: Shell command to execute as root, e.g. "cat /data/data/com.example/databases/db",
                     "chmod 777 /data/local/tmp/test", "setenforce 0".
            serial:  Device serial number (from adb_devices). Required when
                     more than one device is connected; leave empty otherwise.

        Returns:
            Command output, or an error string prefixed with "Error:".
        """
        prefix = ["-s", serial] if serial else []
        return _adb(prefix + ["shell", "su", "-c", command], timeout=60)

    @mcp.tool()
    def adb_install(apk_path: str, serial: str = "", replace: bool = True) -> str:
        """Install an APK on an Android device.

        Args:
            apk_path: Path to the .apk file on the host machine.
            serial:   Device serial number. Leave empty if only one device is connected.
            replace:  If True (default), pass -r to replace an existing installation.

        Returns:
            "Success" on success, or an error string.
        """
        try:
            apk_path = safe_path(apk_path)
        except ValueError as e:
            return str(e)
        prefix = ["-s", serial] if serial else []
        flags = ["-r"] if replace else []
        return _adb(prefix + ["install"] + flags + [apk_path], timeout=120)

    @mcp.tool()
    def adb_uninstall(package: str, serial: str = "", keep_data: bool = False) -> str:
        """Uninstall an app by package name from an Android device.

        Args:
            package:    Package name, e.g. "com.example.app".
            serial:     Device serial number. Leave empty if only one device is connected.
            keep_data:  If True, pass -k to keep app data and cache after uninstall.

        Returns:
            "Success" on success, or an error string.
        """
        prefix = ["-s", serial] if serial else []
        flags = ["-k"] if keep_data else []
        return _adb(prefix + ["uninstall"] + flags + [package])

    @mcp.tool()
    def adb_pull(device_path: str, local_path: str = ".", serial: str = "") -> str:
        """Pull (download) a file or directory from the device to the host.

        Args:
            device_path: Absolute path on the device, e.g. "/sdcard/DCIM/photo.jpg".
            local_path:  Destination path on the host. Defaults to current directory.
            serial:      Device serial number. Leave empty if only one device is connected.

        Returns:
            Transfer summary, or an error string.
        """
        try:
            local_path = safe_path(local_path)
        except ValueError as e:
            return str(e)
        prefix = ["-s", serial] if serial else []
        return _adb(prefix + ["pull", device_path, local_path], timeout=120)

    @mcp.tool()
    def adb_push(local_path: str, device_path: str, serial: str = "") -> str:
        """Push (upload) a file or directory from the host to the device.

        Args:
            local_path:  Path on the host to the file or directory to upload.
            device_path: Destination absolute path on the device, e.g. "/sdcard/test.txt".
            serial:      Device serial number. Leave empty if only one device is connected.

        Returns:
            Transfer summary, or an error string.
        """
        try:
            local_path = safe_path(local_path)
        except ValueError as e:
            return str(e)
        prefix = ["-s", serial] if serial else []
        return _adb(prefix + ["push", local_path, device_path], timeout=120)

    @mcp.tool()
    def adb_logcat(tag: str = "", lines: int = 100, serial: str = "") -> str:
        """Capture recent logcat output from an Android device.

        Uses 'adb logcat -d' (dump and exit) to capture the last N lines.

        Args:
            tag:    Optional logcat tag filter, e.g. "ActivityManager". Leave empty
                    for all tags.
            lines:  Maximum number of log lines to return (default 100, max 2000).
            serial: Device serial number. Leave empty if only one device is connected.

        Returns:
            Logcat lines as a string, or an error string.
        """
        lines = min(lines, 2000)
        prefix = ["-s", serial] if serial else []
        args = prefix + ["logcat", "-d"]
        if tag:
            args += ["-s", f"{tag}:V"]
        try:
            result = subprocess.run(
                ["adb"] + args,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=30
            )
            out = result.stdout.strip()
            if result.returncode != 0:
                return f"Error: {result.stderr.strip() or out}"
            tail = "\n".join(out.splitlines()[-lines:])
            return tail or "(no log output)"
        except FileNotFoundError:
            return "Error: adb is not installed or not on PATH."
        except subprocess.TimeoutExpired:
            return "Error: adb logcat timed out."

    @mcp.tool()
    def adb_getprop(prop: str = "", serial: str = "") -> str:
        """Read Android system properties via 'adb shell getprop'.

        Args:
            prop:   Specific property name to read, e.g. "ro.build.version.release".
                    Leave empty to dump all properties.
            serial: Device serial number. Leave empty if only one device is connected.

        Returns:
            Property value(s), or an error string.
        """
        prefix = ["-s", serial] if serial else []
        cmd = ["getprop"] + ([prop] if prop else [])
        return _adb(prefix + ["shell"] + cmd)

    @mcp.tool()
    def adb_screencap(local_path: str, serial: str = "") -> str:
        """Take a screenshot on the device and pull it to the host.

        Captures /sdcard/_screencap_tmp.png on the device, pulls it to
        local_path, then removes the temporary file from the device.

        Args:
            local_path: Destination path on the host, e.g. "/tmp/screen.png".
            serial:     Device serial number. Leave empty if only one device is connected.

        Returns:
            "Saved to <local_path>" on success, or an error string.
        """
        try:
            local_path = safe_path(local_path)
        except ValueError as e:
            return str(e)
        prefix = ["-s", serial] if serial else []
        tmp = "/sdcard/_screencap_tmp.png"
        out = _adb(prefix + ["shell", "screencap", "-p", tmp])
        if out.startswith("Error"):
            return out
        out = _adb(prefix + ["pull", tmp, local_path], timeout=30)
        if out.startswith("Error"):
            return out
        _adb(prefix + ["shell", "rm", tmp])
        return f"Saved to {local_path}"

    @mcp.tool()
    def adb_forward(host_port: int, device_port: int, serial: str = "") -> str:
        """Forward a TCP port from the host to the device (adb forward).

        Args:
            host_port:   Local port on the host machine.
            device_port: Port on the Android device to forward to.
            serial:      Device serial number. Leave empty if only one device is connected.

        Returns:
            The forwarded port number, or an error string.
        """
        prefix = ["-s", serial] if serial else []
        return _adb(prefix + ["forward", f"tcp:{host_port}", f"tcp:{device_port}"])

    @mcp.tool()
    def adb_package_list(filter: str = "", serial: str = "") -> str:
        """List installed packages on an Android device.

        Args:
            filter: Optional substring to filter package names, e.g. "google".
                    Leave empty to list all packages.
            serial: Device serial number. Leave empty if only one device is connected.

        Returns:
            One package name per line, or an error string.
        """
        prefix = ["-s", serial] if serial else []
        out = _adb(prefix + ["shell", "pm", "list", "packages"])
        if out.startswith("Error"):
            return out
        packages = [line.removeprefix("package:") for line in out.splitlines()]
        if filter:
            packages = [p for p in packages if filter in p]
        return "\n".join(packages) or "(no packages matched)"


if __name__ == "__main__":
    import sys

    class _MockMCP:
        def __init__(self): self._tools = {}
        def tool(self):
            def dec(fn): self._tools[fn.__name__] = fn; return fn
            return dec

    mcp = _MockMCP()
    register(mcp)
    T = mcp._tools

    _pass = _fail = 0
    def check(name, cond, got=""):
        global _pass, _fail
        if cond:
            print(f"  PASS  {name}")
            _pass += 1
        else:
            print(f"  FAIL  {name}  →  {got!r}")
            _fail += 1

    print("=== adb_tools tests ===")

    # adb_devices: always exits 0 regardless of connected devices
    r = T["adb_devices"]()
    if r.startswith("Error: adb is not installed"):
        print(f"  SKIP  all adb runtime tests (adb not on PATH)")
    else:
        check("adb_devices returns header", "List of devices" in r, r)

        # adb_shell / adb_root_shell / adb_getprop / adb_package_list / adb_logcat:
        # with no device they return a recognisable error; with a device they return output.
        # Either outcome is acceptable — we just check there's no Python exception.
        for tool_name, kwargs in [
            ("adb_shell",      {"command": "echo hello"}),
            ("adb_root_shell", {"command": "id"}),
            ("adb_getprop",    {"prop": "ro.build.version.release"}),
            ("adb_package_list", {}),
            ("adb_logcat",     {"lines": 5}),
        ]:
            try:
                r = T[tool_name](**kwargs)
                check(f"{tool_name} no crash", True)
            except Exception as e:
                check(f"{tool_name} no crash", False, str(e))

        # adb_forward: no device → error, but no Python exception
        try:
            r = T["adb_forward"](12345, 12345)
            check("adb_forward no crash", True)
        except Exception as e:
            check("adb_forward no crash", False, str(e))

    # --- safe_path rejection tests (no device needed) ---
    r = T["adb_install"]("../../evil.apk")
    check("adb_install traversal rejected", r.startswith("Error:"), r)

    r = T["adb_pull"]("/sdcard/file.txt", "../../escape")
    check("adb_pull local_path traversal rejected", r.startswith("Error:"), r)

    r = T["adb_push"]("../../escape", "/sdcard/file.txt")
    check("adb_push local_path traversal rejected", r.startswith("Error:"), r)

    r = T["adb_screencap"]("../../escape.png")
    check("adb_screencap local_path traversal rejected", r.startswith("Error:"), r)

    print(f"\n{_pass} passed, {_fail} failed")
    sys.exit(0 if _fail == 0 else 1)
