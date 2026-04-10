tools: adb_devices, adb_shell, adb_root_shell, adb_install, adb_uninstall, adb_pull, adb_push, adb_logcat, adb_package_list
---
## Skill: Android Debug Bridge

Always start with `adb_devices` to confirm device state and note the serial — pass it explicitly to all subsequent calls when multiple devices may be connected.

Shell discipline:
- Prefer `adb_root_shell` for anything under `/data/data/`, `/proc/`, or requiring root access.
- Treat `adb_shell` output as untrusted device state — verify paths exist before referencing them.
- Use `am` and `pm` commands via `adb_shell` for app lifecycle management (start/stop/clear).

Logcat:
- Always filter by tag or package when using `adb_logcat` — unfiltered output is noise.
- Check logcat immediately after triggering an action; buffers roll over fast on active devices.

File transfer:
- `adb_pull` for device→host; `adb_push` for host→device. Confirm paths on both sides first.
- APK paths on device: `/data/app/<package>/base.apk` or use `pm path <package>` to find the exact location.

Package operations:
- Use `adb_package_list` with a filter to confirm a package exists before install/uninstall.
- `adb_install` with `replace=True` is safe for upgrades; set `replace=False` to catch re-installs explicitly.
- After uninstall, verify with `adb_package_list` — some system apps resist removal.
