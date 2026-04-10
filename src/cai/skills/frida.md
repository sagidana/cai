tools: frida_server_start, frida_server_stop, frida_list_classes, frida_list_methods, frida_list_fields, frida_get_class_hierarchy, frida_call_method, frida_get_field_value, frida_eval, frida_hook_method, frida_unhook_method, frida_hook_log, frida_list_hooks, adb_devices, adb_shell
---
## Skill: Frida Dynamic Instrumentation

Workflow order: verify device/server → enumerate classes → enumerate methods → hook or eval.

Never assume a class or method exists — enumerate first with `frida_list_classes` / `frida_list_methods`. Class names in Frida use dot notation (`com.example.Foo`), not slash notation.

Hooking discipline:
- Understand what the method does before hooking it — read the Smali or use `frida_list_fields` to understand state.
- Prefer `frida_hook_method` for persistent hooks; use `frida_eval` for one-shot introspection.
- After hooking, use `frida_hook_log` to verify the hook is firing before drawing conclusions.
- Clean up with `frida_unhook_method` when done — stale hooks cause interference.
- Use `frida_list_hooks` to audit what's active before adding more.

Script safety:
- `frida_eval` scripts run in the app's process — avoid operations that crash the target (null derefs, uncaught exceptions in callbacks).
- Overloaded methods: always specify the full signature when multiple overloads exist.
- Instance vs static: `Java.use()` gives you the class; `.call(instance, ...)` or `.$new()` for constructors.

Server management:
- Check `adb_devices` before starting frida_server — confirm the target device is connected.
- If `frida_server_start` fails, check if the server binary matches the device architecture (arm, arm64, x86, x86_64).
