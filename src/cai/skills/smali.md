tools: smali_find_callers, smali_find_callees, smali_resolve_descriptor, smali_find_implementations, search, read_file, list_files
---
## Skill: Smali Analysis

Before touching anything: map the class hierarchy, identify entry points, and trace the relevant data flows. State your methodology before conclusions.

Register discipline:
- Distinguish v* (local) from p* (parameter) registers at every step.
- Map register assignments across the full method before proposing any edit.
- Note every `move-result` and `move-result-object` — these are the only way to capture invoke return values.

Editing rules:
- Preserve all label names and their exact positions — branches and try/catch blocks depend on them.
- Preserve alignment; dalvik's verifier is strict about it.
- Understand the dalvik/art execution model before patching control flow: method descriptors, invoke-kind semantics (virtual vs direct vs static vs interface), and type correctness all matter.
- After any edit, mentally trace the modified path to confirm register types are still consistent.

Analysis workflow:
- Use `smali_resolve_descriptor` to look up types before assuming class relationships.
- Use `smali_find_callers` / `smali_find_callees` to map call graphs before concluding about reachability.
- Use `smali_find_implementations` to find concrete classes behind interfaces.
