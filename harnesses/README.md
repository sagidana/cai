# harness examples

A collection of ready-to-use `.harness.cai` orchestration files for common
software engineering tasks.

## Usage

```
cai --harness harnesses/<name>.harness.cai -- "describe your task here"
```

---

## Harnesses

### `context-and-execute.harness.cai`
**The canonical pattern.** Gather context → verify sufficiency → execute.

Best for any task where executing without full context risks wrong output:
refactors, feature additions, edits to unfamiliar code.

```
cai --harness harnesses/context-and-execute.harness.cai -- "add pagination to the user list endpoint"
```

---

### `bug-fix.harness.cai`
**Systematic bug fixing with a self-review loop.**

Stages: gather → verify → fix → self-review → summary.
The self-review loop catches mistakes before declaring done.

```
cai --harness harnesses/bug-fix.harness.cai -- "fix the NoneType crash in auth.py when the token has expired"
```

---

### `code-review.harness.cai`
**Thorough code review with branching verdict reports.**

Stages: gather → verify → deep analysis → severity classification → report.
Branches into blocking / suggestions / clean report formats based on findings.

```
cai --harness harnesses/code-review.harness.cai -- "review the changes in src/payments/"
```

---

### `test-writer.harness.cai`
**Comprehensive test suite generation with a plan-validation loop.**

Stages: gather source → gather test patterns → plan → validate plan → write.
The plan is validated for completeness before a single test is written.

```
cai --harness harnesses/test-writer.harness.cai -- "write tests for src/cai/harness.py"
```

---

### `refactor.harness.cai`
**Safe, structured refactoring with two safety loops.**

Stages: gather → verify → plan → validate plan → execute → sanity check → summary.
The plan-validate loop catches design mistakes; the sanity loop catches regressions.

```
cai --harness harnesses/refactor.harness.cai -- "extract the retry logic in api.py into a reusable decorator"
```

---

### `feature.harness.cai`
**Disciplined feature implementation: design first, implement second.**

Stages: gather → verify → design → validate design → implement → write tests.
The design is validated for consistency with the codebase before any code is written.

```
cai --harness harnesses/feature.harness.cai -- "add rate limiting to all public API endpoints"
```

---

### `security-audit.harness.cai`
**Systematic security audit with threat modelling and severity-branched reports.**

Stages: gather → verify → threat model → deep audit → severity classification → report.
Produces different report formats for critical / high / medium / low / clean findings.

```
cai --harness harnesses/security-audit.harness.cai -- "audit the authentication and file upload modules"
```

---

### `migrate.harness.cai`
**Safe dependency or API migration with a complete usage inventory.**

Stages: gather → audit all usages → validate audit → plan → execute → verify completeness → summary.
The audit stage catalogues every callsite before touching any code.

```
cai --harness harnesses/migrate.harness.cai -- "migrate from requests to httpx throughout the codebase"
```

---

### `explain.harness.cai`
**Deep, accurate code explanation with a structured trace step.**

Stages: gather → verify → trace execution/data flow → write layered explanation.
The trace step builds a precise structural model before narrating it for a reader.

```
cai --harness harnesses/explain.harness.cai -- "explain how request authentication works end to end"
```

---

### `document.harness.cai`
**Accurate API documentation that matches project conventions.**

Stages: gather code → gather doc conventions → outline → validate outline → write.
Studies existing docs before writing so output matches the project's format exactly.

```
cai --harness harnesses/document.harness.cai -- "document the public API of src/cai/harness.py"
```

---

### `web-search.harness.cai`
**Generic web research with automatic task decomposition and synthesis.**

Handles any information-gathering task by breaking it into 3–5 focused
sub-questions, researching each independently via web search, then merging
all findings into one comprehensive answer.

The only tool used is `fetch_url`. Web search is performed by fetching
DuckDuckGo's HTML endpoint (`html.duckduckgo.com/html/?q=…`) and then
reading the most relevant result pages. Each sub-question runs in an
isolated context so quality does not degrade with topic breadth.

Stages: decompose → (per sub-question: search → verify → report) → synthesize.

```
cai --harness harnesses/web-search.harness.cai -- "what are the trade-offs between PostgreSQL and MongoDB for a high-write SaaS product?"
cai --harness harnesses/web-search.harness.cai -- "what happened at the 2024 Paris Olympics?"
cai --harness harnesses/web-search.harness.cai -- "explain how transformer attention mechanisms work"
```

The sub-harness (`web-search-fetch.harness.cai`) is invoked automatically
for each sub-question via `for-each`. It can also be run directly to
research a single focused question:

```
cai --harness harnesses/web-search-fetch.harness.cai -- "what is the current Python GIL removal status?"
```

---

## Format Reference

```
# Comments start with #

label:               # jump target (word followed by colon)

---                  # opens a block
    --name "x"               # required: block name for branching
    --enrich-global-context  # or --dont-enrich-global-context (required)
    --prepend-user-prompt    # prepend user task to this block's prompt
    --tools read, list_files # internal tool names (comma or space separated)
    --model gpt-4o           # override model for this block
    --max-turns 100          # override max tool-call turns
    --strict-format "regex:^(ok|retry)$"  # enforce output format
    --system-prompt "..."    # block-specific system prompt
    --force-tools            # require at least one tool call
    '''
    Prompt text goes here.
    Multiple lines are fine.
    '''
---

if x == ok: goto label       # conditional jump (exact string match)
goto label                   # unconditional jump
exit                         # terminate harness
compact-if-more-than <percentage>           # compact global context if usage exceeds <percentage>% of window
if-more-than <number> <label>               # goto label if this point has been passed more than <number> times
for-each <item> in <block>: harness "<path>"  # run sub-harness for each line of block output
```

### Context enrichment

- `--enrich-global-context`: after the block runs, its **full message history**
  (user prompt, all tool calls, tool results, assistant turns) is added to the
  global context that subsequent blocks receive. Use for blocks that gather
  information others need.

- `--dont-enrich-global-context`: the block's messages are discarded after it
  runs. Use for quality-gate blocks (verify, classify) whose deliberation is
  not useful to subsequent blocks.

### `compact-if-more-than <percentage>`

Placed anywhere in the control flow (typically before a block that will add
a large new exchange, or at the top of a retry loop), this command checks
whether the accumulated `global_messages` are consuming more than
`<percentage>%` of the model's context window. If they are, it summarises
the middle turns into a single `[memory]` entry, preserving the first exchange
and the last four messages verbatim. If not, it is a no-op.

```
enrichment:
compact-if-more-than 30   # trim context before re-running enrichment
---
    --name "enrichment"
    ...
---
```

Use it in retry loops where a block may run many times and accumulate large
amounts of tool output in the global context.

### `for-each <item> in <block>: harness "<path>"`

Runs a sub-harness once for each line of a named block's output. Each run is
fully isolated — it starts with a fresh context and receives one line as its
`user_prompt`, exactly as if the user had typed it on the command line.

After all iterations complete, a single structured message is injected into
the parent's `global_messages`:

```
[for-each results: decompose]
─── task: src/cai/harness.py
    → Tests written: 12 cases covering parse_harness_file and execute_harness
─── task: src/cai/cli.py
    → Tests written: 8 cases covering argument parsing and call_llm dispatch
```

Subsequent parent blocks (e.g. an `aggregate` block) see this message
naturally in context — no special API needed. The sub-harness and the parent's
later blocks are both completely unaware of the for-each mechanism.

```
---
    --name "decompose"
    --prepend-user-prompt
    --dont-enrich-global-context
    --system-prompt "Output one task per line. No bullets, no numbering."
    '''
    Break the user's task into independent atomic subtasks.
    Output one per line.
    '''
---

for-each task in decompose: harness "harnesses/context-and-execute.harness.cai"

---
    --name "aggregate"
    --dont-enrich-global-context
    '''
    All subtasks are complete. Summarise the overall result for the user.
    '''
---
```

See `harnesses/multi-task.harness.cai` for a complete working example.

