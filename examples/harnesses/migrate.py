"""Port of harnesses/migrate.harness.cai.

Six stages: gather → audit → validate-audit → plan → execute → verify → done.

Usage:
    python examples/harnesses/migrate.py "migrate from requests to httpx, replacing all usages"
"""
from cai import Harness
from _helpers import get_task, prepend_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search", "symbol_search", "fetch_url",
]
EXEC_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "edit_file", "create_file", "rename_file",
]


def main() -> None:
    task = get_task()
    h = Harness()

    # stage 1: gather + audit. loop up to 3 times until validate_audit says 'ok'.
    for _ in range(3):
        r = h.run_agent(
            tools=READ_TOOLS,
            system_prompt=(
                "You are a migration engineer in the context-gathering phase. Find "
                "and read every file that uses the old API being migrated. Miss a "
                "file and it becomes a silent regression."
            ),
            prompt=prepend_task(task, (
                "Gather the complete inventory of code that will need to change. "
                "Search for imports, usages, configuration, manifests, CI. Read "
                "every found file in full. Map old→new patterns. Do not change code yet."
            )),
        )
        r.wait()
        h.enrich(r.messages)

        r = h.run_agent(
            system_prompt=(
                "You are a migration engineer cataloguing every usage that needs to "
                "change. Every usage you miss becomes a production bug."
            ),
            prompt=(
                "Produce a migration audit: per-file inventory of usages "
                "(location, current pattern → new pattern), then SUMMARY, "
                "BEHAVIOUR CHANGES. Be exhaustive."
            ),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "incomplete"],
            "Is the migration audit complete and ready to drive a safe migration? "
            "'ok' only if every file/usage catalogued, config/tests included, "
            "behaviour differences documented.",
        )
        if verdict == "ok":
            break

    # stage 2: plan the migration sequence.
    r = h.run_agent(
        system_prompt=(
            "You are a migration engineer sequencing a safe migration. Order "
            "matters: migrate shared utilities before callers; tests last."
        ),
        prompt=(
            "Produce a step-by-step migration plan: MIGRATION SEQUENCE, "
            "DEPENDENCY CHANGES, BACKWARDS COMPATIBILITY, ROLLBACK."
        ),
    )
    r.wait()
    h.enrich(r.messages)

    # stage 3: execute. loop up to 3 times until verify says 'clean'.
    for _ in range(3):
        r = h.run_agent(
            tools=EXEC_TOOLS,
            system_prompt=(
                "You are a migration engineer executing an approved plan. Every "
                "usage in the audit must be migrated. Do not make changes not "
                "required by the migration."
            ),
            prompt=(
                "Execute the plan in order. For each file, migrate every listed "
                "usage. Update dependency manifests. Show complete updated code."
            ),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["clean", "missed"],
            "Cross-reference against the audit. 'clean' only if every file "
            "migrated, every usage replaced, manifests updated, no old usages "
            "remain, behaviour differences addressed.",
            system_prompt="You are a senior engineer verifying a migration is complete. Answer only 'clean' or 'missed'.",
        )
        if verdict == "clean":
            break

    # stage 4: done. final report.
    r = h.run_agent(
        system_prompt="You are a clear technical communicator writing a migration completion report.",
        prompt=(
            "Report: what was migrated, how many files/usages changed, behaviour "
            "changes and how handled, any workarounds, what to verify next."
        ),
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
