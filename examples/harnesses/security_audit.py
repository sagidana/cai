"""Port of harnesses/security-audit.harness.cai.

Five stages: gather → threat-model → audit → severity → report (branches on
severity: critical / high / medium / low / clean).

Usage:
    python examples/harnesses/security_audit.py "audit the authentication and payment modules"
"""
from cai import Harness
from _helpers import get_task, prepend_task


READ_TOOLS = [
    "read", "read_lines", "list_files", "pattern_search",
    "symbol_search", "file_code_outline",
]
AUDIT_TOOLS = ["read", "read_lines", "list_files", "pattern_search", "symbol_search"]


def main() -> None:
    task = get_task()
    h = Harness()

    # stage 1: gather. loop up to 3 times until verify says 'ok'.
    for _ in range(3):
        r = h.run_agent(
            tools=READ_TOOLS,
            system_prompt=(
                "You are an expert security engineer performing a code audit. Read "
                "every file that handles user input, auth, data persistence, external "
                "API calls, or sensitive data. Security bugs hide in edge cases — "
                "read thoroughly."
            ),
            prompt=prepend_task(task, (
                "Gather source code for a thorough security audit: entry points, "
                "authentication/authorisation, data handling, sensitive data, "
                "configuration. Do not audit yet — only gather."
            )),
        )
        r.wait()
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "retry"],
            "Have you gathered enough code for a thorough security audit? 'ok' or 'retry'.",
        )
        if verdict == "ok":
            break

    # stage 2: threat model.
    r = h.run_agent(
        system_prompt=(
            "You are a senior application security engineer building a threat "
            "model. Think like an attacker. Map every attack surface "
            "systematically."
        ),
        prompt=(
            "Produce: ATTACK SURFACE MAP, TRUST BOUNDARIES, DATA FLOWS, "
            "AUTH MODEL, HIGH-RISK AREAS. Be systematic and specific."
        ),
    )
    r.wait()
    h.enrich(r.messages)

    # stage 3: audit for vulnerabilities.
    r = h.run_agent(
        tools=AUDIT_TOOLS,
        system_prompt=(
            "You are an expert application security engineer performing a "
            "detailed vulnerability audit. Every finding must be grounded in "
            "actual code."
        ),
        prompt=(
            "Audit every high-risk area for: INJECTION, AUTH FLAWS, AUTHZ FLAWS, "
            "SENSITIVE DATA EXPOSURE, CRYPTO ISSUES, DEPENDENCIES, INPUT "
            "VALIDATION. For each finding: file+location, class, pattern, attack "
            "scenario, impact. Be thorough."
        ),
    )
    r.wait()
    h.enrich(r.messages)

    # stage 4: severity. pick one word.
    verdict = h.gate(
        ["critical", "high", "medium", "low", "clean"],
        "Classify severity based on the worst finding: critical, high, "
        "medium, low, or clean.",
        system_prompt="You are a senior security engineer assigning overall severity. Answer with exactly one word.",
    )

    # stage 5: report. pick system/prompt per severity and write final report.
    if verdict == "critical":
        system = (
            "You are a senior security engineer writing a critical security advisory. "
            "Be urgent, precise, direct. Do not soften findings."
        )
        prompt = (
            "Write: ## SECURITY AUDIT REPORT — CRITICAL — Executive Summary, Critical "
            "Findings (with attack scenario and remediation for each), High Findings, "
            "Medium/Low summary, Positive Controls, Recommended Immediate Actions."
        )
    elif verdict == "high":
        system = (
            "You are a senior security engineer writing a high-severity report. "
            "Precise, direct, actionable."
        )
        prompt = (
            "Write: ## SECURITY AUDIT REPORT — HIGH SEVERITY — Summary, High Findings "
            "(with attack scenario and remediation), Medium Findings, Low Findings, "
            "Security Strengths, Remediation Priority."
        )
    elif verdict == "medium":
        system = (
            "You are a senior security engineer writing a medium-severity report. "
            "Constructive and specific."
        )
        prompt = (
            "Write: ## SECURITY AUDIT REPORT — MEDIUM SEVERITY — Summary, Findings "
            "(ordered by severity, with remediation), Security Strengths, Recommended "
            "Improvements."
        )
    elif verdict == "low":
        system = (
            "You are a senior security engineer writing a low-severity report. "
            "Constructive. The code is mostly secure."
        )
        prompt = (
            "Write: ## SECURITY AUDIT REPORT — LOW SEVERITY — Summary, Minor Findings, "
            "Security Strengths, Hardening Recommendations."
        )
    else:
        system = (
            "You are a senior security engineer writing a clean sign-off. Be specific "
            "about what makes the code secure."
        )
        prompt = (
            "Write: ## SECURITY AUDIT REPORT — NO SIGNIFICANT FINDINGS — Summary, "
            "Areas Reviewed, Security Strengths (be specific), Optional Hardening."
        )
    r = h.run_agent(system_prompt=system, prompt=prompt)
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
