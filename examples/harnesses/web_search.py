"""Port of harnesses/web-search.harness.cai.

decompose → for-each sub-question, do (search → verify → report) in an
isolated harness → synthesize.

Each sub-question is researched in its own fresh Harness so quality does not
degrade as topic breadth grows.

Usage:
    python examples/harnesses/web_search.py "what are the trade-offs between PostgreSQL and MongoDB for a high-write SaaS product?"
"""
from cai import Harness
from _helpers import get_task


def main() -> None:
    task = get_task()

    # stage 1: decompose the task into focused sub-questions.
    h_decompose = Harness()
    r = h_decompose.agent(
        system_prompt=(
            "Output a plain list of focused research sub-questions — one per "
            "line, no bullets, no numbering, no extra text. Maximum 5."
        ),
        prompt=(
            f"User task:\n{task}\n\n"
            "Break into focused, independent, searchable research sub-questions "
            "that together answer it comprehensively. 3–5 sub-questions, one per "
            "line, no prefixes. If already single and scoped, output as-is."
        ),
    )
    r.wait()
    questions = [line.strip() for line in r.text.splitlines() if line.strip()][:5]
    print(f"[decomposed into {len(questions)} sub-questions]")

    # stage 2: research each sub-question in its own isolated harness.
    findings: list[tuple[str, str]] = []
    for i, question in enumerate(questions, 1):
        print(f"\n=== researching ({i}/{len(questions)}): {question}")
        h_research = Harness()

        # search loop: loop up to 3 times until verify says 'ok'.
        for _ in range(3):
            r = h_research.agent(
                tools=["fetch_url"],
                system_prompt=(
                    "You are a focused web research agent. Your only tool is "
                    "fetch_url. Search DuckDuckGo HTML and read the most relevant "
                    "result pages. Read thoroughly."
                ),
                prompt=(
                    f"Research question:\n{question}\n\n"
                    "STEP 1 fetch https://html.duckduckgo.com/html/?q=<url-encoded-query>.\n"
                    "STEP 2 parse result links, extract 5–8 destination URLs.\n"
                    "STEP 3 fetch the 3–5 most authoritative. Prefer official docs / "
                    "academic / well-known publishers.\n"
                    "STEP 4 follow references when a page points at a more "
                    "authoritative source.\n"
                    "STEP 5 refine and retry the search if results are weak.\n"
                    "Track key facts, expert positions, caveats, source URLs. "
                    "Produce no output yet."
                ),
            )
            r.wait()
            h_research.enrich(r.messages)

            verdict = h_research.gate(
                ["ok", "retry"],
                "Have you gathered enough relevant web content to answer the "
                "question thoroughly? 'ok' only if 3+ relevant pages read, "
                "specific factual info, key sub-aspects covered.",
            )
            if verdict == "ok":
                break

        # report: produce the per-question sourced summary.
        r = h_research.agent(
            system_prompt=(
                "You are a research analyst distilling web findings into a precise, "
                "well-sourced summary. Every key claim must have a source URL."
            ),
            prompt=(
                f"Research question:\n{question}\n\n"
                "Structure:\n"
                "**Question:** one-line restate.\n"
                "**Findings:** specific facts, figures, dates, names.\n"
                "**Caveats and gaps:** uncertainties and unsourced areas.\n"
                "**Sources:** URLs used."
            ),
        )
        r.wait()
        findings.append((question, r.text))

    # stage 3: synthesize all per-question findings into one answer.
    print("\n=== synthesis\n")
    h_synth = Harness()
    for q, ans in findings:
        h_synth.enrich(f"Sub-question: {q}\n\n{ans}")

    r = h_synth.agent(
        system_prompt=(
            "You are an expert research analyst. Synthesize gathered research "
            "into a comprehensive, well-structured, accurate answer. Prioritise "
            "depth and accuracy over brevity."
        ),
        messages=h_synth.messages,
        prompt=(
            f"Original question:\n{task}\n\n"
            "Synthesize everything into a single comprehensive answer. Open with "
            "a direct 1–2 sentence answer, then sections by aspect. Cite sources "
            "as [Source: <URL>]. Flag contradictions, caveats, gaps. Do not "
            "invent facts."
        ),
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
