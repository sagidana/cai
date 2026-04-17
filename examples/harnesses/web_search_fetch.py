"""Port of harnesses/web-search-fetch.harness.cai.

Given a single research question, searches the web via fetch_url, reads a
handful of pages, and produces a focused, sourced summary.

Three stages: search → verify → report.

Usage:
    python examples/harnesses/web_search_fetch.py "what is the Paxos algorithm?"
"""
from cai import Harness
from _helpers import get_task


def main() -> None:
    question = get_task()
    h = Harness()

    # stage 1: search. loop up to 3 times until verify says 'ok'.
    for _ in range(3):
        r = h.agent(
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
        h.enrich(r.messages)

        verdict = h.gate(
            ["ok", "retry"],
            "Have you gathered enough relevant web content to answer the "
            "question thoroughly? 'ok' only if 3+ relevant pages read, "
            "specific factual info, key sub-aspects covered.",
        )
        if verdict == "ok":
            break

    # stage 2: report. produce the sourced summary.
    r = h.agent(
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
    print(r.text)


if __name__ == "__main__":
    main()
