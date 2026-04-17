"""Port of harnesses/haystack-challenge.harness.cai.

Four stages: start → (search → verify)* → submit. The verify gate retries the
search block until the model answers 'ok'.

Usage:
    python examples/harnesses/haystack_challenge.py
"""
from cai import Harness


SOLVER_PROMPT = (
    "you are challenger solver, your goal is to solve the challenges before you. "
    "take your time, do it step by step, double check yourself each step to make "
    "sure you are on the right track, if not, re-iterate. TAKE YOUR TIME."
)


def main() -> None:
    h = Harness(mcp_servers=["aidle mcp --username bob"])

    # stage 1: start. gather all buckets.
    r = h.run_agent(
        system_prompt=SOLVER_PROMPT,
        prompt=(
            "choose the haystack challenge in medium difficulty and start reading "
            "all buckets. keep all buckets as is, this phase is only the gathering "
            "phase. return a list of all the buckets you have and key word to look for."
        ),
    )
    r.wait()
    h.enrich(r.messages)   # --enrich full

    # stage 2: search → verify. loop up to 30 times until verify says 'ok'.
    for _ in range(30):
        r = h.run_agent(
            system_prompt=SOLVER_PROMPT,
            prompt=(
                "your goal is to return for each bucket the places (if there are "
                "any) of the needle (the word you need to look for). you have all "
                "the information you need in the history! return for each bucket "
                "— the word position(s) if there are any."
            ),
        )
        r.wait()
        h.enrich(r.text)   # --enrich result-only

        verdict = h.gate(
            ["ok", "retry"],
            "Do you have a final result with good confidence? You MUST double "
            "check all given results to see you reach the same conclusion, if "
            "there are any mismatches, retry. Answer 'ok' only if YES. "
            "Answer 'retry' otherwise!",
        )
        if verdict == "ok":
            break

    # stage 3: submit.
    r = h.run_agent(
        system_prompt=SOLVER_PROMPT,
        prompt="you have the result, submit it and finish the challenge!",
    )
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
