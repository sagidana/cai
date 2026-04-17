"""Port of harnesses/maze-challenge.harness.cai.

Single-block harness: hand the task to the aidle MCP server and let it solve.

Usage:
    python examples/harnesses/maze_challenge.py
"""
from cai import Harness


def main() -> None:
    h = Harness(
        mcp_servers=["aidle mcp --username bob"],
        system_prompt=(
            "you are challenger solver, your goal is to solve the challenges before "
            "you as fast and as efficient as possible."
        ),
    )
    r = h.agent(prompt="choose the maze challenge and solve it as fast as possible")
    r.wait()
    print(r.text)


if __name__ == "__main__":
    main()
