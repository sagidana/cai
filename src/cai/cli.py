"""cli: the command-line entry point.

A minimal driver: build an Agent (which reads config.json + api_key and builds
its own client), run the prompt from argv, and stream the answer to stdout. No
interactive mode yet - that is a later layer."""
import argparse
import sys

from cai.agent import Agent
from cai.events import EventType


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cai",
        description="Send a prompt to an LLM and stream the answer.")
    parser.add_argument("prompt",
                        nargs="+",
                        help="the prompt to send (trailing words are joined)")
    parser.add_argument("--model",
                        default=None,
                        help="model id (default: the `model` field in config.json)")
    args = parser.parse_args(argv)

    try:
        agent = Agent(model=args.model)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    prompt = " ".join(args.prompt)
    for event in agent.run(prompt):
        if event.type == EventType.REASONING:
            sys.stdout.write(event.text)
            sys.stdout.flush()
        if event.type == EventType.CONTENT:
            sys.stdout.write(event.text)
            sys.stdout.flush()
    print()
    return 0
