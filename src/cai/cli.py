"""cli: the command-line entry point.

A minimal driver over the Layer 0/1 stack: read the API key (config), build the
OpenRouter client (api), run the agentic loop (llm) over a prompt from argv, and
stream the answer to stdout. No tools and no interactive mode yet - those are
later layers."""
import argparse
import sys

from cai import config
from cai.api import OpenAiApi
from cai.events import EventType
from cai.llm import call_llm


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cai",
        description="Send a prompt to an LLM and stream the answer.")
    parser.add_argument("prompt",
                        nargs="+",
                        help="the prompt to send (trailing words are joined)")
    parser.add_argument("--model",
                        default=config.DEFAULT_MODEL,
                        help=f"model id (default: {config.DEFAULT_MODEL})")
    args = parser.parse_args(argv)

    try:
        api_key = config.load_api_key()
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    prompt = " ".join(args.prompt)
    api = OpenAiApi(config.OPENROUTER_BASE_URL, api_key)
    messages = [{"role": "user", "content": prompt}]

    for event in call_llm(messages, args.model, api):
        if event.type == EventType.REASONING:
            sys.stdout.write(event.text)
            sys.stdout.flush()
        if event.type == EventType.CONTENT:
            sys.stdout.write(event.text)
            sys.stdout.flush()
    print()
    return 0
