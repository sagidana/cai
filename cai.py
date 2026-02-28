from pathlib import Path
import argparse
import json
import sys

from api import OpenAiApi
from tools import get_tools, call_tool

base_dir = Path(__file__).resolve().parent
config = json.loads(open(base_dir/"config.json").read())
tools = get_tools()
openai_api = OpenAiApi(config.get('base_url'), open(base_dir / 'api_key').read().strip())


def read_stdin_if_available():
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None


ACTION_PROMPT = "prompt"
def action_prompt(args):
    if not args.prompt:
        print("this action require --prompt to be provided.")
        return

    messages = []

    if args.system_prompt:
        messages.append({ "role": "system", "content": args.system_prompt })
    if args.stdin:
        messages.append({ "role": "user", "content": args.stdin })

    messages.append({ "role": "user", "content": args.prompt })

    result = openai_api.chat(messages, model=args.model)
    if not result: return
    content, reasoning, tool_calls = result

    print(f"{content}")

ACTION_GREP = "grep"
def action_grep(args):
    pass

ACTION_KNOWIT = "knowit"
def action_knowit(args):
    global tools

    if not args.prompt:
        print("this action require --prompt to be provided.")
        return

    messages = []

    if args.system_prompt:
        messages.append({ "role": "system", "content": args.system_prompt })
    if args.stdin:
        messages.append({ "role": "user", "content": args.stdin })

    messages.append({ "role": "user", "content": args.prompt })

    result = openai_api.chat(messages, model=args.model, tools=tools)
    if not result: return

    content, reasoning, tool_calls = result

    if tool_calls:
        for call in tool_calls:
            if call.get('type') != 'function':
                print(f"[!] got tool call with invalid type: {call.get('type')}")
                continue
            try:
                call_id = call.get('id')
                call_function = call.get('function')
                call_name = call_function.get('name')
                call_args = json.loads(call_function.get('arguments'))

                result = call_tool(call_name, call_args)

                message = {}
                message['role'] = 'tool'
                message['tool_call_id'] = call_id
                message['content'] = result

                messages.append(message)
            except Exception as e:
                print(f"[!] tool_call exception: {e}")
                continue
        result = openai_api.chat(messages, model=args.model, tools=tools)
        if not result: return

        content, reasoning, tool_calls = result
        # print(f"{content=}")
        # print(f"{reasoning=}")
        # print(f"{tool_calls=}")

    print(content)

def main():
    parser = argparse.ArgumentParser(description="cai is a command line utility to make use of LLM intelegent in multiple cases.")

    parser.add_argument("-a", "--action",
                        choices=[
                            ACTION_PROMPT,
                            ACTION_GREP,
                            ACTION_KNOWIT,
                            ],
                        required=True,
                        help="the actiont to be performed.")
    parser.add_argument("-p", "--prompt", help="the prompt to send to the LLM.")
    parser.add_argument("--system-prompt", help="the system prompt to send to the LLM.")
    parser.add_argument("--file", help="file path to include in the LLM context.")
    parser.add_argument("--stdin", help="for internal use.")
    parser.add_argument("--model", default="stepfun/step-3.5-flash:free", help="the model to be used by the LLM")

    args = parser.parse_args()

    args.stdin = read_stdin_if_available()

    if args.action == ACTION_PROMPT:
        action_prompt(args)
    if args.action == ACTION_GREP:
        action_grep(args)
    if args.action == ACTION_KNOWIT:
        action_knowit(args)


if __name__ == "__main__":
    main()
