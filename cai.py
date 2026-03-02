import argparse
import json
import sys
import os
import re

from api import OpenAiApi, OpenRouterApi
from tools import get_tools, call_tool

config_dir = os.path.expanduser("~/.config/cai")

config = json.loads(open(os.path.join(config_dir, "config.json")).read())
tools = get_tools()
api_key = open(os.path.join(config_dir, "api_key")).read().strip()
openai_api = OpenAiApi(config.get('base_url'), api_key)
openrouter_api = OpenRouterApi(api_key)
models = openrouter_api.get_models()
# stats = openrouter_api.get_account_stats()
# price_so_far = stats.get('data', {}).get('usage')

def get_model_context_length(model):
    global models
    for _model in models:
        if _model.get('id') == model:
            return _model.get('context_length')

def read_stdin_if_available():
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None

def handle_tool_calls(args, tool_calls, messages):
    for call in tool_calls:
        if call.get('type') != 'function':
            print(f"[!] got tool call with invalid type: {call.get('type')}")
            continue
        try:
            call_id = call.get('id')
            call_function = call.get('function')
            call_name = call_function.get('name')
            arguments = call_function.get('arguments')
            if len(arguments) > 0:
                call_args = json.loads(call_function.get('arguments'))
            else:
                call_args = {}

            result = call_tool(call_name, call_args)

            request_message = {}
            request_message['role'] = 'assistant'
            request_message['tool_calls'] = [{}]
            request_message['tool_calls'][0]['id'] = call_id
            request_message['tool_calls'][0]['type'] = "function"
            request_message['tool_calls'][0]['function'] = {}
            request_message['tool_calls'][0]['function']['arguments'] = json.dumps(call_args)
            request_message['tool_calls'][0]['function']['name'] = call_name

            response_message = {}
            response_message['role'] = 'tool'
            response_message['tool_call_id'] = call_id
            response_message['content'] = result

            messages.append(request_message)
            messages.append(response_message)
        except Exception as e:
            print(f"[!] tool_call exception: {e} {json.dumps(call, indent=2)}")
            continue
    return openai_api.chat(messages, model=args.model) # no tools this time!

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
    if args.file:
        file_content = []
        i = 1
        for line in open(args.file).readlines():
            file_content.append(f"{i}: {line}")
            i += 1
        file_content = '\n'.join(file_content)
        messages.append({ "role": "user", "content": f"<file_content> {file_content} </file_content>" })
    if args.location:
        m = re.match(r"^(?P<file_path>.*):(?P<line_num>\d+):(?P<col_num>\d+)$", args.location)
        if m:
            file_path = m.group('file_path')
            line_num = m.group('line_num')
            col_num = m.group('col_num')

            file_content = []
            i = 1
            for line in open(file_path).readlines():
                file_content.append(f"{i}: {line}")
                i += 1
            file_content = '\n'.join(file_content)
            messages.append({ "role": "user", "content": f"<file_content> {file_content} </file_content>" })
            messages.append({ "role": "user", "content": f"<cursor_location> line number: {line_num}, column number: {col_num} </cursor_location>" })

    messages.append({ "role": "user", "content": args.prompt })

    included_tools = []
    for tool in tools:
        tool_name = tool.get('function',{}).get('name')

        if args.codebase:
            if tool_name in ("fetch_codebase_metadata"):
                included_tools.append(tool)

    result = openai_api.chat(messages, model=args.model, tools=included_tools)
    if not result: return
    content, reasoning, tool_calls = result

    if tool_calls:
        result = handle_tool_calls(args, tool_calls, messages)
        if not result: return

        content, reasoning, tool_calls = result

    print(f"{content}")

ACTION_GREP = "grep"
def action_grep(args):
    pass

ACTION_KNOWIT = "knowit"
def action_knowit(args):
    pass


def main():
    parser = argparse.ArgumentParser(description="cai is a command line utility to make use of LLM intelegent in multiple cases.")

    parser.add_argument("-a", "--action",
                        choices=[
                            ACTION_PROMPT,
                            ACTION_GREP,
                            ACTION_KNOWIT,
                            ],
                        default=ACTION_PROMPT,
                        help="the actiont to be performed.")
    parser.add_argument("-p", "--prompt", help="the prompt to send to the LLM.")
    parser.add_argument("--system-prompt", help="the system prompt to send to the LLM.")
    parser.add_argument("--cwd", default=".", help="the current working for the script to operate at.")
    parser.add_argument("--file", help="file path to include in the LLM context.")
    parser.add_argument("--location", help="the location in the codebase to be used by the action. in the format of => <file_path>:<line_num>:<col_num>")
    parser.add_argument("--stdin", help="for internal use.")
    parser.add_argument("--model",
                        # default="arcee-ai/trinity-mini:free",
                        default="anthropic/claude-opus-4.6",
                        help="the model to be used by the LLM")
    parser.add_argument("--codebase", action="store_true", help="let the action be aware of the codebase")

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
