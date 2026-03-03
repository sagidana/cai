import argparse
import json
import sys
import os
import re

from cai.api import OpenAiApi, OpenRouterApi
from cai.tools import (get_tools,
                       call_tool,
                       get_external_tools,
                       call_external_tool)

global config
global tools
global api_key
global openai_api
global openrouter_api
global external_mcps
external_mcps = {}


def init():
    global config
    global tools
    global api_key
    global openai_api
    global openrouter_api

    config_dir = os.path.expanduser("~/.config/cai")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    config_path = os.path.join(config_dir, "config.json")
    if not os.path.exists(config_path):
        default_config = {
            "base_url": "https://openrouter.ai/api/v1",
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        print(f"[*] Created default config at {config_path}")

    api_key_path = os.path.join(config_dir, "api_key")
    if not os.path.exists(api_key_path):
        with open(api_key_path, "w") as f:
            f.write("")
        print(f"[*] Created empty api_key file at {api_key_path}")

    config = json.loads(open(config_path).read())
    tools = get_tools()
    api_key = open(api_key_path).read().strip()
    openai_api = OpenAiApi(config.get('base_url'), api_key)
    openrouter_api = OpenRouterApi(api_key)

    # models = openrouter_api.get_models()
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

def handle_tool_calls(tool_calls, messages):
    global external_mcps
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

            called_external = False
            result = None

            for mcp_path in external_mcps:
                tools = external_mcps[mcp_path]
                for tool in tools:
                    tool_name = tool.get('function',{}).get('name')
                    if tool_name == call_name:
                        result = call_external_tool(mcp_path, call_name, call_args)
                        called_external = True

            if not called_external:
                result = call_tool(call_name, call_args)

            assert result is not None, "[!] failed to call tool: {call=}"

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

ACTION_PROMPT = "prompt"
def action_prompt(args):
    global external_mcps

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

    # adding external tools
    for mcp_path in external_mcps:
        mcp_tools = external_mcps[mcp_path]
        included_tools.extend(mcp_tools)

    tool_calls_happened = False
    for content, tool_calls in openai_api.chat_stream(messages,
                                                      model=args.model,
                                                      tools=included_tools):
        if tool_calls:
            print("\n", end="", flush=True)
            handle_tool_calls(tool_calls, messages) # updates the messages
            print("\n", end="", flush=True)
            tool_calls_happened = True

        if content:
            print(content, end="", flush=True)

    if tool_calls_happened:
        for content, _ in openai_api.chat_stream(messages, model=args.model):
            if content:
                print(content, end="", flush=True)

    print('\n', end='', flush=True)

ACTION_GREP = "grep"
def action_grep(args):
    pass

ACTION_KNOWIT = "knowit"
def action_knowit(args):
    pass


def main():
    global external_mcps
    global config
    init()

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

    default_model = config.get('model', "arcee-ai/trinity-mini:free")
    # default_model = config.get('model', "anthropic/claude-opus-4.6")
    parser.add_argument("--model", default=default_model, help="the model to be used by the LLM")
    parser.add_argument("--codebase", action="store_true", help="let the action be aware of the codebase")
    parser.add_argument('-t',
                        '--tools',
                        nargs='+',
                        default=[],
                        help="list of mcp tools to give the LLM. the tools come in the form of abosult paths to the python files implementing the mcp server.")

    args = parser.parse_args()

    args.stdin = read_stdin_if_available()

    external_mcps = {}
    for mcp_server_path in args.tools:
        external_mcps[mcp_server_path] = get_external_tools(mcp_server_path)

    if args.action == ACTION_PROMPT:
        action_prompt(args)
    if args.action == ACTION_GREP:
        action_grep(args)
    if args.action == ACTION_KNOWIT:
        action_knowit(args)


if __name__ == "__main__":
    main()
