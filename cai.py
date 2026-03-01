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
openai_api = OpenAiApi(config.get('base_url'),
                       open(os.path.join(config_dir, "api_key")).read().strip())
models = OpenRouterApi().get_models()

def get_model_context_length(model):
    global models
    for _model in models:
        if _model.get('id') == model:
            return _model.get('context_length')

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
    if not result:
        print(f"[!] failed to get response from LLM")
        return

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
        result = openai_api.chat(messages, model=args.model)
        if not result: return

        content, reasoning, tool_calls = result

    # print(f"{content=}")
    # print(f"{reasoning=}")
    # print(f"{tool_calls=}")

    print(content)

ACTION_IMPL = "impl"
def action_impl(args):
    if not args.prompt:
        print("this action require --prompt to be provided.")
        return
    if not args.location:
        print("this action require --location to be provided.")
        return

    messages = []

    system_prompt = ""
    system_prompt += f"- your task is to provide implementation compliance with the prompt given by the user.\n"
    system_prompt += f"- your output MUST only contains source code that can be placed as is into the file.\n"
    system_prompt += f"- the language of the implementation MUST be {args.output_language}.\n"
    system_prompt += f"- you will be provided with the whole file currently the cursor of the user is at.\n"
    system_prompt += f"- you will also be provided with the current location the curser of the use is at.\n"
    system_prompt += f"- your response should contain only the source code that changes or added to the file, be as concise as you can be.\n"

    system_prompt += f"- in case you will need more information about other files in the current working directory, "
    system_prompt += f"you can use tool call named 'fetch_codebase_infra' that will give you addition information about the current project structure in the following format:\n"
    system_prompt += f"<format>\n"
    system_prompt += """
    {
        "file_name": {
            "class_name": {
                "method_name": "method_prototype",
            },
            "method_name": "method_prototype",
        },
    }
    """
    system_prompt += f"</format>\n"

    messages.append({ "role": "system", "content": system_prompt })

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
            messages.append({ "role": "user", "content": f"<cursor_location> line number: {line_num}, column number: {col_num} </cursor_location>" })

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
                print(f"[!] tool_call exception: {e}")
                continue
        result = openai_api.chat(messages, model=args.model)
        if not result: return

        content, reasoning, tool_calls = result

    print(f"{content}")
    # print(f"{reasoning=}")
    # print(f"{tool_calls=}")


def main():
    parser = argparse.ArgumentParser(description="cai is a command line utility to make use of LLM intelegent in multiple cases.")

    parser.add_argument("-a", "--action",
                        choices=[
                            ACTION_PROMPT,
                            ACTION_GREP,
                            ACTION_KNOWIT,
                            ACTION_IMPL,
                            ],
                        required=True,
                        help="the actiont to be performed.")
    parser.add_argument("-p", "--prompt", help="the prompt to send to the LLM.")
    parser.add_argument("--system-prompt", help="the system prompt to send to the LLM.")
    parser.add_argument("--cwd", default=".", help="the current working for the script to operate at.")
    parser.add_argument("--file", help="file path to include in the LLM context.")
    parser.add_argument("--location", help="the location in the codebase to be used by the action. in the format of => <file_path>:<line_num>:<col_num>")
    parser.add_argument("--stdin", help="for internal use.")
    parser.add_argument("--model",
                        default="arcee-ai/trinity-mini:free",
                        # default="anthropic/claude-opus-4.6",
                        help="the model to be used by the LLM")
    parser.add_argument("--output-language", default="python", help="the programming language the model will output.")

    args = parser.parse_args()

    args.stdin = read_stdin_if_available()

    if args.action == ACTION_PROMPT:
        action_prompt(args)
    if args.action == ACTION_GREP:
        action_grep(args)
    if args.action == ACTION_KNOWIT:
        action_knowit(args)
    if args.action == ACTION_IMPL:
        action_impl(args)


if __name__ == "__main__":
    main()
