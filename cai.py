import argparse
import sys

from api import OpenAiApi

openai_api = None

def read_stdin_if_available():
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None

ACTION_PROMPT = "prompt"
def action_prompt(args):
    if not args.prompt:
        print("action prompt reruire --prompt to be provided.")
        return


    messages = []

    if args.stdin: messages.append({ "role": "user", "content": args.stdin })

    messages.append({ "role": "user", "content": args.prompt })

    result = openai_api.chat(messages)
    if not result: return
    content, reasoning = result

    print(f"{content}")

ACTION_GREP = "grep"
def action_grep(args):
    pass

def main():
    global openai_api

    openai_api = OpenAiApi("https://openrouter.ai/api/v1", open('./api_key').read().strip())

    parser = argparse.ArgumentParser(description="cai is a command line utility to make use of LLM intelegent in multiple cases.")

    parser.add_argument("-a", "--action",
                        choices=[
                            ACTION_PROMPT,
                            ACTION_GREP,
                            ],
                        required=True,
                        help="the actiont to be performed.")
    parser.add_argument("-p", "--prompt", help="the prompt to send to the LLM.")
    parser.add_argument("--system-prompt", help="the system prompt to send to the LLM.")
    parser.add_argument("--file", help="file path to include in the LLM context.")
    parser.add_argument("--stdin", help="for internal use.")

    args = parser.parse_args()

    args.stdin = read_stdin_if_available()

    if args.action == ACTION_PROMPT:
        action_prompt(args)
    if args.action == ACTION_GREP:
        action_grep(args)


if __name__ == "__main__":
    main()
