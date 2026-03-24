import json
import logging
import requests

log = logging.getLogger("cai")


class OpenRouterApi:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_models(self):
        url = "https://openrouter.ai/api/v1/models"
        r = requests.get(url)
        if r.status_code != 200:
            print(f"[!] request {url} failed with {r.status_code}, {r.text}")
            return
        response = r.json()

        return response.get('data')

    def get_account_stats(self):
        url = "https://openrouter.ai/api/v1/auth/key"
        headers = {}
        headers['Authorization'] = f"Bearer {self.api_key}"

        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"[!] request {url} failed with {r.status_code}, {r.text}")
            return

        return r.json()

class OpenAiApi:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

    def chat(self, messages, model, system_prompt=None, tools=None, tool_choice="auto"):
        url = f"{self.base_url}/chat/completions"
        headers = {}
        headers['Authorization'] = f"Bearer {self.api_key}"
        headers['Content-Type'] = "application/json"

        data = {}

        data['model'] = model
        data['messages'] = []
        if tools:
            data['tools'] = tools
            data['tool_choice'] = tool_choice

        if system_prompt:
            data['messages'].append(system_prompt)

        data['messages'].extend(messages)

        r = requests.post(url, headers=headers, json=data)
        if r.status_code != 200:
            print(f"[!] request {url} failed: {r.status_code}, {r.text}")
            return

        result = r.json()
        choices = result.get("choices", [])
        if len(choices) != 1:
            print(f"[!] len(choices) != 1: {choices}")
            return

        choice = choices[0]

        message = choice.get('message', None)
        if not message:
            print(f"[!] choice message is None")
            return

        content = message.get('content', "")
        reasoning = message.get('reasoning', "")
        tool_calls = message.get('tool_calls', None)
        usage = result.get('usage', {})

        return content, reasoning, tool_calls, usage

    def chat_stream(self, messages, model, system_prompt=None, tools=None, tool_choice="auto"):
        url = f"{self.base_url}/chat/completions"
        headers = {}
        headers['Authorization'] = f"Bearer {self.api_key}"
        headers['Content-Type'] = "application/json"

        data = {}

        data['model'] = model
        data['messages'] = []
        if tools:
            data['tools'] = tools
            data['tool_choice'] = tool_choice

        if system_prompt:
            data['messages'].append(system_prompt)

        data['messages'].extend(messages)
        data['stream'] = True
        data['stream_options'] = {"include_usage": True}

        log.info("chat_stream: model=%s tools=%d tool_choice=%s", model, len(tools or []), tool_choice)

        finished_tool_calls = None
        tool_calls = {}
        usage = {}

        with requests.post(url, headers=headers, json=data, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line: continue

                if not line.startswith(b"data: "): continue

                response_data = line[len(b"data: "):]

                if response_data == b"[DONE]": break
                chunk = json.loads(response_data)

                # Final usage chunk has empty choices
                if chunk.get('usage'):
                    usage = chunk['usage']

                if len(chunk["choices"]) != 1: continue

                choice = chunk["choices"][0]
                finish_reason = choice.get('finish_reason', None)
                delta = choice["delta"]

                # Process tool call delta fragments BEFORE snapshotting finish_reason,
                # so the snapshot captures data from the final chunk too.
                if "tool_calls" in delta:
                    for tool_call in delta['tool_calls']:
                        idx = tool_call['index']
                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                'index': tool_call.get('index'),
                                'id': tool_call.get('id'),
                                'type': "function",
                                'function': {
                                    'name': tool_call.get('function', {}).get('name'),
                                    'arguments': "",
                                },
                            }
                        args = tool_call.get('function', {}).get('arguments')
                        if args:
                            tool_calls[idx]['function']['arguments'] += args

                # "tool_calls" (OpenAI) and "tool_use" (Anthropic native) are both valid.
                # Guard: only snapshot when tool_calls is non-empty — some providers fire
                # finish_reason="tool_calls" twice, and the second time tool_calls is already
                # reset to {}, which would overwrite the valid snapshot with an empty list.
                if finish_reason in ("tool_calls", "tool_use") and tool_calls:
                    log.info("chat_stream: finish_reason=%s tool_calls_count=%d", finish_reason, len(tool_calls))
                    finished_tool_calls = list(tool_calls.values())
                    tool_calls = {}
                elif finish_reason:
                    log.info("chat_stream: finish_reason=%s", finish_reason)

                content = delta.get('content', None)

                if content or tool_calls:
                    yield content, finished_tool_calls, {}
            yield None, finished_tool_calls, usage

class AnthropicApi:
    def __init__(self, api_key):
        self.api_key = api_key

    def messages(self, messages):
        url = "https://api.anthropic.com/v1/messages"
        headers = {}
        headers['x-api-key'] = self.api_key
        headers['anthropic-version'] = "2023-06-01"
        headers['Content-Type'] = "application/json"

        data = {}
        data['model'] = "claude-3-haiku-20240307"
        data['max_tokens'] = 200
        data['messages'] = messages

        r = requests.post(url, headers=headers, json=data)
        if r.status_code != 200:
            print(f"[!] request {url} failed: {r.status_code}, {r.text}")
            return
        print(r.json())


def main():
    import os
    api_key = open(os.path.expanduser('~/.config/cai/api_key')).read().strip()
    api = OpenAiApi("https://openrouter.ai/api/v1", api_key)

    tools = [{ "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": {"type": "string"} },
                        "required": ["city"]
                    }
                } }]

    model = "arcee-ai/trinity-mini:free"
    # model = "anthropic/claude-opus-4.6"

    content, reasoning, tool_calls, usage = api.chat([{ "role":"user", "content":"write python script the print the current weather at London" }], model=model, tools=tools)
    print(f"{content=}")
    print(f"{reasoning=}")
    print(f"{tool_calls=}")

    # for content, tool_calls in api.chat_stream([{ "role":"user", "content":"write python script the print the current weather at London" }], model=model, tools=tools):
        # if tool_calls:
            # print(f"{tool_calls=}")
        # if content:
            # print(content, end="", flush=True)

if __name__ == "__main__":
    main()
