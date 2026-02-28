import json
import requests


class OpenAiApi:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

    def chat(self, messages, model, system_prompt=None, tools=None):
        url = f"{self.base_url}/chat/completions"
        headers = {}
        headers['Authorization'] = f"Bearer {self.api_key}"
        headers['Content-Type'] = "application/json"

        data = {}

        data['model'] = model
        data['messages'] = []
        if tools:
            data['tools'] = tools

        if system_prompt:
            data['messages'].append(system_prompt)

        data['messages'].extend(messages)
        data['tool_choice'] = "auto"

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

        return content, reasoning, tool_calls

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
    api = OpenAiApi("https://openrouter.ai/api/v1", open('./api_key').read().strip())

    api.chat([
            {
                "role": "user",
                "content": "is this working?",
            }
        ])

if __name__ == "__main__":
    main()
