import json
import logging
import warnings
import requests
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

log = logging.getLogger("cai")


class OpenRouterApi:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_models(self):
        url = "https://openrouter.ai/api/v1/models"
        try:
            r = requests.get(url)
        except requests.RequestException as e:
            log.error(f"[!] request {url} failed: {e}")
            return
        if r.status_code != 200:
            log.error(f"[!] request {url} failed with {r.status_code}, {r.text}")
            return
        try:
            response = r.json()
        except ValueError as e:
            log.error(f"[!] request {url} returned invalid JSON: {e}")
            return

        return response.get('data')

    def get_account_stats(self):
        url = "https://openrouter.ai/api/v1/auth/key"
        headers = {}
        headers['Authorization'] = f"Bearer {self.api_key}"

        try:
            r = requests.get(url, headers=headers)
        except requests.RequestException as e:
            log.error(f"[!] request {url} failed: {e}")
            return
        if r.status_code != 200:
            log.error(f"[!] request {url} failed with {r.status_code}, {r.text}")
            return

        try:
            return r.json()
        except ValueError as e:
            log.error(f"[!] request {url} returned invalid JSON: {e}")
            return

def _is_o_series(model):
    """Check if model is an OpenAI o-series reasoning model (o1, o3, o4-mini, etc.)."""
    # Match openai/o1, openai/o3, openai/o4-mini, or bare o1, o3, etc.
    parts = model.rsplit('/', 1)
    name = parts[-1]
    return name.startswith('o') and len(name) >= 2 and name[1:2].isdigit()


class OpenAiApi:
    def __init__(self, base_url, api_key, ssl_verify=True, error_cb=None):
        self.base_url = base_url
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        self.error_cb = error_cb

    def _report(self, msg):
        """Log an error and, if a sink is attached, forward it there too."""
        log.error(msg)
        if self.error_cb is not None:
            try:
                self.error_cb(msg)
            except Exception as e:
                log.error(f"[!] error_cb raised: {e}")

    def get_models(self):
        url = f"{self.base_url}/models"
        headers = {'Authorization': f"Bearer {self.api_key}"}
        try:
            r = requests.get(url, headers=headers, verify=self.ssl_verify)
        except requests.RequestException as e:
            self._report(f"[!] request {url} failed: {e}")
            return None
        if r.status_code != 200:
            return None
        try:
            return [m.get('id') for m in r.json().get('data', [])]
        except ValueError as e:
            self._report(f"[!] request {url} returned invalid JSON: {e}")
            return None

    def chat(self, messages, model, system_prompt=None, tools=None, tool_choice="auto", reasoning_effort=None, temperature=None):
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
        if reasoning_effort:
            data['reasoning'] = {"effort": reasoning_effort}
        if temperature is not None:
            if _is_o_series(model):
                log.warning("temperature=%.1f ignored for o-series model %s (fixed at 1)", temperature, model)
            else:
                data['temperature'] = temperature

        if _is_o_series(model):
            for msg in messages:
                if msg.get('role') == 'system':
                    log.warning("o-series model %s does not support role='system' — use role='developer' instead", model)
                    break

        if system_prompt:
            data['messages'].append(system_prompt)

        data['messages'].extend(messages)

        try:
            r = requests.post(url, headers=headers, json=data, verify=self.ssl_verify)
        except requests.RequestException as e:
            self._report(f"[!] request {url} failed: {e}")
            return
        if r.status_code != 200:
            self._report(f"[!] request {url} failed: {r.status_code}, {r.text}")
            return

        try:
            result = r.json()
        except ValueError as e:
            self._report(f"[!] request {url} returned invalid JSON: {e}")
            return
        choices = result.get("choices", [])
        if len(choices) != 1:
            self._report(f"[!] len(choices) != 1: {choices}")
            return

        choice = choices[0]

        message = choice.get('message', None)
        if not message:
            self._report(f"[!] choice message is None")
            return

        content = message.get('content', "")
        reasoning = message.get('reasoning', "")
        tool_calls = message.get('tool_calls', None)
        usage = result.get('usage', {})

        return content, reasoning, tool_calls, usage

    def chat_stream(self, messages, model, system_prompt=None, tools=None, tool_choice="auto", reasoning_effort=None, temperature=None):
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
        if reasoning_effort:
            data['reasoning'] = {"effort": reasoning_effort}
        if temperature is not None:
            if _is_o_series(model):
                log.warning("temperature=%.1f ignored for o-series model %s (fixed at 1)", temperature, model)
            else:
                data['temperature'] = temperature

        if _is_o_series(model):
            for msg in messages:
                if msg.get('role') == 'system':
                    log.warning("o-series model %s does not support role='system' — use role='developer' instead", model)
                    break

        if system_prompt:
            data['messages'].append(system_prompt)

        data['messages'].extend(messages)
        data['stream'] = True
        data['stream_options'] = {"include_usage": True}

        log.info("chat_stream: model=%s tools=%d tool_choice=%s", model, len(tools or []), tool_choice)

        finished_tool_calls = None
        tool_calls = {}
        usage = {}

        try:
            response_cm = requests.post(url, headers=headers, json=data, stream=True, verify=self.ssl_verify)
        except requests.RequestException as e:
            self._report(f"[!] request {url} failed: {e}")
            yield None, None, None, {}
            return

        with response_cm as response:
            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                self._report(f"[!] request {url} failed: {e.response.status_code}, {e.response.text}")
                yield None, None, None, {}
                return

            try:
                line_iter = response.iter_lines()
            except requests.RequestException as e:
                self._report(f"[!] request {url} stream failed: {e}")
                yield None, None, finished_tool_calls, usage
                return

            while True:
                try:
                    line = next(line_iter)
                except StopIteration:
                    break
                except requests.RequestException as e:
                    self._report(f"[!] request {url} stream interrupted: {e}")
                    yield None, None, finished_tool_calls, usage
                    return

                if not line: continue

                if not line.startswith(b"data: "): continue

                response_data = line[len(b"data: "):]

                if response_data == b"[DONE]": break
                try:
                    chunk = json.loads(response_data)
                except json.JSONDecodeError as e:
                    self._report(f"[!] request {url} stream returned invalid JSON: {e}")
                    continue

                # Final usage chunk has empty choices
                if chunk.get('usage'):
                    usage = chunk['usage']

                if len(chunk.get("choices") or []) != 1: continue

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
                reasoning = delta.get('reasoning', None)

                if content or reasoning or tool_calls:
                    yield content, reasoning, finished_tool_calls, {}
            yield None, None, finished_tool_calls, usage

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

        try:
            r = requests.post(url, headers=headers, json=data)
        except requests.RequestException as e:
            log.error(f"[!] request {url} failed: {e}")
            return
        if r.status_code != 200:
            log.error(f"[!] request {url} failed: {r.status_code}, {r.text}")
            return
        try:
            print(r.json())
        except ValueError as e:
            log.error(f"[!] request {url} returned invalid JSON: {e}")
            return


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
