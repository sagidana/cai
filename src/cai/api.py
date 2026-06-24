import json
import logging
import warnings
import requests
from urllib3.exceptions import InsecureRequestWarning

log = logging.getLogger("cai")


def _wire_messages(messages):
    """strip local-only ("_"-prefixed) keys before a message goes on the
    wire - some strict providers reject unknown message fields. the single
    choke point every request passes through."""
    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue

        has_local_keys = False
        for key in message:
            if key.startswith('_'):
                has_local_keys = True
                break
        if not has_local_keys:
            cleaned.append(message)
            continue

        wire_message = {}
        for key, value in message.items():
            if key.startswith('_'): continue
            wire_message[key] = value
        cleaned.append(wire_message)
    return cleaned


class OpenAiApi:
    """A minimal OpenAI-compatible chat client: one blocking HTTP POST to
    /chat/completions that returns the assistant's message. This is the
    bottom of the stack - it knows nothing about tools-as-code, sessions,
    streaming, or retries; it just speaks the wire format."""

    def __init__(self,
                 base_url,
                 api_key,
                 ssl_verify=True,
                 timeout=(10, 120)):
        self.base_url = base_url
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        if not ssl_verify:
            # only mute InsecureRequestWarning when verification is actually off.
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        # (connect, read) seconds.
        if isinstance(timeout, (list, tuple)):
            timeout = tuple(timeout)
        self.timeout = timeout

    def _request_data(self,
                      messages,
                      model,
                      system_prompt,
                      tools,
                      tool_choice,
                      reasoning_effort,
                      temperature):
        data = {}
        data['model'] = model
        data['messages'] = []
        if tools:
            data['tools'] = tools
            data['tool_choice'] = tool_choice
        if reasoning_effort:
            # send both shapes: OpenRouter reads `reasoning`,
            # vLLM/DeepSeek read `reasoning_effort`.
            data['reasoning'] = {"effort": reasoning_effort}
            data['reasoning_effort'] = reasoning_effort
        if temperature is not None:
            data['temperature'] = temperature

        if system_prompt:
            data['messages'].append(system_prompt)
        data['messages'].extend(_wire_messages(messages))
        return data

    def chat(self,
             messages,
             model,
             system_prompt=None,
             tools=None,
             tool_choice="auto",
             reasoning_effort=None,
             temperature=None,
             stream=False):
        """One chat-completion request. This is a dispatcher, not itself a
        generator, so its return shape depends on `stream`:

        - stream=False: blocks and returns a single tuple
          (content, reasoning, tool_calls, usage), or None on failure.
        - stream=True: returns a generator that yields incremental
          (content, reasoning, finished_tool_calls, usage) tuples; the final
          yield carries the full usage and the assembled tool calls.

        `system_prompt`, when given, is a ready-built message dict prepended
        to `messages`."""
        url = f"{self.base_url}/chat/completions"
        headers = {}
        headers['Authorization'] = f"Bearer {self.api_key}"
        headers['Content-Type'] = "application/json"

        data = self._request_data(messages,
                                  model,
                                  system_prompt,
                                  tools,
                                  tool_choice,
                                  reasoning_effort,
                                  temperature)

        if stream:
            data['stream'] = True
            data['stream_options'] = {"include_usage": True}
            return self._stream(url, headers, data)
        return self._complete(url, headers, data)

    def _complete(self, url, headers, data):
        """Blocking path: one POST, parse the single JSON body, return the
        (content, reasoning, tool_calls, usage) tuple or None."""
        try:
            r = requests.post(url,
                              headers=headers,
                              json=data,
                              timeout=self.timeout,
                              verify=self.ssl_verify)
        except requests.RequestException as e:
            log.error(f"[!] request {url} failed: {e}")
            return
        if r.status_code != 200:
            log.error(f"[!] request {url} failed with {r.status_code}, {r.text}")
            return
        try:
            result = r.json()
        except ValueError as e:
            log.error(f"[!] request {url} returned invalid JSON: {e}")
            return

        choices = result.get("choices", [])
        if len(choices) != 1:
            log.error(f"[!] len(choices) != 1: {choices}")
            return

        message = choices[0].get('message', None)
        if not message:
            log.error("[!] choice message is None")
            return

        content = message.get('content', "")
        reasoning = message.get('reasoning') or message.get('reasoning_content') or ""
        tool_calls = message.get('tool_calls', None)
        usage = result.get('usage', {})

        return content, reasoning, tool_calls, usage

    def _stream(self, url, headers, data):
        """Streaming path: POST with stream=True, parse the SSE `data:` lines,
        reassemble tool-call argument fragments, and yield deltas as they
        arrive. The final yield carries the assembled tool calls + full usage."""
        finished_tool_calls = None
        tool_calls = {}
        usage = {}

        try:
            r = requests.post(url,
                              headers=headers,
                              json=data,
                              stream=True,
                              timeout=self.timeout,
                              verify=self.ssl_verify)
        except requests.RequestException as e:
            log.error(f"[!] request {url} failed: {e}")
            yield None, None, None, {}
            return
        if r.status_code != 200:
            log.error(f"[!] request {url} failed with {r.status_code}, {r.text}")
            r.close()
            yield None, None, None, {}
            return

        with r:
            for line in r.iter_lines():
                if not line: continue
                if not line.startswith(b"data: "): continue

                payload = line[len(b"data: "):]
                if payload == b"[DONE]": break

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as e:
                    log.error(f"[!] request {url} stream returned invalid JSON: {e}")
                    continue

                # the final usage chunk has empty choices.
                if chunk.get('usage'):
                    usage = chunk['usage']

                if len(chunk.get("choices") or []) != 1: continue

                choice = chunk["choices"][0]
                finish_reason = choice.get('finish_reason', None)
                delta = choice["delta"]

                # process tool-call fragments before snapshotting finish_reason,
                # so the snapshot captures data from the final chunk too.
                if "tool_calls" in delta:
                    for tool_call in delta['tool_calls']:
                        idx = tool_call['index']
                        if idx not in tool_calls:
                            function = {}
                            function['name'] = tool_call.get('function', {}).get('name')
                            function['arguments'] = ""

                            tool_calls[idx] = {}
                            tool_calls[idx]['index'] = tool_call.get('index')
                            tool_calls[idx]['id'] = tool_call.get('id')
                            tool_calls[idx]['type'] = "function"
                            tool_calls[idx]['function'] = function
                        args = tool_call.get('function', {}).get('arguments')
                        if args:
                            tool_calls[idx]['function']['arguments'] += args

                # only snapshot when tool_calls is non-empty - some providers
                # fire finish_reason="tool_calls" twice, and the second time
                # tool_calls is already reset to {}.
                if finish_reason in ("tool_calls", "tool_use") and tool_calls:
                    finished_tool_calls = list(tool_calls.values())
                    tool_calls = {}

                content = delta.get('content', None)
                reasoning = delta.get('reasoning') or delta.get('reasoning_content')

                if content or reasoning or tool_calls:
                    yield content, reasoning, finished_tool_calls, {}
            yield None, None, finished_tool_calls, usage
