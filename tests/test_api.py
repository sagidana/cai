"""Tests for cai.api - the Layer 0 LLM HTTP client.

Fully offline: requests.post is monkeypatched with a fake that records the
outgoing request and replays a canned blocking body or SSE line stream. No
network, no API key, no real provider.
"""
import json
import threading

import pytest
import requests

import cai.api as api
from cai.api import ApiError, OpenAiApi, _wire_messages


# --------------------------------------------------------------------------
# fakes / helpers
# --------------------------------------------------------------------------

class FakeResponse:
    """Stand-in for a requests.Response covering both the blocking
    (status_code/json) and streaming (iter_lines/context-manager) paths."""

    def __init__(self, status_code=200, body=None, lines=None, raise_json=False):
        self.status_code = status_code
        self.text = "error-body"
        self._body = body
        self._lines = lines or []
        self._raise_json = raise_json
        self.closed = False

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._body

    def iter_lines(self):
        for line in self._lines:
            yield line

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


class Recorder:
    """A fake requests.post that records calls and returns a programmed
    response (or raises a programmed exception). `script` replays a sequence -
    each item a response or an exception, one per call - so a retry path can
    fail first and succeed later."""

    def __init__(self, response=None, exc=None, script=None):
        self.response = response
        self.exc = exc
        self.script = list(script or [])
        self.calls = []

    def __call__(self, url, **kwargs):
        record = {}
        record['url'] = url
        record['kwargs'] = kwargs
        self.calls.append(record)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step
        if self.exc is not None:
            raise self.exc
        return self.response

    @property
    def last(self):
        return self.calls[-1]

    @property
    def data(self):
        """The JSON payload of the last request."""
        return self.calls[-1]['kwargs']['json']


def sse(obj):
    """Encode a dict as one SSE `data: ` line, the way a provider streams it."""
    return b"data: " + json.dumps(obj).encode()


def install(monkeypatch, response=None, exc=None, script=None):
    rec = Recorder(response=response, exc=exc, script=script)
    monkeypatch.setattr(api.requests, "post", rec)
    # retries wait between attempts; a test never should.
    monkeypatch.setattr(api, "_RETRY_BACKOFF", 0)
    return rec


def client():
    return OpenAiApi("https://example.test/v1", "sk-test")


def blocking_body(content="hi", reasoning=None, tool_calls=None, usage=None, choices=None):
    message = {}
    message['content'] = content
    if reasoning is not None:
        message['reasoning'] = reasoning
    message['tool_calls'] = tool_calls
    body = {}
    if choices is None:
        choices = [{"message": message}]
    body['choices'] = choices
    if usage is not None:
        body['usage'] = usage
    return body


# --------------------------------------------------------------------------
# constructor
# --------------------------------------------------------------------------

def test_timeout_list_coerced_to_tuple():
    c = OpenAiApi("u", "k", timeout=[5, 30])
    assert c.timeout == (5, 30)
    assert isinstance(c.timeout, tuple)


def test_ssl_verify_default_true():
    c = OpenAiApi("u", "k")
    assert c.ssl_verify is True


def test_ssl_verify_false_stored():
    c = OpenAiApi("u", "k", ssl_verify=False)
    assert c.ssl_verify is False


# --------------------------------------------------------------------------
# _wire_messages
# --------------------------------------------------------------------------

def test_wire_strips_underscore_keys():
    out = _wire_messages([{"role": "user", "content": "hi", "_display": "x", "_reasoning": "y"}])
    assert out == [{"role": "user", "content": "hi"}]


def test_wire_keeps_clean_messages_untouched():
    msg = {"role": "assistant", "content": "ok"}
    out = _wire_messages([msg])
    assert out == [msg]


def test_wire_passes_non_dicts_through():
    out = _wire_messages(["raw", 7, {"role": "user", "content": "hi"}])
    assert out == ["raw", 7, {"role": "user", "content": "hi"}]


def test_wire_does_not_mutate_input():
    original = {"role": "user", "content": "hi", "_display": "x"}
    _wire_messages([original])
    assert "_display" in original


def test_wire_handles_empty():
    assert _wire_messages([]) == []


# --------------------------------------------------------------------------
# request building (observed via the recorded payload)
# --------------------------------------------------------------------------

def test_request_sets_model_and_messages(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([{"role": "user", "content": "hi"}], "my-model")
    assert rec.data['model'] == "my-model"
    assert rec.data['messages'] == [{"role": "user", "content": "hi"}]


def test_request_prepends_system_prompt(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    sysmsg = {"role": "system", "content": "be brief"}
    client().chat([{"role": "user", "content": "hi"}], "m", system_prompt=sysmsg)
    assert rec.data['messages'][0] == sysmsg
    assert rec.data['messages'][1]['content'] == "hi"


def test_request_omits_system_prompt_when_none(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([{"role": "user", "content": "hi"}], "m")
    assert len(rec.data['messages']) == 1


def test_request_includes_tools_and_tool_choice(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    tools = [{"type": "function", "function": {"name": "f"}}]
    client().chat([], "m", tools=tools, tool_choice="required")
    assert rec.data['tools'] == tools
    assert rec.data['tool_choice'] == "required"


def test_request_omits_tools_when_absent(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([], "m")
    assert 'tools' not in rec.data
    assert 'tool_choice' not in rec.data


def test_request_reasoning_effort_sends_both_shapes(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([], "m", reasoning_effort="high")
    assert rec.data['reasoning'] == {"effort": "high"}
    assert rec.data['reasoning_effort'] == "high"


def test_request_omits_reasoning_when_absent(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([], "m")
    assert 'reasoning' not in rec.data
    assert 'reasoning_effort' not in rec.data


def test_request_temperature_included(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([], "m", temperature=0.7)
    assert rec.data['temperature'] == 0.7


def test_request_temperature_zero_is_sent(monkeypatch):
    # 0.0 is a real temperature - `is not None` must let it through.
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([], "m", temperature=0.0)
    assert rec.data['temperature'] == 0.0


def test_request_temperature_omitted_when_none(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([], "m")
    assert 'temperature' not in rec.data


def test_request_strips_local_keys_from_messages(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([{"role": "user", "content": "hi", "_display": "x"}], "m")
    assert rec.data['messages'] == [{"role": "user", "content": "hi"}]


def test_request_headers_and_verify_and_timeout(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    OpenAiApi("https://example.test/v1", "sk-abc", ssl_verify=False, timeout=(3, 9)).chat([], "m")
    kwargs = rec.last['kwargs']
    assert kwargs['headers']['Authorization'] == "Bearer sk-abc"
    assert kwargs['headers']['Content-Type'] == "application/json"
    assert kwargs['verify'] is False
    assert kwargs['timeout'] == (3, 9)


def test_request_url_built_from_base(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([], "m")
    assert rec.last['url'] == "https://example.test/v1/chat/completions"


def test_blocking_does_not_set_stream(monkeypatch):
    rec = install(monkeypatch, FakeResponse(body=blocking_body()))
    client().chat([], "m")
    assert 'stream' not in rec.data


def test_streaming_sets_stream_and_usage_option(monkeypatch):
    rec = install(monkeypatch, FakeResponse(lines=[b"data: [DONE]"]))
    list(client().chat([], "m", stream=True))
    assert rec.data['stream'] is True
    assert rec.data['stream_options'] == {"include_usage": True}


# --------------------------------------------------------------------------
# blocking path (_complete)
# --------------------------------------------------------------------------

def test_blocking_happy_path(monkeypatch):
    body = blocking_body(content="hello", usage={"total_tokens": 5})
    install(monkeypatch, FakeResponse(body=body))
    out = client().chat([{"role": "user", "content": "hi"}], "m")
    assert out == ("hello", "", None, {"total_tokens": 5})


def test_blocking_reasoning_field(monkeypatch):
    install(monkeypatch, FakeResponse(body=blocking_body(reasoning="because")))
    content, reasoning, tool_calls, usage = client().chat([], "m")
    assert reasoning == "because"


def test_blocking_reasoning_content_fallback(monkeypatch):
    body = {"choices": [{"message": {"content": "x", "reasoning_content": "deep"}}]}
    install(monkeypatch, FakeResponse(body=body))
    _, reasoning, _, _ = client().chat([], "m")
    assert reasoning == "deep"


def test_blocking_content_defaults_empty(monkeypatch):
    body = {"choices": [{"message": {"tool_calls": None}}]}
    install(monkeypatch, FakeResponse(body=body))
    content, _, _, _ = client().chat([], "m")
    assert content == ""


def test_blocking_tool_calls_returned(monkeypatch):
    tcs = [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    install(monkeypatch, FakeResponse(body=blocking_body(tool_calls=tcs)))
    _, _, tool_calls, _ = client().chat([], "m")
    assert tool_calls == tcs


def test_blocking_usage_defaults_empty(monkeypatch):
    install(monkeypatch, FakeResponse(body=blocking_body(usage=None)))
    _, _, _, usage = client().chat([], "m")
    assert usage == {}


def test_blocking_connection_error_retried_then_raises(monkeypatch):
    rec = install(monkeypatch, exc=requests.ConnectionError("boom"))
    with pytest.raises(ApiError) as err:
        client().chat([], "m")
    assert err.value.status is None
    assert len(rec.calls) == 3          # the default retries=3 attempts


def test_blocking_timeout_retried_then_raises(monkeypatch):
    rec = install(monkeypatch, exc=requests.Timeout("slow"))
    with pytest.raises(ApiError):
        client().chat([], "m")
    assert len(rec.calls) == 3


def test_blocking_500_retried_then_succeeds(monkeypatch):
    script = [FakeResponse(status_code=500), FakeResponse(body=blocking_body(content="ok"))]
    rec = install(monkeypatch, script=script)
    content, _, _, _ = client().chat([], "m")
    assert content == "ok"
    assert len(rec.calls) == 2


def test_blocking_429_retried_then_succeeds(monkeypatch):
    script = [FakeResponse(status_code=429), FakeResponse(body=blocking_body(content="ok"))]
    rec = install(monkeypatch, script=script)
    content, _, _, _ = client().chat([], "m")
    assert content == "ok"
    assert len(rec.calls) == 2


def test_blocking_non_200_exhausts_and_raises_with_status(monkeypatch):
    rec = install(monkeypatch, FakeResponse(status_code=500))
    with pytest.raises(ApiError) as err:
        client().chat([], "m")
    assert err.value.status == 500
    assert len(rec.calls) == 3


def test_blocking_400_not_retried(monkeypatch):
    rec = install(monkeypatch, FakeResponse(status_code=400))
    with pytest.raises(ApiError) as err:
        client().chat([], "m")
    assert err.value.status == 400
    assert len(rec.calls) == 1          # permanent: a retry cannot fix it


def test_retries_floor_is_one_attempt(monkeypatch):
    rec = install(monkeypatch, FakeResponse(status_code=500))
    with pytest.raises(ApiError):
        OpenAiApi("https://example.test/v1", "sk-test", retries=0).chat([], "m")
    assert len(rec.calls) == 1


def test_blocking_invalid_json_raises_without_retry(monkeypatch):
    rec = install(monkeypatch, FakeResponse(raise_json=True))
    with pytest.raises(ApiError):
        client().chat([], "m")
    assert len(rec.calls) == 1          # the POST succeeded; the body is broken


def test_blocking_zero_choices_raises(monkeypatch):
    install(monkeypatch, FakeResponse(body={"choices": []}))
    with pytest.raises(ApiError):
        client().chat([], "m")


def test_blocking_multiple_choices_raises(monkeypatch):
    body = {"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}]}
    install(monkeypatch, FakeResponse(body=body))
    with pytest.raises(ApiError):
        client().chat([], "m")


def test_blocking_missing_message_raises(monkeypatch):
    install(monkeypatch, FakeResponse(body={"choices": [{}]}))
    with pytest.raises(ApiError):
        client().chat([], "m")


# --------------------------------------------------------------------------
# streaming path (_stream)
# --------------------------------------------------------------------------

def drain(gen):
    out = []
    for delta in gen:
        out.append(delta)
    return out


def test_streaming_returns_generator(monkeypatch):
    install(monkeypatch, FakeResponse(lines=[b"data: [DONE]"]))
    gen = client().chat([], "m", stream=True)
    assert hasattr(gen, "__next__")


def test_streaming_content_deltas(monkeypatch):
    lines = []
    lines.append(sse({"choices": [{"delta": {"content": "he"}, "finish_reason": None}]}))
    lines.append(sse({"choices": [{"delta": {"content": "llo"}, "finish_reason": None}]}))
    lines.append(b"data: [DONE]")
    install(monkeypatch, FakeResponse(lines=lines))
    deltas = drain(client().chat([], "m", stream=True))
    texts = []
    for content, _, _, _ in deltas:
        if content:
            texts.append(content)
    assert "".join(texts) == "hello"


def test_streaming_reasoning_delta(monkeypatch):
    lines = []
    lines.append(sse({"choices": [{"delta": {"reasoning": "think"}, "finish_reason": None}]}))
    lines.append(b"data: [DONE]")
    install(monkeypatch, FakeResponse(lines=lines))
    deltas = drain(client().chat([], "m", stream=True))
    assert deltas[0][1] == "think"


def test_streaming_reasoning_content_delta(monkeypatch):
    lines = []
    lines.append(sse({"choices": [{"delta": {"reasoning_content": "deep"}, "finish_reason": None}]}))
    lines.append(b"data: [DONE]")
    install(monkeypatch, FakeResponse(lines=lines))
    deltas = drain(client().chat([], "m", stream=True))
    assert deltas[0][1] == "deep"


def test_streaming_assembles_tool_call_fragments(monkeypatch):
    lines = []
    lines.append(sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "t1", "function": {"name": "do", "arguments": "{\"a\":"}}]}, "finish_reason": None}]}))
    lines.append(sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"}]}))
    lines.append(b"data: [DONE]")
    install(monkeypatch, FakeResponse(lines=lines))
    final = drain(client().chat([], "m", stream=True))[-1]
    tool_calls = final[2]
    assert len(tool_calls) == 1
    assert tool_calls[0]['id'] == "t1"
    assert tool_calls[0]['type'] == "function"
    assert tool_calls[0]['function']['name'] == "do"
    assert tool_calls[0]['function']['arguments'] == '{"a":1}'


def test_streaming_multiple_tool_calls(monkeypatch):
    lines = []
    lines.append(sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "t1", "function": {"name": "f0", "arguments": "{}"}}]}, "finish_reason": None}]}))
    lines.append(sse({"choices": [{"delta": {"tool_calls": [
        {"index": 1, "id": "t2", "function": {"name": "f1", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}))
    lines.append(b"data: [DONE]")
    install(monkeypatch, FakeResponse(lines=lines))
    final = drain(client().chat([], "m", stream=True))[-1]
    tool_calls = final[2]
    assert len(tool_calls) == 2
    assert tool_calls[0]['function']['name'] == "f0"
    assert tool_calls[1]['function']['name'] == "f1"


def test_streaming_tool_use_finish_reason(monkeypatch):
    # Anthropic-native finish_reason="tool_use" must also snapshot.
    lines = []
    lines.append(sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "t1", "function": {"name": "f", "arguments": "{}"}}]}, "finish_reason": "tool_use"}]}))
    lines.append(b"data: [DONE]")
    install(monkeypatch, FakeResponse(lines=lines))
    final = drain(client().chat([], "m", stream=True))[-1]
    assert final[2] is not None
    assert len(final[2]) == 1


def test_streaming_double_finish_does_not_clobber(monkeypatch):
    # some providers fire finish_reason="tool_calls" twice; the second time
    # tool_calls is already reset, so the snapshot must survive.
    lines = []
    lines.append(sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "t1", "function": {"name": "f", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}))
    lines.append(sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}))
    lines.append(b"data: [DONE]")
    install(monkeypatch, FakeResponse(lines=lines))
    final = drain(client().chat([], "m", stream=True))[-1]
    assert final[2] is not None
    assert len(final[2]) == 1


def test_streaming_final_yield_carries_usage(monkeypatch):
    lines = []
    lines.append(sse({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}))
    lines.append(sse({"choices": [], "usage": {"total_tokens": 42}}))
    lines.append(b"data: [DONE]")
    install(monkeypatch, FakeResponse(lines=lines))
    final = drain(client().chat([], "m", stream=True))[-1]
    assert final == (None, None, None, {"total_tokens": 42})


def test_streaming_skips_invalid_json_line(monkeypatch):
    lines = []
    lines.append(b"data: {not json")
    lines.append(sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}))
    lines.append(b"data: [DONE]")
    install(monkeypatch, FakeResponse(lines=lines))
    deltas = drain(client().chat([], "m", stream=True))
    assert deltas[0][0] == "ok"


def test_streaming_ignores_non_data_lines(monkeypatch):
    lines = []
    lines.append(b": keep-alive comment")
    lines.append(b"")
    lines.append(sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}))
    lines.append(b"data: [DONE]")
    install(monkeypatch, FakeResponse(lines=lines))
    deltas = drain(client().chat([], "m", stream=True))
    assert deltas[0][0] == "ok"


def test_streaming_done_terminates_before_trailing_lines(monkeypatch):
    lines = []
    lines.append(sse({"choices": [{"delta": {"content": "a"}, "finish_reason": None}]}))
    lines.append(b"data: [DONE]")
    lines.append(sse({"choices": [{"delta": {"content": "ignored"}, "finish_reason": None}]}))
    install(monkeypatch, FakeResponse(lines=lines))
    deltas = drain(client().chat([], "m", stream=True))
    texts = []
    for content, _, _, _ in deltas:
        if content:
            texts.append(content)
    assert texts == ["a"]


def test_streaming_non_200_exhausts_and_raises(monkeypatch):
    resp = FakeResponse(status_code=500)
    rec = install(monkeypatch, resp)
    with pytest.raises(ApiError) as err:
        drain(client().chat([], "m", stream=True))
    assert err.value.status == 500
    assert resp.closed is True
    assert len(rec.calls) == 3


def test_streaming_non_200_retried_then_streams(monkeypatch):
    lines = []
    lines.append(sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}))
    lines.append(b"data: [DONE]")
    rec = install(monkeypatch, script=[FakeResponse(status_code=503), FakeResponse(lines=lines)])
    deltas = drain(client().chat([], "m", stream=True))
    assert deltas[0][0] == "ok"
    assert len(rec.calls) == 2


def test_streaming_connection_error_retried_then_raises(monkeypatch):
    rec = install(monkeypatch, exc=requests.ConnectionError("boom"))
    with pytest.raises(ApiError):
        drain(client().chat([], "m", stream=True))
    assert len(rec.calls) == 3


def test_streaming_mid_stream_drop_raises_without_retry(monkeypatch):
    # once bytes flowed a retry would replay output the consumer already saw:
    # the partial deltas arrive, then the drop surfaces as ApiError, one POST.
    def lines_then_die():
        yield sse({"choices": [{"delta": {"content": "par"}, "finish_reason": None}]})
        raise requests.ConnectionError("reset mid-stream")

    resp = FakeResponse()
    resp.iter_lines = lines_then_die
    rec = install(monkeypatch, resp)
    gen = client().chat([], "m", stream=True)
    first = next(gen)
    assert first[0] == "par"
    with pytest.raises(ApiError):
        next(gen)
    assert len(rec.calls) == 1


def test_streaming_empty_stream_yields_final_only(monkeypatch):
    install(monkeypatch, FakeResponse(lines=[b"data: [DONE]"]))
    deltas = drain(client().chat([], "m", stream=True))
    assert deltas == [(None, None, None, {})]


# --------------------------------------------------------------------------
# error propagation through call_llm
# --------------------------------------------------------------------------

class RaisingApi:
    """an api whose every chat raises ApiError (the exhausted-retries case)."""

    def chat(self, messages, model, **kwargs):
        raise ApiError("provider down", status=502)


def test_call_llm_propagates_api_error():
    from cai.llm import call_llm

    gen = call_llm([{"role": "user", "content": "hi"}], "m", RaisingApi())
    with pytest.raises(ApiError) as err:
        next(gen)
    assert err.value.status == 502


def test_call_llm_failed_turn_leaves_messages_untouched():
    # a failed call must not read as the model answering "": nothing from the
    # failed turn is appended, so a resubmit starts from the pre-turn state.
    from cai.llm import call_llm

    messages = [{"role": "user", "content": "hi"}]
    gen = call_llm(messages, "m", RaisingApi(), stream=False)
    with pytest.raises(ApiError):
        next(gen)
    assert messages == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------
# interrupt - the polled paths
# --------------------------------------------------------------------------

class BlockingLinesResponse(FakeResponse):
    """streams `lines`, then blocks like a provider that went quiet; close()
    releases the block the way a real socket shutdown wakes a recv."""

    def __init__(self, lines):
        super().__init__(lines=lines)
        self._released = threading.Event()

    def iter_lines(self):
        for line in self._lines:
            yield line
        self._released.wait()
        raise requests.ConnectionError("shut down")

    def close(self):
        self._released.set()
        super().close()


def test_interrupted_blocking_call_returns_empty(monkeypatch):
    install(monkeypatch, FakeResponse(body=blocking_body(content="late")))
    interrupt = threading.Event()
    interrupt.set()
    out = client().chat([], "m", interrupt=interrupt)
    assert out == ("", "", None, {})


def test_uninterrupted_blocking_call_matches_direct_path(monkeypatch):
    body = blocking_body(content="hello", usage={"total_tokens": 5})
    install(monkeypatch, FakeResponse(body=body))
    out = client().chat([], "m", interrupt=threading.Event())
    assert out == ("hello", "", None, {"total_tokens": 5})


def test_polled_blocking_call_propagates_api_error(monkeypatch):
    install(monkeypatch, FakeResponse(status_code=400))
    with pytest.raises(ApiError) as err:
        client().chat([], "m", interrupt=threading.Event())
    assert err.value.status == 400


def test_uninterrupted_stream_matches_direct_path(monkeypatch):
    lines = [sse({"choices": [{"delta": {"content": "hel"}}]}),
             sse({"choices": [{"delta": {"content": "lo"}}]}),
             b"data: [DONE]"]
    install(monkeypatch, FakeResponse(lines=lines))
    out = list(client().chat([], "m", stream=True, interrupt=threading.Event()))
    assert out == [("hel", None, None, {}),
                   ("lo", None, None, {}),
                   (None, None, None, {})]


def test_interrupt_mid_stream_ends_generator_and_closes_response(monkeypatch):
    response = BlockingLinesResponse([sse({"choices": [{"delta": {"content": "partial"}}]})])
    install(monkeypatch, response)
    interrupt = threading.Event()
    gen = client().chat([], "m", stream=True, interrupt=interrupt)
    content, _, _, _ = next(gen)          # first chunk arrives normally...
    assert content == "partial"
    interrupt.set()                       # ...then the kill lands while the
    assert list(gen) == []                # pump is blocked waiting for more
    assert response.closed is True        # and the abort tore the response down


def test_interrupted_stream_error_is_swallowed(monkeypatch):
    # the abort makes the pump's blocked read raise; that error must die with
    # the pump, not surface after the consumer already walked away.
    response = BlockingLinesResponse([])
    install(monkeypatch, response)
    interrupt = threading.Event()
    gen = client().chat([], "m", stream=True, interrupt=interrupt)
    interrupt.set()
    assert list(gen) == []


def test_interrupt_during_retry_backoff_stops_retrying(monkeypatch):
    rec = install(monkeypatch, exc=requests.ConnectionError("down"))
    interrupt = threading.Event()
    interrupt.set()
    url = "https://example.test/v1/chat/completions"
    with pytest.raises(ApiError):
        client()._post_with_retry(url, {}, {}, stream=False, interrupt=interrupt)
    assert len(rec.calls) == 1            # gave up instead of retrying
