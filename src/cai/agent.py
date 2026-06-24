"""agent: a persistent conversation you can run prompts against.

The minimal session object. It owns a growing `messages` list, a model, and an
api client; `run(prompt)` drives the agentic loop over that conversation - each
call appends the user turn, runs to a final answer, and leaves the new assistant
(and any tool) messages in `messages`, so the next run continues where this one
left off.

This is the smallest useful Agent. Tools, hooks, skills, serving/attach,
saving, and cloning are later layers."""
from __future__ import annotations

from cai import config
from cai.api import OpenAiApi
from cai.llm import call_llm


class Agent:
    def __init__(self, *, model=None, system_prompt=None, api=None):
        if model is None:
            model = config.DEFAULT_MODEL
        if api is None:
            api = OpenAiApi(config.OPENROUTER_BASE_URL, config.load_api_key())
        self.model = model
        self.system_prompt = system_prompt
        self.api = api
        self.messages = []

    def run(self, prompt=None):
        """append `prompt` as a user turn (if given), run the loop to a final
        answer, and return it. `messages` keeps the full evolving transcript."""
        if prompt is not None:
            self.messages.append({"role": "user", "content": prompt})
        # stream=False: run() returns the final string and surfaces no events,
        # so there is nothing to stream to - a single blocking call is simpler.
        gen = call_llm(self.messages,
                       self.model,
                       self.api,
                       system_prompt=self.system_prompt,
                       stream=False)
        final = ""
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            final = stop.value
        return final
