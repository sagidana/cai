import json

from cai import Harness


harness = Harness(system_prompt="", log_path="/tmp/cai/cai.log") # do not use any defaults

r = harness.agent(  system_prompt="",
                    prompt="list current files",
                    skills=['files'])
r.wait()

harness.enrich(r.messages)

r = harness.gate(options=["yes", "no"], prompt="is that enough?")
print(r)
