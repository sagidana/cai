import json

from cai import Harness


harness = Harness(system_prompt="", log_path="/tmp/cai/cai.log") # do not use any defaults

cloned = harness.clone()

def add(a: int, b: int):
    """
    adding a to b, returning sum.
    """
    return a + b
for _ in range(3):
    r = harness.agent(  system_prompt="you are an expert python source writer",
                        prompt="return a list of all functions in this project",
                        skills=['files'])
    r.wait()
    cloned.enrich(r.messages)

r = cloned.agent(system_prompt="you are an expert python source writer",
                 prompt="summerazie all found functions into one comprehensive list")
r.wait()

print(r.text)
# harness.enrich(r.text)
