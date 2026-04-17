import json

from cai import Harness


harness = Harness(system_prompt="", log_path="/tmp/cai/cai.log") # do not use any defaults

r = harness.agent(  system_prompt="return a list of items only",
                    prompt="list all functions in this project",
                    strict_format=r"regex-each-line:^(-).*$",
                    skills=['files'])
r.wait()

for function in r.text.splitlines():
    r = harness.gate(   options=['yes', 'no'],
                        skills=['files'],
                        prompt=f"does the '{function}' function interacting with file system in some way?")
    print(f"{function} -> {r}")

# cloned = harness.clone()
# for _ in range(3):
    # r = harness.agent(  system_prompt="you are an expert python source writer",
                        # prompt="return a list of all functions in this project",
                        # skills=['files'])
    # r.wait()
    # cloned.enrich(r.messages)

# r = cloned.agent(system_prompt="you are an expert python source writer",
                 # prompt="summerazie all found functions into one comprehensive list")
# r.wait()

# print(r.text)
# # harness.enrich(r.text)


# def add(a: int, b: int) -> int:
    # """
    # adding a to b, returning sum.
    # """
    # return a + b

# r = harness.agent(functions=[add], prompt="add 1 and 789")
# print(r.text)
