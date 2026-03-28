#!/usr/bin/env python3
"""
generate_log.py — Continuously write nested log entries to /tmp/cai/cai.log.

Run this in one terminal while `cai --logger` is open in another to test
the TUI interactively.  Runs until Ctrl-C.
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from cai.logger import init, log

TOOL_NAMES = [
    "read_file", "write_file", "execute_code",
    "search_files", "run_tests", "list_directory",
    "grep_codebase", "fetch_url",
]

OUTCOMES = [
    "success",
    "success (128 bytes written)",
    "success (cached, 0ms)",
    "error: file not found",
    "error: permission denied",
    "timeout after 5 s",
    "warning: output truncated",
]

MODELS = ["gpt-4o", "claude-3-5-sonnet", "deepseek-r1", "gemma-3-27b"]

LOREM_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "sphinx of black quartz judge my vow pack my box with five "
    "dozen liquor jugs how vexingly quick daft zebras jump"
).split()


LONG_PROMPTS = [
    (
        "You are a senior software engineer reviewing a pull request. "
        "The diff touches authentication middleware, session token storage, "
        "and three downstream services. Please analyse the security implications, "
        "identify any OWASP Top-10 risks, suggest concrete mitigations for each, "
        "and comment on whether the test coverage is adequate for a production merge."
    ),
    (
        "Refactor the following Python function so that it:\n"
        "  1. Handles all edge cases (empty input, None, negative numbers)\n"
        "  2. Uses a generator instead of building a list in memory\n"
        "  3. Adds proper type annotations\n"
        "  4. Is compatible with Python 3.9+\n\n"
        "def process(data):\n"
        "    result = []\n"
        "    for item in data:\n"
        "        if item > 0:\n"
        "            result.append(item * 2)\n"
        "    return result"
    ),
    (
        "Given the following stack trace, identify the root cause and propose a fix:\n\n"
        "Traceback (most recent call last):\n"
        "  File \"/app/server.py\", line 142, in handle_request\n"
        "    response = pipeline.run(request.body)\n"
        "  File \"/app/pipeline.py\", line 88, in run\n"
        "    result = self._transform(payload)\n"
        "  File \"/app/pipeline.py\", line 61, in _transform\n"
        "    return json.loads(payload.decode('utf-8'))\n"
        "  File \"/usr/lib/python3.11/json/__init__.py\", line 346, in loads\n"
        "    return _default_decoder.decode(s)\n"
        "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 3: "
        "invalid start byte\n\n"
        "The endpoint receives binary uploads from mobile clients running iOS 15 and Android 12. "
        "The issue appears intermittently under high load."
    ),
    (
        "Write a comprehensive technical design document for a distributed rate-limiter "
        "that must satisfy the following requirements:\n"
        "  - Sub-millisecond decision latency at p99\n"
        "  - Support for 50 000 unique API keys\n"
        "  - Sliding-window semantics with configurable window sizes (1 s – 1 h)\n"
        "  - Multi-region active-active deployment with eventual consistency\n"
        "  - Graceful degradation when the backing store is unreachable\n"
        "  - Observable via Prometheus metrics and structured JSON logs\n\n"
        "Include sections on: data model, consistency trade-offs, failure modes, "
        "capacity planning, and a comparison of Redis vs. Cassandra as the backing store."
    ),
]

MULTILINE_RESULTS = [
    (
        "stdout (42 lines):\n"
        "  line 1: initialising subsystem A\n"
        "  line 2: loading config from /etc/app/config.yaml\n"
        "  line 3: config parsed — 17 keys loaded\n"
        "  line 4: connecting to database at 10.0.1.5:5432\n"
        "  line 5: connection pool established (min=2, max=10)\n"
        "  ...\n"
        "  line 40: all health checks passed\n"
        "  line 41: server listening on :8080\n"
        "  line 42: startup complete in 1.34 s"
    ),
    (
        "File contents (src/cai/pipeline.py):\n"
        "  001: import asyncio\n"
        "  002: import logging\n"
        "  003: from typing import AsyncIterator, Optional\n"
        "  004: \n"
        "  005: log = logging.getLogger(__name__)\n"
        "  006: \n"
        "  007: class Pipeline:\n"
        "  008:     def __init__(self, stages: list) -> None:\n"
        "  009:         self.stages = stages\n"
        "  010:         self._running = False\n"
        "  ...\n"
        "  089:     async def run(self, payload: bytes) -> dict:\n"
        "  090:         for stage in self.stages:\n"
        "  091:             payload = await stage.process(payload)\n"
        "  092:         return {'result': payload, 'ok': True}"
    ),
    (
        "grep matches (23 hits across 7 files):\n"
        "  src/cai/cli.py:142:    log.info('starting session')\n"
        "  src/cai/cli.py:388:    log.warning('context limit approaching')\n"
        "  src/cai/api.py:77:    log.error('API call failed: %s', err)\n"
        "  src/cai/tools.py:201:    log.info('tool %s returned %d bytes', name, len(result))\n"
        "  tests/test_cli.py:55:    assert log_output.contains('starting session')\n"
        "  tests/test_api.py:33:    mock_log.assert_called_with('API call failed: %s', ANY)\n"
        "  docs/logging.md:12:  log.info / log.warning / log.error are the approved levels"
    ),
]


def rand_sentence(n: int = 8) -> str:
    return " ".join(random.choices(LOREM_WORDS, k=n))


def rand_long_prompt() -> str:
    return random.choice(LONG_PROMPTS)


def rand_multiline_result() -> str:
    return random.choice(MULTILINE_RESULTS)


def rand_path() -> str:
    dirs  = ["src/cai", "tests", "harnesses", "/tmp/cai", "docs"]
    names = ["main.py", "utils.py", "result.json", "config.yaml", "output.txt"]
    return f"{random.choice(dirs)}/{random.choice(names)}"


def simulate_tool_call(turn: int, tool_idx: int, n_tools: int) -> None:
    tool = random.choice(TOOL_NAMES)
    log(2, f"Tool call [{tool_idx}/{n_tools}]: {tool}")
    time.sleep(random.uniform(0.05, 0.15))

    log(3, f'Arguments: {{"path": "{rand_path()}", "turn": {turn}}}')
    time.sleep(random.uniform(0.03, 0.08))

    # Occasionally go deeper
    if random.random() < 0.45:
        n_substeps = random.randint(1, 4)
        log(3, f"Executing {n_substeps} sub-step(s)…")
        time.sleep(0.05)
        for step in range(1, n_substeps + 1):
            chunk_id = random.randint(100, 9999)
            log(4, f"Sub-step {step}/{n_substeps}: processing chunk #{chunk_id}")
            time.sleep(random.uniform(0.02, 0.07))

            # Occasionally go even deeper
            if random.random() < 0.30:
                log(5, f"Detail: checksum=0x{random.randint(0, 0xFFFF):04X}  "
                       f"size={random.randint(64, 4096)} B")
                time.sleep(0.03)
                if random.random() < 0.25:
                    log(6, f"Byte-level trace: {rand_sentence(6)}")
                    time.sleep(0.02)

    # Occasionally emit a multiline tool result
    if random.random() < 0.25:
        log(3, rand_multiline_result())
        time.sleep(random.uniform(0.05, 0.10))
    else:
        outcome = random.choice(OUTCOMES)
        log(3, f"Result: {outcome}")
    time.sleep(random.uniform(0.05, 0.12))


def simulate_turn(turn: int) -> None:
    model = random.choice(MODELS)
    log(1, f"=== Turn {turn} | model={model} ===")
    time.sleep(random.uniform(0.1, 0.3))

    # LLM reasoning (level 2) — occasionally a long multi-line prompt
    if random.random() < 0.5:
        if random.random() < 0.30:
            log(2, f"Prompt:\n{rand_long_prompt()}")
        else:
            log(2, f"Reasoning: {rand_sentence(12)}")
        time.sleep(random.uniform(0.05, 0.15))

    # Tool calls
    n_tools = random.randint(0, 4)
    for t in range(1, n_tools + 1):
        simulate_tool_call(turn, t, n_tools)

    if n_tools == 0:
        log(2, "No tool calls — direct response")
        time.sleep(0.1)

    # LLM response tokens
    prompt_tok  = random.randint(800, 4000)
    output_tok  = random.randint(50, 600)
    log(2, f"LLM response: {output_tok} tokens out  ({prompt_tok} in context)")
    time.sleep(random.uniform(0.05, 0.15))

    # Occasionally a long top-level annotation or full prompt replay
    if random.random() < 0.15:
        if random.random() < 0.40:
            log(1, "User prompt:\n" + rand_long_prompt())
        else:
            log(1, "Context summary: " + rand_sentence(25))
        time.sleep(0.1)

    time.sleep(random.uniform(0.3, 1.2))


def main() -> None:
    init()
    print(f"Writing to /tmp/cai/cai.log — open 'cai --logger' in another terminal.")
    print("Press Ctrl-C to stop.\n")

    turn = 0
    try:
        while True:
            turn += 1
            simulate_turn(turn)
    except KeyboardInterrupt:
        print(f"\nStopped after {turn} turn(s).")


if __name__ == "__main__":
    main()
