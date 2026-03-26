"""
Chain-of-thought orchestration layer for cai.

Architecture (three layers):
  TaskRunner        — state machine, owns recursion, holds list[ReasoningStrategy]
  ReasoningStrategy — pre/post hook interface (ABC)
  call_llm          — flat agentic loop, unchanged

Adding a new complexity level = new ReasoningStrategy subclass + entry in COMPLEXITY_LEVELS.
TaskRunner never needs to change.
"""

import copy
import json
import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("cai")

# ---------------------------------------------------------------------------
# Task — unit of work
# ---------------------------------------------------------------------------

@dataclass
class Task:
    messages: list          # conversation context (same list reference callers use)
    goal: str               # what this task must accomplish
    depth: int = 0          # recursion depth — guards against infinite decomposition
    parent: Optional['Task'] = None
    children: list = field(default_factory=list)
    state: str = "pending"  # pending → pre_hook → running → post_hook → done
    result: Optional[str] = None


# ---------------------------------------------------------------------------
# ReasoningStrategy — the interface that never changes
# ---------------------------------------------------------------------------

class ReasoningStrategy(ABC):
    def pre(self, task: Task, runner: 'TaskRunner'):
        """
        Called before main execution.
        - Return None  → no decomposition, proceed to call_llm directly.
                         Hook may mutate task.messages in-place.
        - Return [...] → run these subtasks first; results are injected into
                         task.messages before the main call_llm runs.
        """
        return None

    def post(self, task: Task, runner: 'TaskRunner'):
        """
        Called after main execution with task.result set.
        - Return None → keep result as-is.
        - Return str  → replace result (e.g. after verification retry).
        """
        return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = {
    "role": "user",
    "content": (
        "Should the above task be split into independent subtasks? "
        "Output ONLY a JSON array — no explanation, no markdown fences.\n"
        "Format: [\"subtask 1\", \"subtask 2\", ...]\n"
        "\n"
        "STRICT RULES — read carefully before deciding:\n"
        "- DEFAULT ANSWER IS NO DECOMPOSITION. Output [\"<original task as-is>\"] unless you are certain splitting is required.\n"
        "- Split ONLY when the task contains 2-3 clearly separate, independent concerns that cannot be handled in a single pass.\n"
        "- NEVER split a task that is a single question, a single command, or a task whose steps must be done sequentially.\n"
        "- NEVER split just to be thorough or to seem organized. Splitting has high overhead — avoid it.\n"
        "- Maximum 3 subtasks total. If you think you need more, you are over-splitting — output [\"<original task as-is>\"] instead.\n"
        "- If in doubt, do NOT split."
    ),
}

REFLECT_PROMPT = {
    "role": "user",
    "content": (
        "Review the answer above critically. "
        "Does it fully and correctly address the original task?\n"
        "Reply with exactly one of:\n"
        "SATISFACTORY\n"
        "NEEDS_REVISION: <specific issue>\n"
        "Output only one of these two formats, nothing else."
    ),
}


def _parse_subtask_json(plan: str) -> list:
    """Parse a JSON array of subtask strings from a decomposition response.

    Tries JSON first (the expected format when strict_format='json' is used).
    Falls back to extracting a numbered list for robustness.
    """
    if plan:
        try:
            parsed = json.loads(plan)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
    # fallback: numbered list (e.g. "1. subtask\n2. subtask")
    import re
    subtasks = []
    for line in plan.strip().splitlines():
        m = re.match(r'^\s*\d+[\.\)]\s+(.+)$', line)
        if m:
            subtasks.append(m.group(1).strip())
    return subtasks


def _args_for_decompose(args):
    """Return a shallow copy of args tuned for a decomposition call.

    Forces non-streaming and strict JSON output so that enforce_strict_format
    inside call_llm validates the response before it is returned.
    """
    a = copy.copy(args)
    a.strict_format = 'json'
    a.non_streaming = True
    return a


def _fmt_task_line(symbol: str, depth: int, i: int, total: int, goal: str = "") -> str:
    """Format a single task status line with tree indentation.

    depth=1 → 2 spaces, depth=2 → 4 spaces, etc.
    goal is truncated to keep lines readable.
    """
    indent = "  " * depth
    label = f"[{i}/{total}]"
    if goal:
        short = goal if len(goal) <= 60 else goal[:57] + "..."
        return f"{indent}{symbol} {label} {short}\n"
    return f"{indent}{symbol} {label} done\n"


def _format_subtask_results(results: list) -> str:
    """Format subtask (goal, result) pairs for injection into parent task messages."""
    parts = ["The task was broken into subtasks. Results:\n"]
    for i, (goal, result) in enumerate(results, 1):
        parts.append(f"**Subtask {i}**: {goal}\n**Result**: {result}\n")
    parts.append("Now synthesize these results to answer the original task.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class TaskDecomposer(ReasoningStrategy):
    """Asks the LLM to break the task into subtasks and runs each as a child Task.

    Uses strict_format='json' so call_llm's enforce_strict_format validates
    the output is a JSON array before returning it.
    """

    def pre(self, task: Task, runner: 'TaskRunner'):
        if task.depth >= runner.max_depth:
            log.info("TaskDecomposer: depth=%d >= max_depth=%d, skipping", task.depth, runner.max_depth)
            return None

        from cai.cli import call_llm  # deferred to avoid circular import at module level
        decompose_args = _args_for_decompose(runner._args)
        plan = call_llm(task.messages + [DECOMPOSE_PROMPT], decompose_args, [], {},
                        interrupt_event=runner._interrupt_event)
        # Skip decomposition if interrupted (during or after the HTTP call)
        if not plan or (runner._interrupt_event and runner._interrupt_event.is_set()):
            return None
        subtasks_text = _parse_subtask_json(plan)
        # Hard cap: never allow more than 3 subtasks regardless of what the LLM returned
        MAX_SUBTASKS = 3
        if len(subtasks_text) > MAX_SUBTASKS:
            log.warning(
                "TaskDecomposer: LLM returned %d subtasks, capping at %d",
                len(subtasks_text), MAX_SUBTASKS,
            )
            subtasks_text = subtasks_text[:MAX_SUBTASKS]
        log.info("TaskDecomposer: depth=%d parsed %d subtasks", task.depth, len(subtasks_text))

        if len(subtasks_text) < 2:
            return None  # LLM decided task shouldn't be split

        return [
            Task(
                messages=task.messages.copy() + [
                    {"role": "user", "content": f"Focus on this subtask: {g}"}
                ],
                goal=g,
                depth=task.depth + 1,
            )
            for g in subtasks_text
        ]


class AnswerVerifier(ReasoningStrategy):
    """After getting a result, asks the LLM to verify it and optionally retries once."""

    def post(self, task: Task, runner: 'TaskRunner'):
        from cai.cli import call_llm  # deferred to avoid circular import at module level
        verdict = call_llm(
            task.messages
            + [{"role": "assistant", "content": task.result}]
            + [REFLECT_PROMPT],
            runner._args,
            [],
            {},
            interrupt_event=runner._interrupt_event,
        )
        log.info("AnswerVerifier: verdict=%r", verdict[:80] if verdict else "")

        if (not verdict
                or "SATISFACTORY" in verdict
                or (runner._interrupt_event and runner._interrupt_event.is_set())):
            return None  # satisfied, interrupted, or unparseable verdict

        log.info("AnswerVerifier: revision needed, retrying")
        task.messages.append({"role": "user", "content": f"Revision needed: {verdict}"})
        revised = call_llm(
            task.messages,
            runner._args,
            runner._available_tools,
            runner._external_mcps,
            interrupt_event=runner._interrupt_event,
        )
        return revised


# ---------------------------------------------------------------------------
# Complexity levels — the only thing that changes when adding new strategies
# ---------------------------------------------------------------------------

COMPLEXITY_LEVELS = {
    0: [],                                      # default: no hooks, identical to old behavior
    1: [TaskDecomposer()],                      # task decomposition
    2: [TaskDecomposer(), AnswerVerifier()],    # decomposition + answer verification
}


# ---------------------------------------------------------------------------
# TaskRunner — the state machine
# ---------------------------------------------------------------------------

class TaskRunner:
    def __init__(self, hooks=None, max_depth: int = 3):
        self.hooks = hooks or []
        self.max_depth = max_depth
        # stored so strategies can call call_llm for their internal LLM calls
        self._args = None
        self._available_tools = None
        self._external_mcps = None
        self._interrupt_event = None  # forwarded so strategies respect Ctrl-C

    def run(self, task: Task, args, available_tools, external_mcps,
            task_callback=None, **callbacks) -> str:
        # cache for use by strategies
        self._args = args
        self._available_tools = available_tools
        self._external_mcps = external_mcps
        self._interrupt_event = callbacks.get('interrupt_event')

        # Fast-exit: if already interrupted before this task even starts, skip everything.
        if self._interrupt_event and self._interrupt_event.is_set():
            task.state = "done"
            return ""

        # Subtasks run silently — only the root response streams to the screen.
        # Replace stream_callback with a no-op instead of removing it: keeping the
        # streaming path means call_llm checks interrupt_event between chunks, so
        # Ctrl-C aborts the in-flight request on the next received chunk rather than
        # waiting for the full HTTP response to arrive.
        if task.depth > 0:
            callbacks = dict(callbacks)
            if 'stream_callback' in callbacks:
                callbacks['stream_callback'] = lambda chunk: None
            if 'tool_callback' in callbacks:
                _indent = "  " * task.depth
                _orig_tool_cb = callbacks['tool_callback']
                callbacks['tool_callback'] = (
                    lambda line, error=False, _cb=_orig_tool_cb, _ind=_indent:
                        _cb(line, error=error) if line == "\n"
                        else _cb(f"{_ind}{line}", error=error)
                )

        from cai.cli import call_llm  # deferred to avoid circular import

        # --- pre_hook: run all strategies in order ---
        task.state = "pre_hook"
        for hook in self.hooks:
            subtasks = hook.pre(task, self)
            if subtasks:
                results = []
                total = len(subtasks)
                for i, st in enumerate(subtasks, 1):
                    # Stop launching new subtasks the moment interrupt is set.
                    if self._interrupt_event and self._interrupt_event.is_set():
                        break
                    st.parent = task
                    task.children.append(st)
                    if task_callback:
                        task_callback(_fmt_task_line("▸", st.depth, i, total, st.goal))
                    r = self.run(st, args, available_tools, external_mcps,
                                 task_callback=task_callback, **callbacks)
                    if task_callback:
                        task_callback(_fmt_task_line("✓", st.depth, i, total))
                    results.append((st.goal, r))
                task.messages.append({
                    "role": "user",
                    "content": _format_subtask_results(results),
                })
                break  # once decomposed, remaining pre-hooks don't apply

        # --- running ---
        task.state = "running"
        task.result = call_llm(task.messages, args, available_tools, external_mcps, **callbacks)

        # --- post_hook: run all strategies in reverse ---
        task.state = "post_hook"
        for hook in reversed(self.hooks):
            revised = hook.post(task, self)
            if revised is not None:
                task.result = revised

        task.state = "done"
        return task.result

    @classmethod
    def from_args(cls, args) -> 'TaskRunner':
        """Build a TaskRunner from CLI args. complexity=0 → no hooks → identical to old behavior."""
        level = getattr(args, 'complexity', 0)
        hooks = list(COMPLEXITY_LEVELS.get(level, []))
        max_depth = getattr(args, 'max_depth', 3)
        log.info("TaskRunner.from_args: complexity=%d max_depth=%d hooks=%s", level, max_depth, [type(h).__name__ for h in hooks])
        return cls(hooks=hooks, max_depth=max_depth)
