"""events: the value the agentic loop yields to its consumer.

One Event per thing that happened during a run. The loop in cai.llm yields
these as it runs; a consumer iterates them to stream assistant text, render
tool calls, surface token usage, etc. EventType lists every type the loop
currently emits - it grows as later layers add new kinds of output."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    """Every event type the loop currently emits, in one place. A str-Enum, so
    a consumer can match the member (EventType.CONTENT) or the plain string
    ('content') interchangeably - they compare equal."""
    CONTENT = "content"          # a chunk of the assistant's answer (text)
    REASONING = "reasoning"      # a chunk of the model's thinking (text)
    TOOL_CALL = "tool_call"      # the model asked to run a tool
    TOOL_RESULT = "tool_result"  # a tool finished
    USAGE = "usage"              # token accounting for a turn (usage)

    def __str__(self):
        return self.value


@dataclass
class Event:
    """One thing that happened during a run, handed to the consumer. Fields
    that don't apply to the event's type are left at their defaults.

      content     - text
      reasoning   - text
      tool_call   - tool_name / tool_args / tool_call_id
      tool_result - tool_name / tool_result / tool_call_id / is_error
      usage       - usage
    """
    type: str
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_call_id: Optional[str] = None
    tool_result: Optional[str] = None
    is_error: bool = False
    usage: Optional[dict] = None
