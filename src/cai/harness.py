"""
harness.py — orchestration layer for cai.

Parses .harness.cai files and executes them by calling call_llm directly,
managing the messages list to selectively enrich global context per block.

Usage (via cli.py --harness flag):
    cai --harness my.harness.cai -- "user task here"

harness.cai format overview:

    label:              # jump target

    ---                 # opens a block
        --name "x"          # required
        --enrich-global-context   # or --dont-enrich-global-context (required)
        --prepend-user-prompt
        --tools read, list_files
        --model gpt-4o
        --max-turns 50
        --strict-format "regex:^(ok|retry)$"
        --system-prompt "You are a concise assistant."
        --force-tools
        '''
        Prompt text goes here.
        '''
    ---

    if x == ok: goto label
    goto label
    exit
"""

import copy
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class CaiBlock:
    name: str
    prompt: str
    enrich_global_context: bool      # True = enrich, False = dont-enrich
    prepend_user_prompt: bool = False
    tools: list = field(default_factory=list)
    model: Optional[str] = None
    max_turns: Optional[int] = None
    strict_format: Optional[str] = None
    system_prompt: Optional[str] = None
    force_tools: bool = False


@dataclass
class BlockInstruction:
    block: CaiBlock


@dataclass
class LabelInstruction:
    name: str


@dataclass
class IfGotoInstruction:
    block_name: str
    expected_value: str
    label: str


@dataclass
class GotoInstruction:
    label: str


@dataclass
class ExitInstruction:
    pass


# ─── Parser ──────────────────────────────────────────────────────────────────

_NORMAL = 'normal'
_BLOCK_HEADER = 'block_header'
_BLOCK_PROMPT = 'block_prompt'
_BLOCK_FOOTER = 'block_footer'


def _strip_quotes(s):
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def _parse_flag_line(line, flags, lineno):
    """Parse a single --flag or --flag <value> line into the flags dict."""
    m = re.match(r"^--([a-z][a-z0-9-]*)(?:\s+(.+))?$", line.strip())
    if not m:
        raise SyntaxError(f"harness:{lineno}: invalid flag syntax: {line.strip()!r}")
    key = m.group(1).replace('-', '_')
    val = m.group(2)
    flags[key] = True if val is None else _strip_quotes(val.strip())


def _build_block(flags, prompt_lines, lineno):
    """Validate collected flags and build a CaiBlock."""
    name = flags.get('name')
    if not name or name is True:
        raise SyntaxError(f"harness:{lineno}: block missing --name")
    name = _strip_quotes(str(name))

    has_enrich = 'enrich_global_context' in flags
    has_dont = 'dont_enrich_global_context' in flags
    if not has_enrich and not has_dont:
        raise SyntaxError(
            f"harness:{lineno}: block '{name}' requires either "
            "--enrich-global-context or --dont-enrich-global-context"
        )
    if has_enrich and has_dont:
        raise SyntaxError(
            f"harness:{lineno}: block '{name}' cannot have both "
            "--enrich-global-context and --dont-enrich-global-context"
        )

    # Parse tools: "read, list_files" or "read list_files"
    tools_raw = flags.get('tools', '')
    if isinstance(tools_raw, str) and tools_raw:
        tools = [t.strip() for t in re.split(r'[,\s]+', tools_raw) if t.strip()]
    else:
        tools = []

    max_turns_raw = flags.get('max_turns')
    if max_turns_raw is not None and max_turns_raw is not True:
        try:
            max_turns = int(max_turns_raw)
        except (TypeError, ValueError):
            raise SyntaxError(f"harness:{lineno}: --max-turns must be an integer")
    else:
        max_turns = None

    def _opt_str(key):
        v = flags.get(key)
        return str(v) if v and v is not True else None

    return CaiBlock(
        name=name,
        prompt='\n'.join(prompt_lines).strip(),
        enrich_global_context=has_enrich,
        prepend_user_prompt=bool(flags.get('prepend_user_prompt', False)),
        tools=tools,
        model=_opt_str('model'),
        max_turns=max_turns,
        strict_format=_opt_str('strict_format'),
        system_prompt=_opt_str('system_prompt'),
        force_tools=bool(flags.get('force_tools', False)),
    )


def parse_harness_file(path: str):
    """
    Parse a .harness.cai file.

    Returns:
        instructions: list of Instruction objects (in program order)
        label_map:    dict mapping label name → instruction index
    """
    with open(path) as f:
        lines = f.read().splitlines()

    instructions = []
    label_map = {}

    state = _NORMAL
    current_flags = {}
    current_prompt_lines = []
    block_close_lineno = 0

    for lineno, raw_line in enumerate(lines, 1):
        stripped = raw_line.strip()

        if state == _NORMAL:
            if not stripped or stripped.startswith('#'):
                continue

            if stripped == '---':
                state = _BLOCK_HEADER
                current_flags = {}
                current_prompt_lines = []
                continue

            # Label: bare word followed by colon, e.g. "ok:" or "enrichment:"
            if re.match(r"^\w+:$", stripped):
                lname = stripped[:-1]
                label_map[lname] = len(instructions)
                instructions.append(LabelInstruction(name=lname))
                continue

            # if <block_name> == <value>: goto <label>
            m = re.match(r"^if\s+(\w+)\s*==\s*(\S+):\s*goto\s+(\w+)$", stripped)
            if m:
                instructions.append(IfGotoInstruction(
                    block_name=m.group(1),
                    expected_value=m.group(2),
                    label=m.group(3),
                ))
                continue

            # goto <label>
            m = re.match(r"^goto\s+(\w+)$", stripped)
            if m:
                instructions.append(GotoInstruction(label=m.group(1)))
                continue

            # exit
            if stripped == 'exit':
                instructions.append(ExitInstruction())
                continue

            raise SyntaxError(f"harness:{lineno}: unexpected token: {stripped!r}")

        elif state == _BLOCK_HEADER:
            if stripped == "'''":
                state = _BLOCK_PROMPT
                continue
            if stripped == '---':
                raise SyntaxError(f"harness:{lineno}: block closed before prompt (missing ''')")
            _parse_flag_line(stripped, current_flags, lineno)

        elif state == _BLOCK_PROMPT:
            if stripped == "'''":
                state = _BLOCK_FOOTER
                block_close_lineno = lineno
                continue
            current_prompt_lines.append(raw_line)

        elif state == _BLOCK_FOOTER:
            if not stripped:
                continue
            if stripped == '---':
                block = _build_block(current_flags, current_prompt_lines, block_close_lineno)
                instructions.append(BlockInstruction(block=block))
                state = _NORMAL
                continue
            raise SyntaxError(f"harness:{lineno}: expected '---' to close block, got {stripped!r}")

    if state != _NORMAL:
        raise SyntaxError("harness: unexpected end of file (unclosed block)")

    return instructions, label_map


# ─── Executor ────────────────────────────────────────────────────────────────

def _build_block_args(block, base_args):
    """
    Copy base_args and apply per-block flag overrides.
    Returns (block_args, block_external_mcps).
    """
    block_args = copy.copy(base_args)

    if block.model:
        block_args.model = block.model
    if block.max_turns is not None:
        block_args.max_turns = block.max_turns
    if block.strict_format:
        block_args.strict_format = block.strict_format
    if block.system_prompt:
        block_args.system_prompt = block.system_prompt
    block_args.force_tools = block.force_tools

    # Split tools into internal names vs external MCP paths
    block_args.selected_tools = set()
    block_external_mcps = {}
    for tool_entry in block.tools:
        if os.path.isfile(tool_entry) or tool_entry.endswith('.py'):
            from cai.cli import get_external_tools
            block_external_mcps[tool_entry] = get_external_tools(tool_entry)
        else:
            block_args.selected_tools.add(tool_entry)

    return block_args, block_external_mcps


def run_block(block, global_messages, user_prompt, base_args, available_tools):
    """
    Execute a single harness block.

    Calls call_llm directly with a local copy of global_messages extended by
    the block's prompt. Mutates global_messages in-place if enrich_global_context
    is True (appending ALL new messages: user prompt, tool calls, results,
    intermediate turns, and final assistant response).

    Returns the block's final text output (stripped).
    """
    from cai.cli import call_llm, MaxTurnsReached

    # Build the prompt text for this block
    prompt = block.prompt
    if block.prepend_user_prompt and user_prompt:
        prompt = (
            f"The user tasked you with: <user_prompt>{user_prompt}</user_prompt>\n\n"
            f"{prompt}"
        )

    # Build local messages: optional system message + global context snapshot + this prompt.
    # The system message is NOT enriched into global context (it's block-local).
    local_messages = []
    if block.system_prompt:
        local_messages.append({"role": "system", "content": block.system_prompt})

    local_messages.extend(global_messages)

    # Everything from here on is "new" — user prompt + whatever call_llm adds.
    global_end = len(local_messages)

    local_messages.append({"role": "user", "content": prompt})

    block_args, block_external_mcps = _build_block_args(block, base_args)

    prefix = f"[{block.name}]"

    def _tool_cb(chunk, error=False):
        sys.stderr.write(chunk)
        sys.stderr.flush()

    def _status_cb(text):
        if text:
            sys.stderr.write(f"{prefix}[{text}]\n")
            sys.stderr.flush()

    def _ctx_cb(ctx_str):
        sys.stderr.write(f"{prefix}[{ctx_str}]\n")
        sys.stderr.flush()

    # Use streaming for non-strict-format blocks so output appears live.
    # Strict-format blocks must use the non-streaming path so enforcement works.
    use_streaming = not block.strict_format
    stream_cb = (lambda chunk: (sys.stdout.write(chunk), sys.stdout.flush())) if use_streaming else None

    _status_cb(f"running")
    try:
        content = call_llm(
            local_messages,
            block_args,
            available_tools,
            block_external_mcps,
            stream_callback=stream_cb,
            tool_callback=_tool_cb,
            status_callback=_status_cb,
            ctx_callback=_ctx_cb,
        )
        if use_streaming:
            print()  # newline after streamed output
    except MaxTurnsReached as e:
        content = ""
        sys.stderr.write(f"{prefix}[!] reached max turns ({e.max_turns})\n")
        sys.stderr.flush()

    # Enrich: extend global context with ALL messages added during this block
    # (user prompt, tool calls, tool results, assistant turns, final response).
    if block.enrich_global_context:
        global_messages.extend(local_messages[global_end:])

    return content.strip()


def execute_harness(instructions, label_map, user_prompt, base_args, available_tools):
    """
    Execute a parsed harness program.

    :param instructions:   list of Instruction objects from parse_harness_file()
    :param label_map:      dict mapping label name → instruction index
    :param user_prompt:    the user's task string (from cai -- <prompt>)
    :param base_args:      argparse Namespace from cli.py main()
    :param available_tools: internal MCP tool definitions (module-level global from cli.py)
    :returns:              the last executed block's output, or ""
    """
    global_messages = []
    block_results = {}   # block name → last text output
    pc = 0

    while pc < len(instructions):
        instr = instructions[pc]

        if isinstance(instr, BlockInstruction):
            output = run_block(
                instr.block,
                global_messages,
                user_prompt,
                base_args,
                available_tools,
            )
            block_results[instr.block.name] = output
            pc += 1

        elif isinstance(instr, IfGotoInstruction):
            actual = block_results.get(instr.block_name, "")
            if actual == instr.expected_value:
                target = label_map.get(instr.label)
                if target is None:
                    raise RuntimeError(f"harness: undefined label '{instr.label}'")
                pc = target
            else:
                pc += 1

        elif isinstance(instr, GotoInstruction):
            target = label_map.get(instr.label)
            if target is None:
                raise RuntimeError(f"harness: undefined label '{instr.label}'")
            pc = target

        elif isinstance(instr, LabelInstruction):
            pc += 1  # no-op; just a jump target

        elif isinstance(instr, ExitInstruction):
            break

        else:
            pc += 1

    # Emit the last block's output to stdout
    last_output = block_results.get(next(reversed(block_results), None), "") if block_results else ""
    if last_output:
        print(last_output)
    return last_output
