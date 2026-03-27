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
    no-more-than <number>              # exit if this point has been passed more than <number> times
    compact-if-more-than <percentage>  # compact global context if usage exceeds <percentage>% of window
    for-each <item> in <block>: harness "<path>"   # run sub-harness for each line of block output
    exit
"""

import copy
import logging
import os
import re
import signal
import sys
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("cai.harness")


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


@dataclass
class CompactInstruction:
    threshold: float   # compact when global context exceeds this % of context window (0–100)


@dataclass
class ForEachInstruction:
    source_block: str   # block whose output is split into items (one per line)
    harness_path: str   # sub-harness to run for each item


@dataclass
class NoMoreThanInstruction:
    limit: int


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

            # compact-if-more-than <percentage>
            m = re.match(r"^compact-if-more-than\s+(\d+(?:\.\d+)?)$", stripped)
            if m:
                instructions.append(CompactInstruction(threshold=float(m.group(1))))
                continue

            # no-more-than <number>
            m = re.match(r"^no-more-than\s+(\d+)$", stripped)
            if m:
                instructions.append(NoMoreThanInstruction(limit=int(m.group(1))))
                continue

            # for-each <item> in <block>: harness "<path>"
            m = re.match(r'^for-each\s+\w+\s+in\s+(\w+):\s+harness\s+"([^"]+)"$', stripped)
            if m:
                instructions.append(ForEachInstruction(
                    source_block=m.group(1),
                    harness_path=m.group(2),
                ))
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

    block_count = sum(1 for i in instructions if isinstance(i, BlockInstruction))
    log.info("parse_harness_file: path=%s instructions=%d blocks=%d labels=%s",
             path, len(instructions), block_count, list(label_map.keys()))
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
    # All streamed output goes to stderr; only the final harness result is written to stdout.
    use_streaming = not block.strict_format
    stream_cb = (lambda chunk: (sys.stderr.write(chunk), sys.stderr.flush())) if use_streaming else None

    log.info("run_block: name=%s model=%s tools=%s max_turns=%s enrich=%s "
             "prepend_user_prompt=%s strict_format=%s prompt_len=%d global_messages=%d",
             block.name, block_args.model, block.tools, block_args.max_turns,
             block.enrich_global_context, block.prepend_user_prompt,
             block.strict_format, len(prompt), len(global_messages))

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
            sys.stderr.write('\n')
            sys.stderr.flush()
    except MaxTurnsReached as e:
        content = ""
        sys.stderr.write(f"{prefix}[!] reached max turns ({e.max_turns})\n")
        sys.stderr.flush()
        log.warning("run_block: name=%s reached max_turns=%d", block.name, e.max_turns)

    result = content.strip()
    log.info("run_block: name=%s done result_len=%d result_preview=%r",
             block.name, len(result), result[:120])

    # Enrich: extend global context with ALL messages added during this block
    # (user prompt, tool calls, tool results, assistant turns, final response).
    if block.enrich_global_context:
        new_msgs = len(local_messages) - global_end
        global_messages.extend(local_messages[global_end:])
        log.info("run_block: name=%s enriched global_messages with %d new messages (total=%d)",
                 block.name, new_msgs, len(global_messages))

    return result


def _compact_global_if_needed(global_messages, base_args, threshold_pct):
    """
    Compact global_messages if estimated token usage exceeds threshold_pct% of the context window.

    Uses character count as a token proxy (≈4 chars/token). Delegates to
    cli._compact_messages which summarises the middle turns into a single
    [memory] system message, preserving the first exchange and the last four
    messages verbatim.
    """
    from cai.cli import _compact_messages, get_model_profile

    if not global_messages:
        return

    total_chars = sum(len(str(m.get('content', ''))) for m in global_messages)
    estimated_tokens = total_chars // 4

    profile = get_model_profile(base_args.model)
    context_limit = profile.get('context', 16000)
    threshold = threshold_pct / 100.0

    if estimated_tokens < context_limit * threshold:
        msg = (f"[compact-if-more-than {threshold_pct}%] skipped "
               f"(~{estimated_tokens} tokens, {estimated_tokens/context_limit:.0%} of {context_limit})")
        sys.stderr.write(msg + '\n')
        sys.stderr.flush()
        log.info("compact_global: skipped estimated_tokens=%d context_limit=%d ratio=%.2f threshold=%.0f%%",
                 estimated_tokens, context_limit, estimated_tokens / context_limit, threshold_pct)
        return

    msg = (f"[compact-if-more-than {threshold_pct}%] compacting "
           f"(~{estimated_tokens} tokens, {estimated_tokens/context_limit:.0%} of {context_limit})")
    sys.stderr.write(msg + '\n')
    sys.stderr.flush()
    log.info("compact_global: compacting estimated_tokens=%d context_limit=%d ratio=%.2f threshold=%.0f%% messages=%d",
             estimated_tokens, context_limit, estimated_tokens / context_limit, threshold_pct, len(global_messages))
    _compact_messages(global_messages, base_args.model)
    log.info("compact_global: done messages_after=%d", len(global_messages))


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
    interrupted = False
    no_more_than_counts = {}   # pc → execution count

    log.info("execute_harness: start instructions=%d user_prompt_len=%d model=%s",
             len(instructions), len(user_prompt or ""), base_args.model)

    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        # Raise KeyboardInterrupt so the current call_llm/streaming is interrupted immediately.
        # The except block below will set `interrupted` and break the loop cleanly.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while pc < len(instructions):
            instr = instructions[pc]

            if isinstance(instr, BlockInstruction):
                log.info("execute_harness: pc=%d block=%s", pc, instr.block.name)
                try:
                    output = run_block(
                        instr.block,
                        global_messages,
                        user_prompt,
                        base_args,
                        available_tools,
                    )
                except KeyboardInterrupt:
                    interrupted = True
                    sys.stderr.write(f"\n[harness] interrupted during block '{instr.block.name}' — stopping.\n")
                    sys.stderr.flush()
                    log.warning("execute_harness: interrupted during block=%s pc=%d", instr.block.name, pc)
                    break
                block_results[instr.block.name] = output
                pc += 1

            elif isinstance(instr, IfGotoInstruction):
                actual = block_results.get(instr.block_name, "")
                if actual == instr.expected_value:
                    target = label_map.get(instr.label)
                    if target is None:
                        raise RuntimeError(f"harness: undefined label '{instr.label}'")
                    log.info("execute_harness: pc=%d if %s==%r -> goto %s (pc=%d)",
                             pc, instr.block_name, instr.expected_value, instr.label, target)
                    pc = target
                else:
                    log.info("execute_harness: pc=%d if %s==%r -> no jump (actual=%r)",
                             pc, instr.block_name, instr.expected_value, actual)
                    pc += 1

            elif isinstance(instr, GotoInstruction):
                target = label_map.get(instr.label)
                if target is None:
                    raise RuntimeError(f"harness: undefined label '{instr.label}'")
                log.info("execute_harness: pc=%d goto %s (pc=%d)", pc, instr.label, target)
                pc = target

            elif isinstance(instr, LabelInstruction):
                pc += 1  # no-op; just a jump target

            elif isinstance(instr, ExitInstruction):
                log.info("execute_harness: pc=%d exit", pc)
                break

            elif isinstance(instr, CompactInstruction):
                log.info("execute_harness: pc=%d compact-if-more-than %.0f%%", pc, instr.threshold)
                _compact_global_if_needed(global_messages, base_args, instr.threshold)
                pc += 1

            elif isinstance(instr, NoMoreThanInstruction):
                count = no_more_than_counts.get(pc, 0) + 1
                no_more_than_counts[pc] = count
                log.info("execute_harness: pc=%d no-more-than %d (count=%d)", pc, instr.limit, count)
                if count > instr.limit:
                    sys.stderr.write(
                        f"[harness] no-more-than {instr.limit} exceeded (ran {count} times) — stopping.\n"
                    )
                    sys.stderr.flush()
                    break
                pc += 1

            elif isinstance(instr, ForEachInstruction):
                raw = block_results.get(instr.source_block, "")
                items = [line.strip() for line in raw.splitlines() if line.strip()]
                log.info("execute_harness: pc=%d for-each source=%s items=%d harness=%s",
                         pc, instr.source_block, len(items), instr.harness_path)
                sub_instructions, sub_label_map = parse_harness_file(instr.harness_path)
                results = []
                for item in items:
                    sys.stderr.write(f"[for-each] running: {item!r}\n")
                    sys.stderr.flush()
                    try:
                        sub_result = execute_harness(
                            sub_instructions, sub_label_map,
                            user_prompt=item,
                            base_args=base_args,
                            available_tools=available_tools,
                        )
                    except KeyboardInterrupt:
                        interrupted = True
                        sys.stderr.write(f"\n[harness] interrupted during for-each item {item!r} — stopping.\n")
                        sys.stderr.flush()
                        break
                    results.append((item, sub_result))
                # Inject a single structured message into parent context so subsequent
                # blocks can reference all results without knowing about the for-each mechanism.
                if results:
                    summary = "\n".join(
                        f"─── task: {item}\n    → {result}" for item, result in results
                    )
                    global_messages.append({
                        "role": "assistant",
                        "content": f"[for-each results: {instr.source_block}]\n{summary}",
                    })
                if interrupted:
                    break
                pc += 1

            else:
                pc += 1

    except KeyboardInterrupt:
        interrupted = True
        sys.stderr.write("\n[harness] interrupted — stopping.\n")
        sys.stderr.flush()
        log.warning("execute_harness: interrupted at pc=%d", pc)

    finally:
        signal.signal(signal.SIGINT, original_sigint)

    # Emit the last block's output to stdout
    last_output = block_results.get(next(reversed(block_results), None), "") if block_results else ""
    log.info("execute_harness: done blocks_executed=%d last_block=%s output_len=%d interrupted=%s",
             len(block_results), next(reversed(block_results), None) if block_results else None,
             len(last_output), interrupted)
    if last_output:
        print(last_output)
    return last_output
