"""Footer rendering: status bar + prompt area with scroll-anchor management."""

import sys

from .ansi import (
    CUR_SAVE, CUR_RESTORE, CUR_HOME, CUR_REV_INDEX,
    ERASE_TO_END, ERASE_LINE,
    SGR_BOLD_AZURE, SGR_DIM_GRAY, SGR_RESET,
)


def _count_visual_rows(line: str, prefix_len: int, cols: int) -> int:
    """Visual (terminal-wrapped) row count for one logical prompt line."""
    total = prefix_len + len(line)
    return max(1, (total + cols - 1) // cols)


def _cursor_visual_pos(
    input_buf: list,
    cursor_pos: int,
    prompt_prefix: str,
    cont_prefix: str,
    cols: int,
    line_vrows: list,
) -> tuple:
    """Return (rows_down_from_first_prompt_row, cursor_col) for the current cursor."""
    chars_before  = ''.join(input_buf[:cursor_pos])
    line_idx      = chars_before.count('\n')
    last_nl       = chars_before.rfind('\n')
    cursor_in_line = len(chars_before) - last_nl - 1
    prefix_len    = len(prompt_prefix if line_idx == 0 else cont_prefix)
    visual_col_abs = prefix_len + cursor_in_line
    cursor_vrow   = sum(line_vrows[:line_idx]) + visual_col_abs // cols
    cursor_col    = visual_col_abs % cols
    return cursor_vrow, cursor_col


def redraw_footer(
    *,
    input_buf: list,
    cursor_pos: int,
    rows: int,
    cols: int,
    status_text: str,
    cmd_overlay_idx: int,
    overlay_matches: list,
    footer_rows_reserved: int,
    scroll_debt: int,
    prompt_prefix: str,
    cont_prefix: str,
    status_style: str,
) -> tuple:
    """
    Restore to the saved footer anchor, clear to EOS, and redraw status + prompt.

    Returns (new_footer_rows_reserved, new_scroll_debt).
    Must be called only after CUR_SAVE has been written.
    """
    buf_str = ''.join(input_buf)
    lines   = buf_str.split('\n')
    cols    = max(1, cols)

    line_vrows        = [_count_visual_rows(ln, len(prompt_prefix if i == 0 else cont_prefix), cols)
                         for i, ln in enumerate(lines)]
    total_prompt_vrows = sum(line_vrows)
    overlay_rows       = len(overlay_matches)
    total_needed       = overlay_rows + 1 + total_prompt_vrows

    if total_needed != footer_rows_reserved:
        sys.stdout.write(f'{CUR_RESTORE}\r{ERASE_TO_END}')
        if total_needed > footer_rows_reserved:
            # Expand: push space below anchor (may scroll); track the debt
            diff         = total_needed - footer_rows_reserved
            scroll_debt += diff
            sys.stdout.write(f'\n' * total_needed + f'\033[{total_needed}A')
        else:
            # Shrink: restore scrollback rows, slide anchor down
            shrink  = footer_rows_reserved - total_needed
            restore = min(shrink, scroll_debt)
            scroll_debt -= restore
            if restore > 0:
                # \033M (Reverse Index) at top-left pulls lines back from scrollback
                sys.stdout.write(CUR_HOME + CUR_REV_INDEX * restore)
            sys.stdout.write(f'{CUR_RESTORE}\033[{shrink}B')
        sys.stdout.write(CUR_SAVE)
        footer_rows_reserved = total_needed

    # Return to anchor, clear to end of screen
    sys.stdout.write(f'{CUR_RESTORE}\r{ERASE_TO_END}')

    # Inline slash-command overlay (drawn above the status line)
    for i, name in enumerate(overlay_matches):
        style  = SGR_BOLD_AZURE if i == cmd_overlay_idx else SGR_DIM_GRAY
        marker = '▶' if i == cmd_overlay_idx else ' '
        sys.stdout.write(f'\r\n{style} {marker} /{name}{ERASE_LINE}{SGR_RESET}')

    # Status line
    text = status_text[:cols - 1]
    sys.stdout.write(f'\r\n{status_style}{text}{ERASE_LINE}{SGR_RESET}')

    # Prompt line(s) — multi-line input (Alt-Enter / backslash continuation)
    for i, line in enumerate(lines):
        prefix = prompt_prefix if i == 0 else cont_prefix
        sys.stdout.write(f'\r\n{prefix}{line}')

    # Position cursor within the prompt area
    cursor_vrow, cursor_col = _cursor_visual_pos(
        input_buf, cursor_pos, prompt_prefix, cont_prefix, cols, line_vrows
    )
    rows_down = overlay_rows + 2 + cursor_vrow
    sys.stdout.write(CUR_RESTORE)
    if rows_down > 0:
        sys.stdout.write(f'\033[{rows_down}B')
    sys.stdout.write(f'\r\033[{cursor_col}C' if cursor_col > 0 else '\r')

    return footer_rows_reserved, scroll_debt


def handle_resize(
    *,
    input_buf: list,
    cursor_pos: int,
    rows: int,
    cols: int,
    status_text: str,
    cmd_overlay_idx: int,
    overlay_matches: list,
    scroll_debt: int,
    prompt_prefix: str,
    cont_prefix: str,
    status_style: str,
) -> tuple:
    """Re-anchor footer at the terminal bottom after resize. Returns (reserved, debt).

    After resize the saved cursor (CUR_RESTORE) may be stale — the terminal
    may have reflowed or clamped the saved row.  Instead of restoring to an
    unknown position, we pre-compute exactly how many rows the footer needs
    and use absolute cursor addressing to place the anchor at the bottom of
    the new terminal size, then let redraw_footer draw without re-expanding.
    """
    cols_eff = max(1, cols)
    lines    = ''.join(input_buf).split('\n')
    line_vrows = [
        _count_visual_rows(ln, len(prompt_prefix if i == 0 else cont_prefix), cols_eff)
        for i, ln in enumerate(lines)
    ]
    total_needed = len(overlay_matches) + 1 + sum(line_vrows)

    # Anchor row: the footer occupies `total_needed` rows below the anchor,
    # so anchor lands at rows - total_needed (1-indexed).
    anchor_row = max(1, rows - total_needed)
    sys.stdout.write(f'\033[{anchor_row};1H\r{ERASE_TO_END}')
    sys.stdout.write(CUR_SAVE)

    # Pass footer_rows_reserved=total_needed so redraw_footer skips re-expansion
    # (space is already correct) and goes straight to drawing.
    return redraw_footer(
        input_buf=input_buf,
        cursor_pos=cursor_pos,
        rows=rows,
        cols=cols,
        status_text=status_text,
        cmd_overlay_idx=cmd_overlay_idx,
        overlay_matches=overlay_matches,
        footer_rows_reserved=total_needed,
        scroll_debt=0,
        prompt_prefix=prompt_prefix,
        cont_prefix=cont_prefix,
        status_style=status_style,
    )
