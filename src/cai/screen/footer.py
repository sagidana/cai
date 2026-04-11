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


def _diversify_overlay(items: list, max_visible: int) -> list:
    """Pick a diverse subset that reveals the user's branching options.

    Instead of showing the first N alphabetical matches (which often share a
    long prefix, e.g. six ``anthropic/claude-…`` variants), find the common
    prefix, group by the next distinct segment, and round-robin one
    representative from each group.
    """
    if len(items) <= max_visible:
        return items

    # Common prefix across all matches
    prefix = items[0]
    for item in items[1:]:
        while not item.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                break

    # Group by next segment after the common prefix.
    # A "segment" runs up to (and including) the first separator.
    _SEP = set('/-_:. ')
    groups: dict[str, list] = {}
    for item in items:
        rest = item[len(prefix):]
        key_end = 0
        for i, ch in enumerate(rest):
            key_end = i + 1
            if ch in _SEP:
                break
        key = rest[:key_end] if rest else ''
        groups.setdefault(key, []).append(item)

    # Round-robin through groups so each branch gets fair representation.
    result: list[str] = []
    group_lists = list(groups.values())
    idx = [0] * len(group_lists)
    while len(result) < max_visible:
        added = False
        for g, gl in enumerate(group_lists):
            if len(result) >= max_visible:
                break
            if idx[g] < len(gl):
                result.append(gl[idx[g]])
                idx[g] += 1
                added = True
        if not added:
            break

    result.sort()
    return result


def _window_overlay(overlay_matches: list, cmd_overlay_idx: int, max_visible: int) -> tuple:
    """Return (visible_items, adjusted_selection_index) for a windowed overlay.

    When the overlay has more items than can fit on screen, show a window
    centred on the selected item.  Returns the slice of items to display and
    the index of the selected item within that slice (-1 if none selected).
    """
    total = len(overlay_matches)
    if total <= max_visible:
        return overlay_matches, cmd_overlay_idx

    if cmd_overlay_idx < 0:
        # No selection — show a diverse sample so the user sees their options.
        return _diversify_overlay(overlay_matches, max_visible), -1

    # Active selection — show a sequential window centred on it.
    half = max_visible // 2
    start = cmd_overlay_idx - half
    start = max(0, min(start, total - max_visible))

    end = start + max_visible
    visible = overlay_matches[start:end]
    adj_idx = cmd_overlay_idx - start
    return visible, adj_idx


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

    # Cap overlay to fit within the terminal.  The footer occupies
    # overlay_rows + 1 (status) + prompt_vrows rows below the anchor,
    # and total_needed must stay strictly below `rows` so the anchor
    # never lands at row 0 (which doesn't exist) and drawing the footer
    # doesn't cause untracked scrolls.
    max_overlay = min(6, max(0, rows - 2 - total_prompt_vrows))
    visible_overlay, vis_sel_idx = _window_overlay(
        overlay_matches, cmd_overlay_idx, max_overlay,
    )

    overlay_rows       = len(visible_overlay)
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
    for i, name in enumerate(visible_overlay):
        style  = SGR_BOLD_AZURE if i == vis_sel_idx else SGR_DIM_GRAY
        marker = '▶' if i == vis_sel_idx else ' '
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
    total_prompt_vrows = sum(line_vrows)
    max_overlay = min(6, max(0, rows - 2 - total_prompt_vrows))
    visible_overlay, vis_sel_idx = _window_overlay(
        overlay_matches, cmd_overlay_idx, max_overlay,
    )
    total_needed = len(visible_overlay) + 1 + total_prompt_vrows

    # Anchor row: the footer occupies `total_needed` rows below the anchor,
    # so anchor lands at rows - total_needed (1-indexed).
    anchor_row = max(1, rows - total_needed)
    sys.stdout.write(f'\033[{anchor_row};1H\r{ERASE_TO_END}')
    sys.stdout.write(CUR_SAVE)

    # Pass footer_rows_reserved=total_needed so redraw_footer skips re-expansion
    # (space is already correct) and goes straight to drawing.
    # Pass the already-windowed overlay so redraw_footer doesn't re-cap.
    return redraw_footer(
        input_buf=input_buf,
        cursor_pos=cursor_pos,
        rows=rows,
        cols=cols,
        status_text=status_text,
        cmd_overlay_idx=vis_sel_idx,
        overlay_matches=visible_overlay,
        footer_rows_reserved=total_needed,
        scroll_debt=0,
        prompt_prefix=prompt_prefix,
        cont_prefix=cont_prefix,
        status_style=status_style,
    )
