"""Interactive model picker overlay (alternate screen, fuzzy finder)."""

import select
import shutil
import signal
import sys
import termios
import tty

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_SHOW, CUR_HIDE,
    ERASE_SCREEN,
    SGR_RESET, SGR_REVERSE, SGR_DIM_GRAY, SGR_BOLD_AZURE,
    ansi_pad,
    cur_move,
    KEY_BACKSPACE, KEY_ESC, KEY_ENTER, KEY_CTRL_C,
    KEY_CTRL_D, KEY_CTRL_U, KEY_TAB, KEY_UP, KEY_DOWN,
)
from ..input import read_key, parse_mouse
from .tools import _find_matches, _handle_search_key, _style_cell
from ...favorites import load_favorites, save_favorites


def _overlay_geom(rows, cols, n, selected_idx):
    """list-box geometry shared by the renderer and click hit-testing.
    returns (start_r, start_c, box_w, visible_n, scroll)."""
    inner_w = max(20, int(cols * 0.95) - 2)
    box_w = inner_w + 2
    overhead = 4   # top border + separator + search line + bottom border
    max_box_h = max(overhead + 1, int(rows * 0.95))
    visible_n = max_box_h - overhead
    box_h = visible_n + overhead
    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)
    scroll = 0
    if selected_idx >= visible_n:
        scroll = selected_idx - visible_n + 1
    scroll = max(0, min(scroll, max(0, n - visible_n)))
    return start_r, start_c, box_w, visible_n, scroll


def overlay_click_index(rows, cols, n, selected_idx, click_row, click_col):
    """list index under a click inside the overlay box, or None. the first
    list row sits at start_r + 1 and the list runs for visible_n rows."""
    start_r, start_c, box_w, visible_n, scroll = _overlay_geom(rows, cols, n, selected_idx)
    if not (start_c <= click_col <= start_c + box_w - 1):
        return None
    i = click_row - (start_r + 1)
    if not (0 <= i < visible_n):
        return None
    idx = i + scroll
    if idx >= n:
        return None
    return idx


def _fuzzy_match(pattern, text):
    """simple fuzzy match: every character of pattern must appear in text
    in order. returns (matched, score, match_positions). lower score is
    better - it penalises gaps between matched characters."""
    if not pattern:
        return True, 0, []

    pat = pattern.lower()
    txt = text.lower()
    positions = []
    ti = 0
    for ch in pat:
        found = txt.find(ch, ti)
        if found == -1:
            return False, 0, []
        positions.append(found)
        ti = found + 1

    # score: sum of gaps between consecutive matches + start offset
    score = positions[0]
    for i in range(1, len(positions)):
        score += positions[i] - positions[i - 1] - 1
    return True, score, positions


def _filter_and_sort(models, pattern, favorites=None):
    """models matching pattern, sorted by fuzzy score. each element is
    (model_name, match_positions). when favorites is given, matching favorites
    are grouped first, a (None, None) divider sentinel separates the groups,
    then the rest follow - so the picker shows a favorites section on top."""
    if not pattern:
        matched = []
        for m in models:
            matched.append((m, []))
    else:
        results = []
        for m in models:
            ok, score, positions = _fuzzy_match(pattern, m)
            if ok:
                results.append((score, m, positions))
        results.sort(key=lambda t: t[0])
        matched = []
        for _, m, positions in results:
            matched.append((m, positions))

    if not favorites:
        return matched

    favs = []
    rest = []
    for item in matched:
        if item[0] in favorites:
            favs.append(item)
        else:
            rest.append(item)
    if not favs:
        return rest
    if not rest:
        return favs
    return favs + [(None, None)] + rest


def _is_divider(filtered, idx):
    """True when the row at idx is the favorites/rest divider sentinel."""
    if not (0 <= idx < len(filtered)):
        return False
    return filtered[idx][0] is None


def _fmt_price_num(value):
    """compact USD number: up to 2 decimals with trailing zeros stripped."""
    text = f'{value:.2f}'
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text


def _price_label(price):
    """short '$in/$out' per-1M-token label for a model's price record (with
    'price_in'/'price_out', either possibly missing). 'free' when both are
    known and zero, '' when no price is known."""
    if not price:
        return ''
    pin = price.get('price_in')
    pout = price.get('price_out')
    if pin is None and pout is None:
        return ''
    if pin == 0 and pout == 0:
        return 'free'
    in_s = '?'
    out_s = '?'
    if pin is not None:
        in_s = _fmt_price_num(pin)
    if pout is not None:
        out_s = _fmt_price_num(pout)
    return f'${in_s}/${out_s}'


def _highlight_name(name, positions, inner_w, is_selected, price_label='', is_favorite=False):
    """render a model row: a favorite star (or two-space) prefix, the name with
    fuzzy-matched characters highlighted, and a right-aligned price label. the
    price is dimmed on non-selected rows; on the selected row it rides the
    reverse-video bar so it stays legible."""
    prefix = '  '
    if is_favorite:
        prefix = '★ '

    price_w = 0
    if price_label:
        price_w = len(price_label) + 1  # leading space separator

    name_w = max(1, inner_w - len(prefix) - price_w)
    display = name[:name_w]

    pos_set = set(positions or [])
    parts = []
    for i, ch in enumerate(display):
        if i not in pos_set:
            parts.append(ch)
        elif is_selected:
            parts.append(f'{SGR_RESET}{SGR_BOLD_AZURE}{ch}{SGR_RESET}{SGR_REVERSE}')
        else:
            parts.append(f'{SGR_BOLD_AZURE}{ch}{SGR_RESET}')
    name_part = ''.join(parts)
    name_pad = ' ' * max(0, name_w - len(display))

    price_part = ''
    if price_label and is_selected:
        price_part = f' {price_label}'
    elif price_label:
        price_part = f' {SGR_DIM_GRAY}{price_label}{SGR_RESET}'

    body = f'{prefix}{name_part}{name_pad}{price_part}'
    if is_selected:
        return f'{SGR_REVERSE}{body}{SGR_RESET}'
    return body


def _put_navigate_status(put,
                         visible_n,
                         inner_w,
                         VL,
                         n,
                         noun,
                         search_mode,
                         search_buf,
                         search_direction,
                         search_matches,
                         search_match_idx,
                         search_pattern,
                         hints):
    """status line for the navigate-mode (tools-style) picker."""
    dir_char = '/'
    if search_direction == -1:
        dir_char = '?'

    if search_mode:
        search_text = ''.join(search_buf)
        info = ''
        if search_matches:
            info = f' [{search_match_idx + 1}/{len(search_matches)}]'
        elif search_text:
            info = ' [no match]'
        raw_status = f' {dir_char}{search_text}{info}'
        status_cell = raw_status[:inner_w].ljust(inner_w)
        put(1 + visible_n + 1, f'{VL}{SGR_REVERSE}{status_cell}{SGR_RESET}{VL}')
        return

    left = f' {n} {noun}'
    if search_pattern:
        mlabel = ''
        if search_matches:
            mlabel = f' [{search_match_idx + 1}/{len(search_matches)}]'
        left = f'{left}   {dir_char}{search_pattern}{mlabel}'
    if len(left) + len(hints) <= inner_w:
        left = f'{left}{hints}'
    status_cell = left[:inner_w].ljust(inner_w)
    put(1 + visible_n + 1, f'{VL}{status_cell}{VL}')


def _draw_model_overlay(rows,
                        cols,
                        filtered,
                        selected_idx,
                        search_buf,
                        total_count,
                        prev_lines,
                        first_draw,
                        noun="models",
                        preview_fn=None,
                        navigate=False,
                        search_mode=False,
                        search_matches=None,
                        search_match_idx=-1,
                        search_direction=1,
                        search_pattern='',
                        hints='  j/k /:search ↵:open ESC:cancel',
                        row_colors=None,
                        prices=None,
                        favorites=None):
    """render the model picker overlay. row_colors, when given, is a list
    parallel to filtered: an SGR color per row applied to non-selected,
    non-match rows (used by the agents view to tint by relationship). prices,
    when given, maps model id -> price record for the right-aligned label;
    favorites, when given, is the set of favorited ids drawn with a star."""
    n = len(filtered)

    inner_w = max(20, int(cols * 0.95) - 2)
    start_r, start_c, box_w, visible_n, scroll = _overlay_geom(rows, cols, n, selected_idx)

    # optional preview pane (fzf-style): list left, preview right. only
    # when the box is wide enough to be useful.
    has_preview = preview_fn is not None and inner_w >= 60
    if has_preview:
        list_w = max(28, inner_w // 2)
        prev_w = inner_w - list_w - 1
    else:
        list_w = inner_w
        prev_w = 0

    H = '─'
    TL = '┌'
    TR = '┐'
    BL = '└'
    BR = '┘'
    VL = '│'
    ML = '├'
    MR = '┤'
    h_line = H * inner_w

    # preview lines for the selected item (one ANSI-colored line per row).
    preview_rows = []
    if has_preview and filtered and 0 <= selected_idx < n:
        preview_rows = preview_fn(filtered[selected_idx][0],
                                  prev_w - 1, visible_n)

    new_lines = {}

    def put(row_off, text):
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    # top border
    if has_preview:
        put(0, f'{TL}{H * list_w}┬{H * prev_w}{TR}')
    else:
        put(0, f'{TL}{h_line}{TR}')

    # model list (+ preview pane)
    for i in range(visible_n):
        ai = i + scroll
        if ai >= n:
            cell = ' ' * list_w
        else:
            model_name, positions = filtered[ai]
            is_sel = (ai == selected_idx)
            if model_name is None:
                # favorites/rest divider sentinel: a dim rule, never selectable.
                cell = f'{SGR_DIM_GRAY}{H * list_w}{SGR_RESET}'
            elif navigate:
                display = model_name[:list_w - 2]
                raw_cell = f'  {display}'.ljust(list_w)
                is_match = bool(search_matches) and (ai in search_matches)
                color = None
                if row_colors is not None and 0 <= ai < len(row_colors):
                    color = row_colors[ai]
                cell = _style_cell(raw_cell, is_sel, is_match, color)
            else:
                price_label = ''
                if prices:
                    price_label = _price_label(prices.get(model_name))
                is_fav = bool(favorites) and model_name in favorites
                cell = _highlight_name(model_name, positions, list_w, is_sel,
                                       price_label, is_fav)
        if has_preview:
            preview_line = ''
            if i < len(preview_rows):
                preview_line = preview_rows[i]
            put(1 + i, f'{VL}{cell}{VL}{ansi_pad(" " + preview_line, prev_w)}{VL}')
        else:
            put(1 + i, f'{VL}{cell}{VL}')

    # separator
    if has_preview:
        put(1 + visible_n, f'{ML}{H * list_w}┴{H * prev_w}{MR}')
    else:
        put(1 + visible_n, f'{ML}{h_line}{MR}')

    # search / status line
    if navigate:
        _put_navigate_status(put,
                             visible_n,
                             inner_w,
                             VL,
                             n,
                             noun,
                             search_mode,
                             search_buf,
                             search_direction,
                             search_matches,
                             search_match_idx,
                             search_pattern,
                             hints)
    else:
        n_real = n
        for item in filtered:
            if item[0] is None:
                n_real -= 1
        search_text = ''.join(search_buf)
        match_info = f'{total_count} {noun}'
        if search_text:
            match_info = f'{n_real}/{total_count}'
        hint = ''
        if favorites is not None:
            hint = 'Tab:fav  '
        prompt_str = f' > {search_text}'
        right_str = f'{hint}{match_info} '
        gap = inner_w - len(prompt_str) - len(right_str)
        if gap < 0:
            # truncate search text if needed
            prompt_str = prompt_str[:inner_w - len(right_str) - 1]
            gap = 0
        status_cell = f'{prompt_str}{" " * gap}{SGR_DIM_GRAY}{right_str}{SGR_RESET}'
        put(1 + visible_n + 1, f'{VL}{status_cell}{VL}')

    # bottom border
    put(1 + visible_n + 2, f'{BL}{h_line}{BR}')

    # diff draw
    out = []
    if first_draw:
        sys.stdout.write(ERASE_SCREEN)
        for row_off, (r, text) in new_lines.items():
            out.append(f'{cur_move(r, start_c)}{text}')
    else:
        for row_off, (r, text) in new_lines.items():
            if prev_lines.get(row_off) != text:
                out.append(f'{cur_move(r, start_c)}{text}')
    prev_lines.clear()
    for row_off, (r, text) in new_lines.items():
        prev_lines[row_off] = text

    # position cursor in search field
    cursor_row = start_r + 1 + visible_n + 1
    if navigate and not search_mode:
        out.append(CUR_HIDE)
    elif navigate:
        search_text = ''.join(search_buf)
        cursor_col = start_c + 3 + len(search_text)  # border + " " + dir char
        out.append(f'{CUR_SHOW}{cur_move(cursor_row, cursor_col)}')
    else:
        cursor_col = start_c + 1 + 3 + len(search_text)  # border + " > " + text
        out.append(f'{CUR_SHOW}{cur_move(cursor_row, cursor_col)}')

    sys.stdout.write(''.join(out))
    sys.stdout.flush()


def prompt_model_overlay(screen, models, *, presorted=False,
                         noun="models", preview_fn=None, navigate=False,
                         prices=None, favorites=False):
    """interactive fuzzy picker. returns the selected item or None on
    cancel. used for the model list (sorted alphabetically) and the session
    list (presorted=True to preserve the caller's newest-first order).
    preview_fn(item, width, max_lines) -> list[str] (optional) renders a
    right-hand preview pane for the selected item; lines may carry ANSI
    colors. results are cached per (item, geometry) for the overlay's
    lifetime. navigate=True drives the list with tools/skills-style modal
    navigation (j/k, gg/G, /search) instead of a live fuzzy filter, so the
    preview only follows the cursor. prices, when given, maps model id -> price
    record for a right-aligned label. favorites=True enables a favorites
    section on top (loaded/persisted client-side) with Tab to toggle."""
    if not models:
        return None

    fav_set = set()
    if favorites:
        fav_set = load_favorites()

    if preview_fn is not None:
        _preview_cache = {}
        _inner_fn = preview_fn

        def preview_fn(item, width, max_lines):
            key = (item, width, max_lines)
            if key not in _preview_cache:
                _preview_cache[key] = _inner_fn(item, width, max_lines)
            return _preview_cache[key]

    if not presorted:
        models = sorted(models)
    search_buf = []
    filtered = _filter_and_sort(models, '', fav_set)
    selected_idx = 0
    prev_lines = {}
    first_draw = [True]
    resize_pending = [False]

    # navigate-mode state (unused in fuzzy mode)
    search_mode = False
    search_direction = 1
    search_pattern = ''
    search_matches = []
    search_match_idx = -1
    pre_search_idx = 0
    prev_key = ''

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        resize_pending[0] = True
        first_draw[0] = True

    def _redraw():
        fav_arg = None
        if favorites:
            fav_arg = fav_set
        _draw_model_overlay(
            screen._rows, screen._cols,
            filtered, selected_idx, search_buf,
            len(models), prev_lines, first_draw[0], noun,
            preview_fn,
            navigate, search_mode, search_matches, search_match_idx,
            search_direction, search_pattern,
            prices=prices, favorites=fav_arg,
        )
        first_draw[0] = False

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()

    result = None
    try:
        signal.signal(signal.SIGWINCH, _on_resize)
        tty.setraw(screen._tty_fd)
        _redraw()

        while True:
            if resize_pending[0]:
                resize_pending[0] = False
                _redraw()

            rlist, _, _ = select.select([screen._tty_fd], [], [], 0.05)
            if not rlist:
                continue
            key = read_key(screen._tty_fd)

            mouse = parse_mouse(key)
            if mouse is not None:
                action, _button, mcol, mrow = mouse
                n = len(filtered)
                if action == 'wheel_up':
                    selected_idx = max(0, selected_idx - 1)
                    if _is_divider(filtered, selected_idx) and selected_idx > 0:
                        selected_idx -= 1
                elif action == 'wheel_down':
                    selected_idx = min(max(0, n - 1), selected_idx + 1)
                    if _is_divider(filtered, selected_idx) and selected_idx < n - 1:
                        selected_idx += 1
                elif action == 'press':
                    idx = overlay_click_index(screen._rows, screen._cols, n,
                                              selected_idx, mrow, mcol)
                    if idx is not None and not _is_divider(filtered, idx):
                        selected_idx = idx
                _redraw()
                continue

            if navigate and search_mode:
                search_mode, selected_idx, search_pattern, search_buf, \
                    search_matches, search_match_idx = _handle_search_key(
                        key, filtered, search_mode, selected_idx,
                        search_pattern, search_buf, search_matches,
                        search_match_idx, pre_search_idx, search_direction,
                    )
                _redraw()
                continue

            if navigate:
                if key == KEY_ESC or key == KEY_CTRL_C:
                    break
                if key in KEY_ENTER:
                    if filtered and 0 <= selected_idx < len(filtered):
                        result = filtered[selected_idx][0]
                    break
                n = len(filtered)
                if key in (KEY_UP, 'k'):
                    selected_idx = max(0, selected_idx - 1)
                elif key in (KEY_DOWN, 'j'):
                    selected_idx = min(n - 1, selected_idx + 1)
                elif key in ('/', '?'):
                    search_mode = True
                    search_direction = 1
                    if key == '?':
                        search_direction = -1
                    search_buf = []
                    search_pattern = ''
                    search_matches = []
                    search_match_idx = -1
                    pre_search_idx = selected_idx
                elif key == 'n' and search_matches:
                    search_match_idx = (search_match_idx + search_direction) % len(search_matches)
                    selected_idx = search_matches[search_match_idx]
                elif key == 'N' and search_matches:
                    search_match_idx = (search_match_idx - search_direction) % len(search_matches)
                    selected_idx = search_matches[search_match_idx]
                elif key == 'G':
                    selected_idx = n - 1
                elif key == 'g' and prev_key == 'g':
                    selected_idx = 0
                elif key == KEY_CTRL_U:
                    overhead = 4
                    vis = max(5, int(screen._rows * 0.95)) - overhead
                    selected_idx = max(0, selected_idx - max(1, vis // 2))
                elif key == KEY_CTRL_D:
                    overhead = 4
                    vis = max(5, int(screen._rows * 0.95)) - overhead
                    selected_idx = min(n - 1, selected_idx + max(1, vis // 2))
                prev_key = key
                _redraw()
                continue

            # cancel - first esc clears a pending search, second exits
            if key == KEY_ESC and search_buf:
                search_buf = []
                filtered = _filter_and_sort(models, '', fav_set)
                selected_idx = 0
                _redraw()
                continue

            if key == KEY_ESC or key == KEY_CTRL_C:
                break

            # confirm selection
            if key in KEY_ENTER:
                if filtered and 0 <= selected_idx < len(filtered):
                    name = filtered[selected_idx][0]
                    if name is not None:
                        result = name
                break

            # toggle the selected model in/out of favorites
            if favorites and key == KEY_TAB:
                if filtered and 0 <= selected_idx < len(filtered):
                    name = filtered[selected_idx][0]
                    if name is not None:
                        if name in fav_set:
                            fav_set.discard(name)
                        else:
                            fav_set.add(name)
                        save_favorites(fav_set)
                        filtered = _filter_and_sort(models, ''.join(search_buf), fav_set)
                        selected_idx = 0
                        for i, item in enumerate(filtered):
                            if item[0] == name:
                                selected_idx = i
                                break
                _redraw()
                continue

            # navigation
            if key == KEY_UP:
                if selected_idx > 0:
                    selected_idx -= 1
                    if _is_divider(filtered, selected_idx) and selected_idx > 0:
                        selected_idx -= 1
                _redraw()
                continue

            if key == KEY_DOWN:
                if filtered and selected_idx < len(filtered) - 1:
                    selected_idx += 1
                    if _is_divider(filtered, selected_idx) and selected_idx < len(filtered) - 1:
                        selected_idx += 1
                _redraw()
                continue

            if key == KEY_CTRL_U:
                overhead = 4
                vis = max(5, int(screen._rows * 0.95)) - overhead
                selected_idx = max(0, selected_idx - max(1, vis // 2))
                if _is_divider(filtered, selected_idx):
                    selected_idx = max(0, selected_idx - 1)
                _redraw()
                continue

            if key == KEY_CTRL_D:
                overhead = 4
                vis = max(5, int(screen._rows * 0.95)) - overhead
                selected_idx = min(max(0, len(filtered) - 1),
                                   selected_idx + max(1, vis // 2))
                if _is_divider(filtered, selected_idx):
                    selected_idx = min(max(0, len(filtered) - 1), selected_idx + 1)
                _redraw()
                continue

            # backspace
            if key == KEY_BACKSPACE:
                if search_buf:
                    search_buf.pop()
                    filtered = _filter_and_sort(models, ''.join(search_buf), fav_set)
                    selected_idx = min(selected_idx, max(0, len(filtered) - 1))
                    if _is_divider(filtered, selected_idx):
                        selected_idx = max(0, selected_idx - 1)
                _redraw()
                continue

            # printable character - update search
            if len(key) == 1 and ord(key) >= 32:
                search_buf.append(key)
                filtered = _filter_and_sort(models, ''.join(search_buf), fav_set)
                selected_idx = 0
                _redraw()
                continue

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return result
