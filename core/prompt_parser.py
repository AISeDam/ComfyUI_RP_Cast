"""
Regional Prompter - Prompt Parser (2D support)

1D (ADDCOL only):
  [common] ADDCOMM [base] ADDBASE [col0] ADDCOL [col1] ADDCOL [col2]
  → subprompts = [base, col0, col1, col2]

2D (ADDROW + ADDCOL combined):
  Columns mode:
    ADDROW = row separator  (top→bottom)
    ADDCOL = col separator  (left→right)
  structure: row0_col0 ADDCOL row0_col1 ADDROW row1_col0 ADDCOL row1_col1
  → subprompts order: left→right, top→bottom (row0_col0, row0_col1, row1_col0, row1_col1)
  → is_2d=True, rows_structure: [[n_cols per row]] returned

common is not generated as standalone chunk (SD-WebUI original behavior).
col_lora_map key = 1:1 with subprompts array index.
"""
import re
from typing import List, Dict, Tuple, Optional

KEYCOMM = "ADDCOMM"
KEYBASE = "ADDBASE"
KEYCOL  = "ADDCOL"
KEYROW  = "ADDROW"

_RE_LORA = re.compile(r"<lora:[^>]+>", re.IGNORECASE)


def _strip_lora(text: str) -> str:
    return _RE_LORA.sub("", text).strip()


def _parse_lora_tags(text: str) -> Dict[str, float]:
    result = {}
    for m in re.finditer(r"<lora:([^:> ]+)(?::([0-9.]+))?>", text, re.IGNORECASE):
        name   = m.group(1).strip()
        weight = float(m.group(2)) if m.group(2) else 1.0
        if name:
            result[name] = weight
    return result


def _clean(text: str) -> str:
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().strip(",").strip()


def _join(a: str, b: str) -> str:
    a = a.strip(); b = b.strip()
    if a and b:
        return f"{a}, {b}"
    return a or b


def is_2d_prompt(prompt: str) -> bool:
    """Returns True if both ADDCOL and ADDROW are present (2D mode)."""
    return KEYCOL in prompt and KEYROW in prompt


def parse_prompt(
    prompt: str,
    usebase: bool = False,
    usecom:  bool = True,
) -> Tuple[List[str], List[str], Dict[int, Dict[str, float]], str, str, List[str]]:
    """
    Parse ADDCOMM/ADDBASE/ADDCOL/ADDROW syntax.
    Both ADDCOL + ADDROW present → 2D mode.

    Returns
    -------
    subprompts_raw    : raw chunks incl. LoRA (left→right, top→bottom)
    subprompts_nolora : chunks with LoRA removed
    col_lora_map      : {chunk_index: {lora_name: weight}}
    and_prompt        : AND-joined string
    common_text       : common text (for Prompt-EX)
    col_texts         : col-only parts (for Prompt-EX)
    """
    # ── 1. Split ADDCOMM ──────────────────────────────
    if KEYCOMM in prompt:
        before_comm, after_comm = prompt.split(KEYCOMM, 1)
        _common = _clean(before_comm)
    else:
        _common = ""
        after_comm = prompt

    # ── 2. Split ADDBASE ──────────────────────────────
    if KEYBASE in after_comm:
        before_base, after_cols = after_comm.split(KEYBASE, 1)
        _base_part = _clean(before_base)
    elif usebase:
        # No ADDBASE but use_base=True → first segment before ADDCOL/ADDROW as base
        for kw in (KEYCOL, KEYROW):
            if kw in after_comm:
                before_base, after_cols = after_comm.split(kw, 1)
                _base_part = _clean(before_base)
                after_cols = kw + after_cols  # restore separator
                break
        else:
            _base_part = _clean(after_comm)
            after_cols = ""
    else:
        _base_part = ""
        after_cols = after_comm

    # ── 3. Branch: 2D vs 1D ──────────────────────────
    _two_d = is_2d_prompt(after_cols) or (KEYCOL in after_cols and KEYROW in after_cols)

    if _two_d:
        # ── 2D: split rows by ADDROW, split cols by ADDCOL per row ──
        # Columns mode: ADDROW → row, ADDCOL → col
        row_parts = after_cols.split(KEYROW)
        _col_parts = []
        _row_structure = []  # [[row0_col_count, ...], [row1_col_count, ...]]
        for row_text in row_parts:
            cols_in_row = [_clean(c) for c in row_text.split(KEYCOL)]
            cols_in_row = [c for c in cols_in_row if c]
            if cols_in_row:
                _col_parts.extend(cols_in_row)
                _row_structure.append(len(cols_in_row))
            elif _row_structure:
                _row_structure[-1] = _row_structure[-1]  # ignore empty row
    else:
        # ── 1D: treat ADDCOL/ADDROW identically ────────
        for kw in (KEYCOL, KEYROW):
            after_cols = after_cols.replace(kw, "\x00")
        _col_parts = [_clean(c) for c in after_cols.split("\x00") if _clean(c)]
        _row_structure = [len(_col_parts)] if _col_parts else []

    # ── 4. Build chunks ──────────────────────────────────
    prefix = _common if usecom else ""

    subprompts_raw: List[str] = []

    if usebase:
        base_chunk = _join(prefix, _base_part) if _base_part else prefix
        if base_chunk:
            subprompts_raw.append(base_chunk)

    for col in _col_parts:
        chunk = _join(prefix, col) if col else prefix
        subprompts_raw.append(chunk)

    if not subprompts_raw:
        subprompts_raw = [_clean(prompt)]

    # ── 5. LoRA-removed version ────────────────────────
    subprompts_nolora = [_strip_lora(p) for p in subprompts_raw]

    # ── 6. col_lora_map ──────────────────────────
    col_lora_map: Dict[int, Dict[str, float]] = {}
    for i, raw in enumerate(subprompts_raw):
        loras = _parse_lora_tags(raw)
        if loras:
            col_lora_map[i] = loras

    and_prompt = " AND ".join(subprompts_nolora)

    # For Prompt-EX
    # common_text: global/scene text before ADDCOMM (controlled by use_common)
    common_text = _strip_lora(_clean(prompt.split(KEYCOMM)[0])) if KEYCOMM in prompt else ""

    # col_texts:
    #   usebase=True → [0]=BASE col_only(_base_part), [1..]=DIV col_only(_col_parts)
    #   usebase=False → [0..]=DIV col_only(_col_parts)
    # _base_part = between ADDCOMM and ADDBASE (controlled by use_base)
    # _col_parts[0] = between ADDBASE and first ADDCOL = DIV[0,0]
    # _col_parts[1] = after first ADDCOL = DIV[0,1]
    col_texts_raw = []
    if usebase and _base_part:
        col_texts_raw.append(_base_part)      # BASE prompt (quality tags etc.)
    col_texts_raw.extend(_col_parts)          # DIV[0,0], DIV[0,1], ...
    col_texts = [_strip_lora(t) for t in col_texts_raw]

    return subprompts_raw, subprompts_nolora, col_lora_map, and_prompt, common_text, col_texts


def get_2d_structure(prompt: str, usebase: bool = False) -> Optional[List[int]]:
    """
    Return col count per row for 2D prompt.
    Returns None for 1D.
    Example: "A ADDCOL B ADDROW C ADDCOL D" → [2, 2]
    """
    if KEYCOMM in prompt:
        _, prompt = prompt.split(KEYCOMM, 1)
    if KEYBASE in prompt:
        _, prompt = prompt.split(KEYBASE, 1)
    elif usebase:
        for kw in (KEYCOL, KEYROW):
            if kw in prompt:
                _, rest = prompt.split(kw, 1)
                prompt = kw + rest
                break

    if not (KEYCOL in prompt and KEYROW in prompt):
        return None

    structure = []
    for row_text in prompt.split(KEYROW):
        cols = [c for c in row_text.split(KEYCOL) if _clean(c)]
        if cols:
            structure.append(len(cols))
    return structure if structure else None
