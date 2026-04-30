"""
_rp_txt2img_common - Shared utilities for RP Txt2Img nodes
Common functions used by OpenAI / Gemini / Grok nodes.
"""
from __future__ import annotations
import re, json, urllib.request, urllib.error
from typing import Optional

# ── Keyword constants ───────────────────────────────────────────────────────
_KEYCOMM = "ADDCOMM"
_KEYBASE = "ADDBASE"
_KEYCOL  = "ADDCOL"
_KEYROW  = "ADDROW"

# ── Utility functions ───────────────────────────────────────────────────────
_RE_LORA  = re.compile(r"<lora:[^>]+>", re.IGNORECASE)
_RE_COUNT = re.compile(
    r'\b\d+\s*(?:boys?|girls?|men|women|persons?|peoples?)\b'
    r'|\b(?:a\s+)?(?:couple|pair)\b',
    re.IGNORECASE,
)

def _clean(text: str) -> str:
    text = _RE_LORA.sub("", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().strip(",").strip()

def _strip_lora(text: str) -> str:
    return _RE_LORA.sub("", text).strip()

def _strip_count(text: str) -> str:
    return re.sub(r",?\s*" + _RE_COUNT.pattern, "", text, flags=re.IGNORECASE).strip().strip(",").strip()

def _dbg(msg: str, debug: bool):
    if debug:
        print(f"[RPTxt2Img] {msg}")

# ── Position labels ─────────────────────────────────────────────────────────
_H_LABELS = {1: ["center"], 2: ["left", "right"], 3: ["left", "center", "right"],
              4: ["far-left", "center-left", "center-right", "far-right"]}
_V_LABELS = {1: [""], 2: ["upper", "lower"], 3: ["upper", "middle", "lower"]}

def _h_label(col: int, n_cols: int) -> str:
    """Horizontal position label."""
    row = _H_LABELS.get(n_cols)
    if row: return row[min(col, n_cols - 1)]
    r = col / (n_cols - 1) if n_cols > 1 else 0
    return ["far-left", "center-left", "center-right", "far-right"][min(int(r * 4), 3)]

def _pos_label(row: int, col: int, n_rows: int, n_cols_in_row: int, n_cols_total: int = 0) -> str:
    """
    Cell position label.
    row            : row index
    col            : column index
    n_rows         : total row count
    n_cols_in_row  : column count in this row
    n_cols_total   : max columns across rows (for asymmetric grid center detection)
    """
    _max_cols = n_cols_total if n_cols_total > 0 else n_cols_in_row
    # Horizontal: single col in row but layout has more cols → center
    if n_cols_in_row == 1 and _max_cols > 1:
        h = "center"
    else:
        h = _h_label(col, n_cols_in_row)
    # Vertical: based on total row count
    v_row = _V_LABELS.get(n_rows)
    v = v_row[min(row, n_rows - 1)] if v_row else ("upper" if row == 0 else "lower")
    if n_rows == 1:                              return h   # single row → horizontal only
    if n_cols_in_row == 1 and _max_cols == 1:   return v   # single cell → vertical only
    return f"{v}-{h}"

# ── regional_col_n_row parser ───────────────────────────────────────────────
def _parse_col_n_row(col_n_row: str) -> Optional[tuple[int, int]]:
    """
    Parse "cols x rows" or "cols,rows" format.
    e.g. "3x2" → (3, 2), "2,2" → (2, 2). Returns None if invalid.
    """
    if not col_n_row or not col_n_row.strip():
        return None
    m = re.match(r"^\s*(\d+)\s*[xX,]\s*(\d+)\s*$", col_n_row.strip())
    if m:
        cols, rows = int(m.group(1)), int(m.group(2))
        if cols >= 1 and rows >= 1:
            return (cols, rows)
    return None

# ── RP prompt detection ─────────────────────────────────────────────────────
def _is_rp_prompt(prompt: str) -> bool:
    return _KEYCOMM in prompt

# ── Core conversion function ────────────────────────────────────────────────
def _convert_rp_to_natural(
    prompt: str,
    regional_col_n_row: str = "",
    divide_mode: str = "Horizontal",
    debug: bool = False,
) -> str:
    """
    Convert Regional Prompter syntax to natural language for image generation APIs.

    Parameters
    ----------
    prompt             : RP prompt with ADDCOMM/ADDBASE/ADDCOL/ADDROW syntax
    regional_col_n_row : "cols x rows" format (optional).
                         If provided → use as grid layout.
                         If omitted → auto-detect from ADDROW presence.
    divide_mode        : "Horizontal" | "Vertical". Matches RPRatioParser divide_mode. mode.
    debug              : Print debug output if True
    """
    # 1. Split ADDCOMM
    if _KEYCOMM in prompt:
        common_text, after_comm = prompt.split(_KEYCOMM, 1)
        common_text = _clean(common_text)
    else:
        common_text = ""
        after_comm = prompt

    # 2. Split ADDBASE
    def _first_kw_pos(text: str) -> int:
        positions = [text.find(kw) for kw in (_KEYCOL, _KEYROW) if text.find(kw) >= 0]
        return min(positions) if positions else len(text)

    if _KEYBASE in after_comm:
        base_part, grid_part = after_comm.split(_KEYBASE, 1)
        base_text = _strip_lora(_clean(base_part))
    else:
        base_text = ""
        # Absorb text between ADDCOMM and first ADDCOL/ADDROW as common suffix
        first_kw = _first_kw_pos(after_comm)
        comm_suffix = _clean(after_comm[:first_kw])
        if comm_suffix:
            common_text = ", ".join(filter(None, [common_text, comm_suffix]))
        grid_part = after_comm[first_kw:]

    has_row = _KEYROW in grid_part
    has_col = _KEYCOL in grid_part

    # 3. Parse regional_col_n_row → determine grid
    explicit_grid = _parse_col_n_row(regional_col_n_row)
    _dbg(f"explicit_grid={explicit_grid}, has_row={has_row}, has_col={has_col}", debug)

    # ── Vertical parsing helper ──────────────────────────────────────────
    def _parse_vertical(gpart: str):
        """Vertical: ADDCOL=col axis, ADDROW=row split within col.
        Returns: cells=(row,col,text,n_rows_in_col,n_cols_total), n_cols"""
        col_blocks = gpart.split(_KEYCOL)
        col_blocks = [b for b in col_blocks if _clean(b)]
        n_cols_total = len(col_blocks)
        _cells = []
        for c_idx, block in enumerate(col_blocks):
            rows_in_col = [_strip_lora(_clean(r))
                           for r in block.split(_KEYROW) if _clean(r)]
            n_r = len(rows_in_col)
            for r_idx, text in enumerate(rows_in_col):
                _cells.append((r_idx, c_idx, text, n_r, n_cols_total))
        return _cells, n_cols_total

    def _parse_horizontal(gpart: str):
        """Horizontal: ADDROW=row axis, ADDCOL=col split.
        Returns: cells=(row,col,text,n_rows,n_cols_in_row), n_cols_total"""
        row_blocks = gpart.split(_KEYROW)
        row_blocks = [b for b in row_blocks if _clean(b)] or [gpart]
        n_total_rows = len(row_blocks)
        _cells = []
        n_c = 1
        for r_idx, block in enumerate(row_blocks):
            cols_in_row = [_strip_lora(_clean(c))
                           for c in block.split(_KEYCOL) if _clean(c)]
            n_cols_in_row = len(cols_in_row)
            n_c = max(n_c, n_cols_in_row)
            for c_idx, text in enumerate(cols_in_row):
                _cells.append((r_idx, c_idx, text, n_total_rows, n_cols_in_row))
        return _cells, n_c

    # ── Case A: explicit_grid provided ──────────────────────────────────────
    if explicit_grid:
        n_cols, n_rows = explicit_grid
        # Parse direction from divide_mode (matches RPRatioParser)
        use_vertical = (divide_mode == "Vertical")
        if use_vertical:
            cells, n_cols = _parse_vertical(grid_part)
        else:
            cells, n_cols = _parse_horizontal(grid_part)
        n_rows = max((c[3] for c in cells), default=1) if use_vertical else len(set(c[0] for c in cells))
        _dbg(f"explicit grid {n_rows}x{n_cols}, cells={[(r,c,nr,nc) for r,c,_,nr,nc in cells]}", debug)

    # ── Case B: ADDROW present → use divide_mode ────────────────────────
    elif has_row:
        if divide_mode == "Vertical":
            cells, n_cols = _parse_vertical(grid_part)
        else:
            cells, n_cols = _parse_horizontal(grid_part)
        n_rows = max((c[3] for c in cells), default=1)
        _dbg(f"auto 2D: n_cols={n_cols}  mode={divide_mode}", debug)

    # ── Case C: no ADDROW → 1D Horizontal ──────────────────────────────────
    else:
        col_blocks = grid_part.replace(_KEYROW, _KEYCOL).split(_KEYCOL)
        col_blocks = [_strip_lora(_clean(c)) for c in col_blocks if _clean(c)]
        n_rows = 1
        n_cols = len(col_blocks)
        cells = [(0, col_idx, text, 1, n_cols) for col_idx, text in enumerate(col_blocks)]
        _dbg(f"1D horizontal: {n_cols}cols", debug)

    # 4. Build position label phrases
    phrases = []
    for cell in cells:
        row_idx, col_idx, text = cell[0], cell[1], cell[2]
        n_rows_val    = cell[3]   # Horizontal=total rows / Vertical=rows in col
        n_cols_in_row = cell[4]   # Horizontal=cols in row / Vertical=1
        if not text:
            continue
        lbl = _pos_label(row_idx, col_idx, n_rows_val, n_cols_in_row, n_cols)
        if n_rows_val == 1 and n_cols == 1:
            phrases.append(f"({text})")
        else:
            phrases.append(f"({text}) on the {lbl} side")

    # 5. Join phrases
    if len(phrases) == 0:   sent = ""
    elif len(phrases) == 1: sent = phrases[0]
    elif len(phrases) == 2: sent = f"{phrases[0]} and {phrases[1]}"
    else: sent = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    # 6. Strip person count from common text
    cs = _strip_count(common_text) if common_text else ""

    # 7. Final assembly
    sp = []
    if cs: sp.append(cs)
    if sent:
        sp.append(sent + ", interacting naturally in the same scene, seamless composition")
    scene = ", ".join(sp)

    parts = []
    if scene:     parts.append(scene)
    if base_text: parts.append(base_text)
    result = ", ".join(parts) if parts else _clean(prompt)

    _dbg(f"result:\n{result}", debug)
    return result



import os as _os

def _settings_path() -> str:
    """Return ComfyUI settings.json path."""
    return _os.path.normpath(_os.path.join(
        _os.path.dirname(__file__), "..", "..", "..", "..",
        "user", "default", "comfy.settings.json"))

def _get_setting(key: str) -> str:
    try:
        import json as _json
        with open(_settings_path(), encoding="utf-8") as f:
            return _json.load(f).get(key, "")
    except Exception:
        return ""

def _regions_to_col_n_row(regional_col_n_row, divide_mode: str, debug: bool, tag: str) -> str:
    if regional_col_n_row is None:
        return ""
    try:
        rr = len(regional_col_n_row)
        rc = max(len(r.cols) for r in regional_col_n_row) if rr > 0 else 1
        n_cols, n_rows = (rr, rc) if divide_mode == "Vertical" else (rc, rr)
        if debug:
            print(f"[{tag}] RP_REGIONS -> {n_rows}rows x {n_cols}cols  mode={divide_mode}")
        return f"{n_cols}x{n_rows}"
    except Exception as e:
        print(f"[{tag}] regional_col_n_row parse failed: {e}")
        return ""

def _png_to_tensor(png_bytes: bytes):
    """PNG bytes → (1, H, W, 3) float32 Tensor"""
    import torch, zlib, struct
    # Verify PNG signature
    assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n', "Not a valid PNG"

    # Parse IHDR chunk
    pos = 8
    chunks = {}
    while pos < len(png_bytes):
        length = struct.unpack(">I", png_bytes[pos:pos+4])[0]
        chunk_type = png_bytes[pos+4:pos+8].decode("ascii", errors="ignore")
        chunk_data = png_bytes[pos+8:pos+8+length]
        chunks.setdefault(chunk_type, []).append(chunk_data)
        pos += 12 + length

    ihdr = chunks["IHDR"][0]
    W, H = struct.unpack(">II", ihdr[:8])
    bit_depth, color_type = ihdr[8], ihdr[9]

    idat = b"".join(chunks.get("IDAT", []))
    raw  = zlib.decompress(idat)

    # Determine channel count
    ch_map = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    channels = ch_map.get(color_type, 3)
    stride = W * channels + 1  # +1 for filter byte per row

    pixels = []
    prev_row = bytes(W * channels)
    for r in range(H):
        row_raw = raw[r * stride: (r + 1) * stride]
        ftype   = row_raw[0]
        row     = bytearray(row_raw[1:])
        if ftype == 1:   # Sub
            for i in range(channels, len(row)):
                row[i] = (row[i] + row[i - channels]) & 0xFF
        elif ftype == 2: # Up
            row = bytearray((row[i] + prev_row[i]) & 0xFF for i in range(len(row)))
        elif ftype == 3: # Average
            for i in range(len(row)):
                a = row[i - channels] if i >= channels else 0
                row[i] = (row[i] + (a + prev_row[i]) // 2) & 0xFF
        elif ftype == 4: # Paeth
            def paeth(a, b, c):
                p = a + b - c
                pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
                return a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            for i in range(len(row)):
                a = row[i - channels] if i >= channels else 0
                b = prev_row[i]
                c = prev_row[i - channels] if i >= channels else 0
                row[i] = (row[i] + paeth(a, b, c)) & 0xFF
        pixels.extend(row)
        prev_row = bytes(row)

    t = torch.tensor(pixels, dtype=torch.float32).reshape(H, W, channels) / 255.0
    # Normalize to RGB 3 channels
    if channels == 1:
        t = t.expand(-1, -1, 3)
    elif channels == 4:
        t = t[:, :, :3]
    elif channels == 2:
        t = t[:, :, :1].expand(-1, -1, 3)
    return t.unsqueeze(0)


def _bytes_to_tensor(img_bytes: bytes, mime_type: str = ""):
    """
    Decode image bytes → (1, H, W, 3) float32 Tensor.
    Auto-detects PNG/JPEG/WEBP from byte signature or mimeType.
    """
    import torch

    # Detect format from byte signature (more reliable than mimeType)
    is_png  = img_bytes[:8] == b'\x89PNG\r\n\x1a\n'
    is_jpeg = img_bytes[:2] == b'\xff\xd8'
    is_webp = img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP'

    print(f"[_bytes_to_tensor] mime={mime_type!r}  "
          f"header={img_bytes[:12].hex()}  "
          f"png={is_png} jpeg={is_jpeg} webp={is_webp}")

    if is_png:
        return _png_to_tensor(img_bytes)

    # JPEG / WEBP / other image formats → decode via PIL
    try:
        from PIL import Image
        import io, numpy as np
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)
        print(f"[_bytes_to_tensor] PIL decode OK: {img.size} mode={img.mode}")
        return t
    except ImportError:
        raise RuntimeError(
            "PIL(Pillow) is required: pip install Pillow\n"
            f"mime={mime_type}  header={img_bytes[:8].hex()}")
    except Exception as e:
        raise RuntimeError(
            f"Image decode failed: {e}\n"
            f"mime={mime_type}  header={img_bytes[:12].hex()}")
