"""
Regional Prompter - Region Geometry Core
Refined from SD-WebUI regions.py for ComfyUI.
Removed gradio/modules dependencies; pure Python/Torch only.
"""

import torch

# ──────────────────────────────────────────────
# Keyword constants
# ──────────────────────────────────────────────
KEYROW   = "ADDROW"
KEYCOL   = "ADDCOL"
KEYBASE  = "ADDBASE"
KEYCOMM  = "ADDCOMM"
KEYBRK   = "BREAK"
KEYPROMPT = "ADDP"
DELIMROW = ";"
DELIMCOL = ","

ALLKEYS     = [KEYCOMM, KEYROW, KEYCOL, KEYBASE, KEYPROMPT]
ALLALLKEYS  = [KEYCOMM, KEYROW, KEYCOL, KEYBASE, KEYPROMPT, KEYBRK, "AND"]


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────
class RegionCell:
    """Single region cell (col split unit)."""
    def __init__(self, st: float, ed: float, base: float, breaks: int = 0):
        self.st     = st       # start ratio (0.0 ~ 1.0)
        self.ed     = ed       # end ratio
        self.base   = base     # base prompt mix ratio
        self.breaks = breaks   # unrelated BREAK count in cell

    def __repr__(self):
        return f"({self.st:.2f}:{self.ed:.2f})"


class RegionRow:
    """Row – contains multiple RegionCells."""
    def __init__(self, st: float, ed: float, cols: list):
        self.st   = st
        self.ed   = ed
        self.cols = cols   # List[RegionCell]

    def __repr__(self):
        return f"Row({self.st:.2f}:{self.ed:.2f}) {self.cols}"


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────
def floatdef(x, vdef=1.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return vdef


def _ffloatd(c):
    return lambda x: floatdef(x, c)


def _fcountbrk(x: str) -> int:
    return x.count(KEYBRK)


def _is_l2(l):
    return l and isinstance(l[0], list)


def list_percentify(l):
    """Normalize each row (or list) so values sum to 1.0."""
    if _is_l2(l):
        return [[v / (sum(row) or 1) for v in row] for row in l]
    s = sum(l) or 1
    return [v / s for v in l]


def list_cumsum(l):
    """Cumulative sum (in-place friendly)."""
    if _is_l2(l):
        out = []
        for row in l:
            r = list(row)
            for i in range(1, len(r)):
                r[i] += r[i - 1]
            out.append(r)
        return out
    out = list(l)
    for i in range(1, len(out)):
        out[i] += out[i - 1]
    return out


def list_rangify(l):
    """Convert consecutive value pairs to [start, end] range list."""
    if _is_l2(l):
        return [[[row[i - 1] if i > 0 else 0, row[i]] for i in range(len(row))] for row in l]
    return [[l[i - 1] if i > 0 else 0, l[i]] for i in range(len(l))]


def ratiosdealer(aratios2, aratios2r):
    aratios2  = list_rangify(list_cumsum(list_percentify(aratios2)))
    aratios2r = list_rangify(list_cumsum(list_percentify(aratios2r)))
    return aratios2, aratios2r


def split_l2(s, kr, kc, indsingles=False, fmap=None, basestruct=None, indflip=False):
    """
    Split string into 2D list using kr(row sep), kc(col sep).
    If basestruct given, broadcast to match its structure.
    """
    if fmap is None:
        fmap = lambda x: x

    if indflip:
        kr, kc = kc, kr

    if basestruct is None:
        lrows = s.split(kr)
        lrows = [row.split(kc) for row in lrows]
        lret  = [[fmap(x) for x in row] for row in lrows]
        if indsingles:
            lsingles = [row[0] for row in lret]
            lcells   = [row[1:] if len(row) > 1 else row[:] for row in lret]
            return lsingles, lcells
        return lret
    else:
        lrows     = s.split(kr)
        r         = 0
        lcells    = []
        lsingles  = []
        vlast     = 1.0

        for row in lrows:
            row2   = [fmap(x) for x in row.split(kc)]
            vlast  = row2[-1]
            indstop = False
            while not indstop:
                if r >= len(basestruct) or len(row2) == 0:
                    indstop = True
                if not indstop:
                    if indsingles:
                        lsingles.append(row2[0])
                        if len(row2) > 1:
                            row2 = row2[1:]
                    if len(basestruct[r]) >= len(row2):
                        indstop  = True
                        broadrow = row2 + [row2[-1]] * (len(basestruct[r]) - len(row2))
                        r += 1
                        lcells.append(broadrow)
                    else:
                        broadrow = row2[:len(basestruct[r])]
                        row2     = row2[len(basestruct[r]):]
                        r += 1
                        lcells.append(broadrow)

        cur = len(lcells)
        while cur < len(basestruct):
            lcells.append([vlast] * len(basestruct[cur]))
            cur += 1

        if indsingles:
            lsingles += [lsingles[-1]] * (len(basestruct) - len(lsingles))
            return lsingles, lcells
        return lcells


# ──────────────────────────────────────────────
# P1: aratios string → RegionRow list
# ──────────────────────────────────────────────
def parse_regions(
    aratios: str,
    bratios: str = "0",
    mode: str    = "Horizontal",
    prompt: str  = "",
) -> list:
    """
    Accepts aratios string and mode, returns List[RegionRow].

    Parameters
    ----------
    aratios : str
        Region ratio string. e.g. "1,1" / "1,1,1" / "1,2;1,1" (Matrix)
    bratios : str
        Base mix ratio string.
    mode : str
        "Horizontal" | "Vertical"
    prompt : str
        Prompt containing ADDROW/ADDCOL keywords (for break count).
        If not given, break=0 assumed.

    Returns
    -------
    List[RegionRow]
    """
    indflip = ("Ver" in mode)

    # Calculate break count from prompt
    if KEYCOL in prompt.upper() or KEYROW in prompt.upper():
        lbreaks = split_l2(prompt, KEYROW, KEYCOL, fmap=_fcountbrk, indflip=indflip)
    else:
        # Build with dummy break=0 from aratios structure
        # Always indflip=False: _auto_aratios already produces correct format
        tmp_rows = split_l2(aratios, DELIMROW, DELIMCOL, fmap=_ffloatd(1), indflip=False)
        lbreaks = [[0] * len(row) for row in tmp_rows]

    # Parse aratios - always indflip=False (separator flip handled by _auto_aratios)
    if DELIMROW not in aratios:
        aratios2  = split_l2(aratios, DELIMROW, DELIMCOL, fmap=_ffloatd(1), indflip=False)
        aratios2r = [1.0]
    else:
        aratios2  = split_l2(aratios, DELIMROW, DELIMCOL, fmap=_ffloatd(1), indflip=False)
        aratios2r = [1.0] * len(aratios2)

    # Parse bratios
    bratios2 = split_l2(
        bratios, DELIMROW, DELIMCOL,
        fmap=_ffloatd(0), basestruct=lbreaks, indflip=False,
    )

    aratios_r, aratiosr_r = ratiosdealer(aratios2, aratios2r)

    # Build RegionRow / RegionCell structures
    drows = []
    for r, _ in enumerate(lbreaks):
        dcells = []
        for c, _ in enumerate(lbreaks[r]):
            cell = RegionCell(
                st     = aratios_r[r][c][0],
                ed     = aratios_r[r][c][1],
                base   = bratios2[r][c] if _is_l2(bratios2) else bratios2[c],
                breaks = lbreaks[r][c],
            )
            dcells.append(cell)
        row = RegionRow(aratiosr_r[r][0], aratiosr_r[r][1], dcells)
        drows.append(row)

    return drows


# ──────────────────────────────────────────────
# P1: Build spatial mask Tensor (makefilters)
# ──────────────────────────────────────────────
def make_filters(
    region_rows: list,
    h: int,
    w: int,
    mode: str       = "Horizontal",
    usebase: bool   = False,
    base_ratio: float = 0.2,
    batch: int      = 1,
    device: str     = "cpu",
) -> list:
    """
    RegionRow list → spatial mask Tensor list.

    Returns
    -------
    List[Tensor[1, H, W]]  length = areas (usebase: areas+1)
    Batch repeat handled externally.
    """
    filters = []
    indflip = ("Ver" in mode)

    # BASE mask (full area = 1.0)
    if usebase:
        base_fil = torch.zeros(1, h, w, device=device)
        base_fil[:] = 1.0
        filters.append(base_fil)

    for drow in region_rows:
        row_st = drow.st
        row_ed = drow.ed

        for dcell in drow.cols:
            fil = torch.zeros(1, h, w, device=device)

            if not indflip:
                # Horizontal: row = height, col = width
                r_st = int(h * row_st)
                r_ed = int(h * row_ed)
                c_st = int(w * dcell.st)
                c_ed = int(w * dcell.ed)
                if row_ed >= 0.999:
                    r_ed = h
                if dcell.ed >= 0.999:
                    c_ed = w
                fil[:, r_st:r_ed, c_st:c_ed] = 1.0
            else:
                # Vertical: row = width, col = height
                c_st = int(w * row_st)
                c_ed = int(w * row_ed)
                r_st = int(h * dcell.st)
                r_ed = int(h * dcell.ed)
                if row_ed >= 0.999:
                    c_ed = w
                if dcell.ed >= 0.999:
                    r_ed = h
                fil[:, r_st:r_ed, c_st:c_ed] = 1.0

            filters.append(fil)

    return filters
