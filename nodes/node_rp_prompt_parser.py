import os, sys, re, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

# Import from _shared (already loaded by __init__.py, or load directly)
def _get_shared():
    import importlib.util
    _key = "nodes._shared"
    if _key in sys.modules:
        return sys.modules[_key]
    _spec = importlib.util.spec_from_file_location(
        _key, os.path.join(_HERE, "_shared.py"))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_key] = _mod
    _spec.loader.exec_module(_mod)
    return _mod

_shared = _get_shared()
_RE_LORA              = _shared._RE_LORA
_enc                  = _shared._enc
_SAMPLERS             = _shared._SAMPLERS
_SCHEDULERS           = _shared._SCHEDULERS
_COMFY_OK             = _shared._COMFY_OK
_TORCH_LOAD_ORIG      = _shared._TORCH_LOAD_ORIG
_ZIMAGE_LATENT_CHANNELS = _shared._ZIMAGE_LATENT_CHANNELS
_ZIMAGE_LATENT_BLOCK  = _shared._ZIMAGE_LATENT_BLOCK
_ZIMAGE_GRID_SIZE     = _shared._ZIMAGE_GRID_SIZE
_ZImageAuraFlow       = _shared._ZImageAuraFlow
parse_prompt          = _shared.parse_prompt
get_2d_structure      = _shared.get_2d_structure
parse_regions         = _shared.parse_regions
make_filters          = _shared.make_filters
RPLatentCompositor    = _shared.RPLatentCompositor
LoRADivisionManager   = _shared.LoRADivisionManager
del _shared, _get_shared

def _count_addcol(prompt: str) -> int:
    return (prompt.upper().count("ADDCOL")
            + prompt.upper().count("ADDROW"))


def _auto_aratios(prompt: str, divide_mode: str = "Horizontal") -> str:
    """
    Auto-calculate divide_ratio based on ADDBASE/ADDCOL/ADDROW structure.

    Rules:
      - Parsed from text after ADDCOMM, ADDBASE
      - Segment count = separator count + 1 (regardless of ADDBASE)
      - Columns(Horizontal): ADDROW → rows(;), ADDCOL → cols(,)
      - Rows(Vertical):      ADDCOL → rows(;), ADDROW → cols(,)
      - 2D: when both ADDROW and ADDCOL are present

    Examples:
      ADDBASE + ADDCOL + ADDCOL  →  segments=3  →  "1,1,1"
      ADDCOL + ADDCOL            →  segments=3  →  "1,1,1"
      ADDBASE + ADDROW + ADDROW  →  segments=3  →  "1;1;1"
      2D: A ADDCOL B ADDCOL C ADDROW D ADDCOL E  →  "1,1,1;1,1"
    """
    from core.prompt_parser import KEYCOL, KEYROW, KEYCOMM, KEYBASE
    is_vertical = "Ver" in divide_mode

    # Only ADDCOMM, ADDBASE sections are used (BASE itself is included in segments)
    p = prompt
    if KEYCOMM in p:
        _, p = p.split(KEYCOMM, 1)
    if KEYBASE in p:
        _, p = p.split(KEYBASE, 1)

    has_col = KEYCOL in p
    has_row = KEYROW in p

    if not has_col and not has_row:
        return "1"

    if is_vertical:
        # Vertical(Rows): ADDCOL separates rows(;), ADDROW separates cols(,)
        if has_col and has_row:
            col_segs = p.split(KEYCOL)  # split rows by ADDCOL
            ratios = []
            for seg in col_segs:
                n = seg.count(KEYROW) + 1  # col count per row
                ratios.append(",".join(["1"] * n))
            return ";".join(ratios)
        elif has_col:
            n = p.count(KEYCOL) + 1
            return ";".join(["1"] * n)
        else:
            n = p.count(KEYROW) + 1
            return ",".join(["1"] * n)
    else:
        # Horizontal(Cols): ADDROW separates rows(;), ADDCOL separates cols(,)
        if has_col and has_row:
            row_segs = p.split(KEYROW)  # split rows by ADDROW
            ratios = []
            for seg in row_segs:
                n = seg.count(KEYCOL) + 1  # col count per row
                ratios.append(",".join(["1"] * n))
            return ";".join(ratios)
        elif has_row:
            n = p.count(KEYROW) + 1
            return ";".join(["1"] * n)
        else:
            n = p.count(KEYCOL) + 1
            return ",".join(["1"] * n)


# ══════════════════════════════════════════════════════
# 1. RPPromptParser
#    - use_base / use_common toggle
#    - aratios auto output
# ══════════════════════════════════════════════════════
class RPPromptParser:
    CATEGORY = "Regional Prompter"
    cnr_id  = "ComfyUI_RP_Cast"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "beautiful scenery, 2girls\n"
                        "ADDCOMM\n"
                        "high quality, masterpiece\n"
                        "ADDBASE\n"
                        "character A, blonde hair\n"
                        "ADDCOL\n"
                        "character B, dark hair <lora:charB:0.8>"
                    ),
                }),
                "divide_mode":    (["Horizontal", "Vertical"],),
                "divide_ratio":   ("STRING",  {"default": "",
                                   "tooltip": "Used when auto_div_calc=manual. Leave empty for auto calculation."
                                              }),
                "auto_div_calc":  (["auto", "manual"], {"default": "auto",
                                   "tooltip": "auto: calculate divide_ratio from prompt automatically.\n"
                                              "manual: use the divide_ratio widget value directly."}),
                "debug":          ("BOOLEAN", {"default": False,
                                   "tooltip": "Print debug log when enabled."}),
            },
            "optional": {},
        }

    RETURN_TYPES  = ("RP_SUBPROMPTS", "RP_LORA_MAP", "RP_DIV_RATIO", "RP_DIV_MODE", "STRING")
    RETURN_NAMES  = ("regional_prompts_nolora", "regional_lora_map", "divide_ratio", "divide_mode", "original_prompts")
    FUNCTION      = "execute"

    def execute(self, prompt, divide_mode="Horizontal", divide_ratio="", auto_div_calc="auto", debug=False):
        _dbg = print if debug else lambda *a, **kw: None
        # Auto-detect usebase/usecom from ADDBASE/ADDCOMM keywords in prompt
        # Reflects prompt structure independently from RPKSampler use_base/use_common
        from core.prompt_parser import KEYBASE, KEYCOMM
        _auto_usebase = KEYBASE in prompt
        _auto_usecom  = KEYCOMM in prompt

        raw, nolora, lora_map, _, common_text, col_texts = parse_prompt(
            prompt=prompt, usebase=_auto_usebase, usecom=_auto_usecom
        )
        auto_ratio = _auto_aratios(prompt, divide_mode)
        if auto_div_calc == "manual":
            manual = (divide_ratio or "").strip()
            final_ratio = manual if (manual and manual.lower() != "auto") else auto_ratio
            src = f"manual({final_ratio})" if (manual and manual.lower() != "auto") else "manual→auto-fallback"
        else:
            final_ratio = auto_ratio
            src = "auto"

        # 2D detection (auto-detect usebase)
        _2d_struct = get_2d_structure(prompt, usebase=_auto_usebase)
        _is_2d = _2d_struct is not None

        # Header always printed
        print(f"[RPPromptParser] areas={len(nolora)}  divide_mode={divide_mode}  "
              f"auto_usebase={_auto_usebase}  auto_usecom={_auto_usecom}  2D={'✓' if _is_2d else '✗'}")

        # Detail logs only when debug=True
        if _is_2d:
            _dbg(f"  [2D] rows_structure={_2d_struct}")
        _dbg(f"  divide_ratio={final_ratio}  ({src})")

        # Map COL index (excl. BASE) → DIV[row,col]
        def _col_to_div_label(col_i, ratio_str, is_vertical):
            try:
                rows = ratio_str.split(";")
                parsed = []
                for rs in rows:
                    cols = [v.strip() for v in rs.split(",") if v.strip()]
                    if cols: parsed.append(len(cols))
            except Exception:
                return f"DIV[0,{col_i}]"
            idx = 0
            if not is_vertical:
                for ri, ncols in enumerate(parsed):
                    for ci in range(ncols):
                        if idx == col_i: return f"DIV[{ri},{ci}]"
                        idx += 1
            else:
                for ci, nrows in enumerate(parsed):
                    for ri in range(nrows):
                        if idx == col_i: return f"DIV[{ri},{ci}]"
                        idx += 1
            return f"DIV[0,{col_i}]"

        _is_v_log = "Ver" in divide_mode
        _start_i  = 1 if _auto_usebase else 0

        # 1. [common]
        _global_text = prompt.split("ADDCOMM")[0].strip() if "ADDCOMM" in prompt else ""
        if _global_text:
            _dbg(f"  [common] '{_global_text[:80]}'")
        else:
            _dbg(f"  [common] (none)")

        # 2. [base]
        if _auto_usebase and col_texts:
            base_col = col_texts[0]
            _dbg(f"  [base]   '{base_col[:80]}'  loras={lora_map.get(0, {})}")
        else:
            _dbg(f"  [base]   (none)")

        # 3. [DIV[row,col]]
        for i, p in enumerate(nolora[_start_i:], start=0):
            col_idx  = i
            map_idx  = col_idx + _start_i
            col_part = col_texts[map_idx] if map_idx < len(col_texts) else ""
            tag      = _col_to_div_label(col_idx, final_ratio, _is_v_log)
            _dbg(f"  [{tag}] col_only='{col_part[:60]}'  loras={lora_map.get(map_idx, {})}")

        prompts_data = {
            "nolora":        nolora,
            "common":        common_text,
            "col_texts":     col_texts,
            "is_2d":         _is_2d,
            "row_structure": _2d_struct or [],
            "has_base":      _auto_usebase,   # ADDBASE detection result
            "has_common":    _auto_usecom,    # ADDCOMM detection result
        }
        return (prompts_data, lora_map, final_ratio, divide_mode, prompt)


# ══════════════════════════════════════════════════════
# 2. RPRatioParser
# ══════════════════════════════════════════════════════
class RPRatioParser:
    CATEGORY = "Regional Prompter"
    cnr_id  = "ComfyUI_RP_Cast"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "divide_ratio": ("RP_DIV_RATIO", {"default": "1,1", "multiline": False,
                                            "forceInput": True}),
                "divide_mode":  ("RP_DIV_MODE", {"default": "Horizontal",
                                            "forceInput": True}),
                "threshold":    ("STRING", {"default": "1.0",
                                            "tooltip": "End step ratio for regional conditioning (0~1). "
                                                       "Per-region: '0.4,0.6'. "
                                                       "1.0=apply for all steps."}),
            },
        }

    RETURN_TYPES  = ("RP_REGIONS", "RP_THRESHOLD")
    RETURN_NAMES  = ("regional_col_n_row", "threshold_out")
    FUNCTION      = "execute"

    def execute(self, divide_ratio, divide_mode, threshold="1.0"):
        region_rows = parse_regions(
            aratios=divide_ratio, bratios="0", mode=divide_mode, prompt=""
        )
        areas = sum(len(r.cols) for r in region_rows)
        n_cols = areas  # COL count (use_base handled in KSampler)

        # Parse threshold: "0.4" or "0.4,0.6" (per-region)
        # threshold_start fixed to 0, only threshold_end is controlled
        def _parse_thresholds(s, n):
            try:
                parts = [float(x.strip()) for x in str(s).split(",") if x.strip()]
                parts = [max(0.0, min(1.0, v)) for v in parts]
                if not parts:
                    parts = [1.0]
                if len(parts) == 1:
                    parts = parts * n
                elif len(parts) < n:
                    parts = parts + [parts[-1]] * (n - len(parts))
                else:
                    parts = parts[:n]
                return parts
            except Exception:
                return [1.0] * n

        threshold_list = _parse_thresholds(threshold, max(n_cols, 1))

        threshold_data = {
            "start_list": [0.0] * len(threshold_list),  # always 0
            "end_list":   threshold_list,
        }

        prefix = "H" if divide_mode == "Horizontal" else "V"
        parts  = []
        _is_v_rp = (divide_mode != "Horizontal")
        for ri, row in enumerate(region_rows):
            cells = " ".join(f"[{c.st:.2f}~{c.ed:.2f}]" for c in row.cols)
            tag = f"C{ri}" if _is_v_rp else f"R{ri}"
            seg_label = "col" if _is_v_rp else "row"
            parts.append(f"{tag}({row.st:.2f}~{row.ed:.2f}): {cells}")
        region_coords = f"{prefix} | " + " | ".join(parts)

        print(f"[RPRatioParser] divide_mode={divide_mode}  areas={areas}  coords={region_coords}")
        print(f"[RPRatioParser] threshold='{threshold}'  →  per-region end={threshold_list}"
              f"  (start=0 fixed)")
        return (region_rows, threshold_data)


# ══════════════════════════════════════════════════════
# 3. RPKSampler
#
# LoRA replacement redesign:
#   Calls separate apply_model per division inside model_function_wrapper.
#   
#   ComfyUI base sampling structure:
#     sampler receives x(noisy latent) and calls apply_model(x, t, cond)
#     → denoised = apply_model(...)
#
#   RP method (model_function_wrapper):
#     positive conditioning is [cond0, cond1, cond2, ...] (per division)
#     ComfyUI batches these as x_repeated = x.repeat(areas+1, ...)
#     → input shape: [(areas+1)*batch, C, H, W]
#
#   Correct blending:
#     apply spatial filter to denoised result in wrapper
#     blend each division's denoised with its filter
#
#   LoRA application:
#     use LoRA-patched model per division when calling apply_model
#     → call apply_model separately per division in wrapper
# ══════════════════════════════════════════════════════
