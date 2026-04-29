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

# ══════════════════════════════════════════════════════
# 7. RPKSamplerZImage
#    - Z-Image(Lumina2) dedicated RPKSampler
#    - Auto-generate 16ch latent (EmptySD3LatentImage style)
#    - ModelSamplingAuraFlow(shift) sigma schedule
#    - Pass Lumina2 conditioning via cross_attn key
#    - Regional prompts processed as single conditioning
#      (Lumina2 does not support area conditioning)
#    - Reference: ZSamplerTurbo2 (martin-rizzo/ComfyUI-ZImagePowerNodes)
# ══════════════════════════════════════════════════════

class RPKSamplerZImage:
    CATEGORY = "Regional Prompter"
    cnr_id  = "ComfyUI_RP_Cast"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":                   ("MODEL",),
                "clip":                    ("CLIP",),
                "vae":                     ("VAE",),
                "regional_prompts_nolora": ("RP_SUBPROMPTS",),
                "regional_col_n_row":      ("RP_REGIONS",),
                "regional_lora_map":       ("RP_LORA_MAP",),
                "negative":                ("CONDITIONING",),
                "width":    ("INT", {"default": 768,  "min": 64,  "max": 4096, "step": 32}),
                "height":   ("INT", {"default": 1024, "min": 64,  "max": 4096, "step": 32}),
                "seed":     ("INT", {"default": 0,
                                     "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps":    ("INT",   {"default": 8,  "min": 1, "max": 100}),
                "cfg":      ("FLOAT", {"default": 1.0, "min": 0.0,
                                       "max": 30.0, "step": 0.1,
                                       "tooltip": "CFG=1.0 recommended for Z-Image Turbo."}),
                "sampler_name": (_SAMPLERS,),
                "scheduler":    (_SCHEDULERS,),
                "shift":    ("FLOAT", {"default": 3.0, "min": 0.0,
                                       "max": 20.0, "step": 0.5,
                                       "tooltip": "AuraFlow sigma shift. Z-Image Turbo recommended: 3~6."}),
                "denoise":  ("FLOAT", {"default": 1.0, "min": 0.0,
                                       "max": 1.0, "step": 0.01}),
                "lora_weight_adj": ("INT", {"default": 0, "min": 0, "max": 500,
                                            "tooltip": "LoRA weight multiplier (%). "
                                                       "0=disabled, 100=original, 50=half, 200=double"}),
                "debug":           ("BOOLEAN", {"default": False,
                                   "tooltip": "Print debug log when enabled."}),
            },
            "optional": {
                "divide_mode":  ("RP_DIV_MODE",  {"default": "Horizontal",
                                             "forceInput": True,
                                             "tooltip": "Connect divide_mode from RPPromptParser."}),
            },
        }

    RETURN_TYPES  = ("LATENT", "IMAGE")
    RETURN_NAMES  = ("latent", "image")
    FUNCTION      = "execute"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def execute(self, model, clip, vae, regional_prompts_nolora,
                regional_col_n_row,
                regional_lora_map, negative,
                width, height, seed, steps, cfg, sampler_name, scheduler,
                shift, denoise,
                divide_mode="Horizontal",
                lora_weight_adj=0, debug=False):

        _dbg = print if debug else lambda *a, **kw: None
        import torch
        import comfy.sample
        import comfy.samplers as _cs
        import comfy.model_management as mm

        if not _COMFY_OK:
            raise RuntimeError("comfy module not found.")

        device = mm.get_torch_device()

        # ── 1. Prepare 16ch latent (auto-generate from width×height) ──
        iw = (width  // _ZIMAGE_GRID_SIZE) * _ZIMAGE_GRID_SIZE
        ih = (height // _ZIMAGE_GRID_SIZE) * _ZIMAGE_GRID_SIZE
        lw = iw // _ZIMAGE_LATENT_BLOCK
        lh = ih // _ZIMAGE_LATENT_BLOCK
        samples = torch.zeros(
            (1, _ZIMAGE_LATENT_CHANNELS, lh, lw),
            device=mm.intermediate_device()
        )
        _dbg(f"  [RPKSamplerZImage] latent generated: "
              f"{iw}×{ih}px → latent {lw}×{lh} (C={_ZIMAGE_LATENT_CHANNELS})")

        samples = samples.to(device)
        h, w = samples.shape[-2], samples.shape[-1]

        # ── 2. Apply ModelSamplingAuraFlow(shift) ─────────────
        _cloned = False
        model_shifted = model
        if _ZImageAuraFlow is not None:
            try:
                _tmp = model.clone()
                _ms_obj = _ZImageAuraFlow(_tmp.model.model_config)
                _ms_obj.set_parameters(shift=shift, multiplier=1.0)
                _tmp.add_object_patch("model_sampling", _ms_obj)
                del _ms_obj
                model_shifted = _tmp
                _cloned = True
                _dbg(f"  [RPKSamplerZImage] AuraFlow sigma shift={shift} applied")
            except Exception as e:
                _dbg(f"  [RPKSamplerZImage] AuraFlow shift failed: {e} → using default sigma")
                try: del _tmp
                except: pass

        # ── 3. LoRA weight adjustment ────────────────────
        col_lora_map = regional_lora_map if regional_lora_map else {}
        if lora_weight_adj > 0 and col_lora_map:
            rate = lora_weight_adj / 100.0
            col_lora_map = {
                d: {n: wt * rate for n, wt in loras.items()}
                for d, loras in col_lora_map.items()
            }
            _dbg(f"  [RP LoRA] Weight Adj {lora_weight_adj}% → ×{rate:.2f}")

        # ── 4. Parse prompts + natural language position + CLIP encode ─
        # regional_col_n_row → compute normalized boundaries(x0,y0,x1,y1) per DIV
        # → Prepend position label (LEFT/CENTER/RIGHT + UPPER/MID/LOWER) to prompt
        if isinstance(regional_prompts_nolora, dict):
            nolora_list  = regional_prompts_nolora["nolora"]
            common_text  = regional_prompts_nolora.get("common", "")
            col_texts    = regional_prompts_nolora.get("col_texts", [])
        else:
            nolora_list  = list(regional_prompts_nolora)
            common_text  = ""
            col_texts    = nolora_list

        # regional_col_n_row → DIV boundary list [(x0,y0,x1,y1), ...]
        div_bounds = []
        div_indices = []   # [(col_idx, row_idx), ...]
        n_cols_total = 1
        n_rows_total = 1

        if regional_col_n_row:
            _is_v = "Ver" in (divide_mode or "Horizontal")
            rs = regional_col_n_row

            # Handles RegionRow/RegionCell or plain list/numbers
            def _is_region_row(obj):
                return hasattr(obj, 'cols') and hasattr(obj, 'st') and hasattr(obj, 'ed')

            def _row_ncols(row):
                """Number of cols in row."""
                if _is_region_row(row): return len(row.cols)
                try: return len(row)
                except: return 1

            def _row_bounds_x(row):
                """X boundaries of each col in row [st0, st1, ..., ed_last]."""
                if _is_region_row(row):
                    segs = [row.cols[0].st] + [c.ed for c in row.cols]
                    return segs
                # plain number list
                vals = [float(v) for v in row]
                total = sum(vals) or 1
                segs = [0.0]; acc = 0.0
                for v in vals: acc += v/total; segs.append(round(acc,4))
                return segs

            def _row_bounds_y(row):
                """Y range of row (st, ed)."""
                if _is_region_row(row): return row.st, row.ed
                return None, None  # 1D case

            if not _is_region_row(rs[0]) and isinstance(rs[0], (int, float)):
                # 1D: simple number list
                vals = [float(v) for v in rs]
                total = sum(vals) or 1
                segs = [0.0]; acc = 0.0
                for v in vals: acc += v/total; segs.append(round(acc,4))
                for i in range(len(rs)):
                    if _is_v:
                        div_bounds.append((segs[i], 0.0, segs[i+1], 1.0))
                    else:
                        div_bounds.append((0.0, segs[i], 1.0, segs[i+1]))
                    div_indices.append((i, 0))
                n_cols_total = len(rs); n_rows_total = 1
            else:
                # 2D: RegionRow or list of list
                if not _is_v:
                    # Horizontal: rs[ri]=row, cols inside
                    n_rows_total = len(rs)
                    n_cols_total = max(_row_ncols(r) for r in rs)
                    for ri, row in enumerate(rs):
                        y0, y1 = _row_bounds_y(row)
                        if y0 is None:
                            y0 = ri / n_rows_total
                            y1 = (ri+1) / n_rows_total
                        x_segs = _row_bounds_x(row)
                        for ci in range(_row_ncols(row)):
                            div_bounds.append((x_segs[ci], y0, x_segs[ci+1], y1))
                            div_indices.append((ci, ri))
                else:
                    # Vertical: rs[ci]=col, rows inside
                    n_cols_total = len(rs)
                    n_rows_total = max(_row_ncols(c) for c in rs)
                    # x boundary: reinterpret each col y-range(st,ed) as x-axis
                    for ci, col in enumerate(rs):
                        x0, x1 = _row_bounds_y(col)  # Vertical: y→x reinterpret
                        if x0 is None:
                            x0 = ci / n_cols_total
                            x1 = (ci+1) / n_cols_total
                        y_segs = _row_bounds_x(col)   # internal split → y-axis
                        for ri in range(_row_ncols(col)):
                            div_bounds.append((x0, y_segs[ri], x1, y_segs[ri+1]))
                            div_indices.append((ci, ri))

        def _pos_label(col_idx, row_idx, n_cols, n_rows):
            """Convert col/row index → natural language position label.
            Horizontal: MORE LEFT, LEFT, CENTER TO LEFT/RIGHT, RIGHT, MORE RIGHT
            Vertical: UPPER / MID TO UPPER, MID, MID TO LOWER ... / LOWER
            """
            # ── Horizontal label sequence ───────────────────────
            H = []
            if n_cols == 1:
                H = ["CENTER"]
            elif n_cols == 2:
                H = ["LEFT", "RIGHT"]
            elif n_cols == 3:
                H = ["LEFT", "CENTER", "RIGHT"]
            elif n_cols == 4:
                H = ["LEFT", "SECOND LEFT", "FIRST RIGHT", "RIGHT"]
            elif n_cols == 5:
                H = ["MORE LEFT", "LEFT", "CENTER", "RIGHT", "MORE RIGHT"]
            elif n_cols == 6:
                H = ["MORE LEFT", "LEFT", "CENTER TO LEFT",
                     "CENTER TO RIGHT", "RIGHT", "MORE RIGHT"]
            else:
                half = n_cols // 2
                le = max(0, half - 2); re = max(0, half - 2)
                cc = n_cols - le - 1 - re - 1
                mp = ["MORE ", "SECOND MORE ", "THIRD MORE "]
                for i in range(le):
                    H.append((mp[i] if i < len(mp) else f"MORE{i+1} ") + "LEFT")
                H.append("LEFT")
                if cc == 1:
                    H.append("CENTER")
                elif cc == 2:
                    H += ["CENTER TO LEFT", "CENTER TO RIGHT"]
                elif cc == 3:
                    H += ["CENTER TO LEFT", "CENTER", "CENTER TO RIGHT"]
                elif cc == 4:
                    H += ["MORE CENTER TO LEFT", "CENTER TO LEFT",
                          "CENTER TO RIGHT", "MORE CENTER TO RIGHT"]
                else:
                    mp_c = ["MORE ", "SECOND MORE "]
                    for i in range(cc // 2):
                        H.append((mp_c[i] if i < len(mp_c) else f"MORE{i+1} ") + "CENTER TO LEFT")
                    if cc % 2 == 1:
                        H.append("CENTER")
                    for i in range(cc // 2 - 1, -1, -1):
                        H.append((mp_c[i] if i < len(mp_c) else f"MORE{i+1} ") + "CENTER TO RIGHT")
                H.append("RIGHT")
                for i in range(re):
                    H.append((mp[i] if i < len(mp) else f"MORE{i+1} ") + "RIGHT")

            h_label = H[col_idx] if col_idx < len(H) else f"COL{col_idx}"

            # ── Vertical label sequence ────────────────────────
            V = []
            if n_rows == 1:
                V = ["MID"]
            elif n_rows == 2:
                V = ["UPPER", "LOWER"]
            elif n_rows == 3:
                V = ["UPPER", "MID", "LOWER"]
            else:
                mid_count = n_rows - 2
                V.append("UPPER")
                if mid_count == 1:
                    V.append("MID")
                elif mid_count == 2:
                    V += ["MID TO UPPER", "MID TO LOWER"]
                elif mid_count == 3:
                    V += ["MID TO UPPER", "MID", "MID TO LOWER"]
                elif mid_count == 4:
                    V += ["MID TO MORE UPPER", "MID TO UPPER",
                          "MID TO LOWER", "MID TO MORE LOWER"]
                elif mid_count == 5:
                    V += ["MID TO MORE UPPER", "MID TO UPPER", "MID",
                          "MID TO LOWER", "MID TO MORE LOWER"]
                else:
                    # 6+: extend with MORE UPPER variants
                    half_m = mid_count // 2
                    more_up = ["MID TO MORE UPPER", "MID TO SECOND MORE UPPER"]
                    more_dn = ["MID TO MORE LOWER", "MID TO SECOND MORE LOWER"]
                    for i in range(half_m - 1, -1, -1):
                        V.append(more_up[i] if i < len(more_up) else f"MID TO MORE{i+1} UPPER")
                    V.append("MID TO UPPER")
                    if mid_count % 2 == 1:
                        V.append("MID")
                    V.append("MID TO LOWER")
                    for i in range(half_m - 1):
                        V.append(more_dn[i] if i < len(more_dn) else f"MID TO MORE{i+1} LOWER")
                    V.append("MID TO MORE LOWER")
                V.append("LOWER")

            v_label = V[row_idx] if row_idx < len(V) else f"ROW{row_idx}"
            return f"{h_label} AND {v_label}"

        # ── _pos_label and parts below use div_indices, n_cols_total ──

        # Separate BASE/COL: check BASE presence from row_structure
        # If RPPromptParser parsed with use_base=True, nolora_list[0]=BASE
        _has_base = isinstance(regional_prompts_nolora, dict) and \
                    len(nolora_list) > 0 and len(col_texts) > 0 and \
                    col_texts[0] != nolora_list[0]  # BASE: full != col_only
        start_i      = 1 if _has_base else 0
        col_list     = nolora_list[start_i:]
        col_text_list = col_texts[start_i:] if col_texts else col_list
        _use_com     = bool(common_text)

        # Combine each DIV prompt with position natural language (Qwen-style merge)
        # merge structure:
        #   Scene  : {COMMON(count-stripped)}, ({DIV0}) on the [loc] side and ({DIV1}) on the [loc] side,
        #            interacting naturally in the same scene, seamless composition
        #   Detail : {BASE}
        base_text = _RE_LORA.sub("", col_texts[0]).strip() if (_has_base and col_texts) else ""
        div_col_texts = [
            _RE_LORA.sub("", col_text_list[i]).strip()
            for i in range(len(col_text_list))
        ]

        H_side = {
            "LEFT":            "left",
            "CENTER":          "center",
            "RIGHT":           "right",
            "SECOND LEFT":     "second from left",
            "FIRST RIGHT":     "first from right",
            "MORE LEFT":       "far left",
            "MORE RIGHT":      "far right",
            "CENTER TO LEFT":  "center-left",
            "CENTER TO RIGHT": "center-right",
        }
        V_side = {"UPPER": "upper", "MID": "middle", "LOWER": "lower"}

        _div_log = []
        div_phrases = []
        for i, col_text in enumerate(div_col_texts):
            if not col_text:
                continue
            if i < len(div_indices):
                ci, ri = div_indices[i]
                if regional_col_n_row and not _is_v:
                    row_ncols_i = _row_ncols(rs[ri]) if ri < len(rs) else n_cols_total
                    row_nrows_i = n_rows_total
                elif regional_col_n_row and _is_v:
                    row_ncols_i = n_cols_total
                    row_nrows_i = _row_ncols(rs[ci]) if ci < len(rs) else n_rows_total
                else:
                    row_ncols_i = n_cols_total
                    row_nrows_i = n_rows_total
                loc = _pos_label(ci, ri, row_ncols_i, row_nrows_i)
                parts_loc = loc.split(" AND ")
                h_raw  = parts_loc[0].strip()
                v_raw  = parts_loc[1].strip() if len(parts_loc) > 1 else ""
                h_side = H_side.get(h_raw, h_raw.lower())
                v_side = V_side.get(v_raw, v_raw.lower()) if v_raw else ""
                side_str = f"{v_side} {h_side}".strip() if v_side else h_side
                phrase = f"({col_text}) on the {side_str} side"
                _div_log.append((ri, ci, side_str, col_text))
            elif i < len(div_bounds):
                x0,y0,x1,y1 = div_bounds[i]
                cx,cy = (x0+x1)/2,(y0+y1)/2
                h = "left" if cx<1/3 else "center" if cx<2/3 else "right"
                v = "upper" if cy<1/3 else "middle" if cy<2/3 else "lower"
                side_str = f"{v} {h}".strip()
                phrase = f"({col_text}) on the {side_str} side"
                _div_log.append((0, i, side_str, col_text))
            else:
                phrase = f"({col_text})"
                _div_log.append((0, i, "", col_text))
            div_phrases.append(phrase)

        # Combine DIV sentences
        if len(div_phrases) == 0:
            div_sentence = ""
        elif len(div_phrases) == 1:
            div_sentence = div_phrases[0]
        elif len(div_phrases) == 2:
            div_sentence = f"{div_phrases[0]} and {div_phrases[1]}"
        else:
            div_sentence = ", ".join(div_phrases[:-1]) + f", and {div_phrases[-1]}"

        # Remove person count from common_text (prevents double-counting with DIVs)
        import re as _re
        _COUNT_RE = _re.compile(
            r'\b\d+\s*(?:boy|girl|boys|girls|man|woman|men|women|person|people)s?\b'
            r'|(?:and\s+)?\b\d+\s*(?:boy|girl|boys|girls|man|woman|men|women)s?\b',
            _re.IGNORECASE
        )
        def _strip_count(text):
            cleaned = _COUNT_RE.sub('', text)
            cleaned = _re.sub(r',\s*,', ',', cleaned)
            cleaned = _re.sub(r'\s{2,}', ' ', cleaned)
            cleaned = _re.sub(r'^[\s,]+|[\s,]+$', '', cleaned)
            return cleaned.strip()

        common_scene = _strip_count(common_text) if common_text else ""

        scene_parts = []
        if common_scene:
            scene_parts.append(common_scene)
        if div_sentence:
            scene_parts.append(
                div_sentence +
                ", interacting naturally in the same scene, seamless composition"
            )
        scene_line = ", ".join(scene_parts)
        detail_line = base_text.strip() if base_text else ""

        if scene_line and detail_line:
            full_prompt = f"Scene : {scene_line}\nDetail : {detail_line}"
        elif scene_line:
            full_prompt = f"Scene : {scene_line}"
        else:
            full_prompt = detail_line

        if not full_prompt:
            full_prompt = " ".join(t for t in nolora_list if t.strip())

        _dbg(f"  [RPKSamplerZImage] merged prompt:")
        _dbg(f"    [BASE]   '{base_text[:60]}'")
        _dbg(f"    [COMMON] '{common_text[:60]}'")
        for ri, ci, side, txt in _div_log:
            _dbg(f"    [DIV[{ri},{ci}]] side='{side}'  '{txt[:60]}'")

        tok  = clip.tokenize(full_prompt)
        cond, pooled = _enc(clip, tok)
        _dbg(f"    cond.shape={tuple(cond.shape)}")

        # Apply merged LoRA (single model for all)
        sample_model = model_shifted
        if col_lora_map and _COMFY_OK:
            from core.lora_manager import _apply_loras
            merged: dict = {}
            for div_loras in col_lora_map.values():
                for n, wt in div_loras.items():
                    merged[n] = (merged[n] + wt) / 2 if n in merged else wt
            if merged:
                sample_model, _ = _apply_loras(model_shifted, clip, merged)
                _dbg(f"  [RP LoRA] merged applied: {merged}")

        # Lumina2 conditioning (cross_attn key)
        positive = [[cond, {
            "pooled_output": pooled,
            "cross_attn":    cond,   # → Lumina2 num_tokens auto-calculated
        }]]

        # ── 6. Sampling ──────────────────────────────────
        print(f"\n[RPKSamplerZImage] sampling start  "
              f"steps={steps}  cfg={cfg}  shift={shift}  denoise={denoise}")
        noise  = comfy.sample.prepare_noise(samples, seed, None)
        output = comfy.sample.sample(
            model        = sample_model,
            noise        = noise,
            steps        = steps,
            cfg          = cfg,
            sampler_name = sampler_name,
            scheduler    = scheduler,
            positive     = positive,
            negative     = negative,
            latent_image = samples,
            denoise      = denoise,
            seed         = seed,
        )
        if isinstance(output, dict):
            output_tensor = output["samples"]
        else:
            output_tensor = output
        print(f"[RPKSamplerZImage] sampling done  shape={output_tensor.shape}")

        # ── 7. VAE decode ────────────────────────────────
        image = None
        try:
            image = vae.decode(output_tensor)
            print(f"[RPKSamplerZImage] VAE decode  image.shape={image.shape}")
        except Exception as e:
            print(f"[RPKSamplerZImage] VAE decode failed: {e}")

        if image is None:
            import torch as _t
            image = _t.zeros(1, h * _ZIMAGE_LATENT_BLOCK,
                             w * _ZIMAGE_LATENT_BLOCK, 3)

        # Release model reference (prevent circular ref / memory leak)
        try:
            if hasattr(sample_model, 'patches'):
                sample_model.patches.clear()
            if hasattr(sample_model, 'object_patches'):
                sample_model.object_patches.clear()
            if hasattr(sample_model, 'object_patches_backup'):
                sample_model.object_patches_backup.clear()
            del sample_model
        except Exception:
            pass
        try:
            if _cloned and model_shifted is not model:
                if hasattr(model_shifted, 'patches'):
                    model_shifted.patches.clear()
                if hasattr(model_shifted, 'object_patches'):
                    model_shifted.object_patches.clear()
                if hasattr(model_shifted, 'object_patches_backup'):
                    model_shifted.object_patches_backup.clear()
                del model_shifted
        except Exception:
            pass
        import gc; gc.collect()
        try:
            import comfy.model_management as _cmm
            _cmm.soft_empty_cache()
        except Exception:
            pass

        return ({"samples": output_tensor}, image)
