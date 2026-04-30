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

# 8. RPKSamplerQwen
#    - Qwen Image / Qwen-Image-Layered dedicated RPKSampler
#    - Same position natural-language insertion as RPKSamplerZImage
#    - latent: external LATENT input (EmptyQwenImageLayeredLatentImage)
#    - conditioning: keep original structure without cross_attn key
#    - 5D latent [B,C,T,H,W] handling
# ══════════════════════════════════════════════════════
class RPKSamplerQwen:
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
                "width":    ("INT", {"default": 768,  "min": 64, "max": 4096, "step": 16}),
                "height":   ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "seed":     ("INT", {"default": 0,
                                     "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps":    ("INT",   {"default": 20, "min": 1, "max": 100,
                                       "tooltip": "Qwen recommended: 15~20 steps."}),
                "cfg":      ("FLOAT", {"default": 1.0, "min": 0.0,
                                       "max": 30.0, "step": 0.1,
                                       "tooltip": "CFG=1.0 recommended for Qwen distilled."}),
                "sampler_name": (_SAMPLERS,),
                "scheduler":    (_SCHEDULERS,),
                "shift":    ("FLOAT", {"default": 3.0, "min": 0.0,
                                       "max": 20.0, "step": 0.5,
                                       "tooltip": "AuraFlow sigma shift for Qwen. "
                                                  "Raise to 12~13 if output is blurry."}),
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
                regional_col_n_row, regional_lora_map, negative,
                width, height, seed, steps, cfg, sampler_name, scheduler,
                shift, denoise,
                divide_mode="Horizontal", lora_weight_adj=0, debug=False):

        _dbg = print if debug else lambda *a, **kw: None
        import torch
        import comfy.sample
        import comfy.samplers as _cs
        import comfy.model_management as mm

        if not _COMFY_OK:
            raise RuntimeError("comfy module not found.")

        device = mm.get_torch_device()

        # ── 1. Auto-generate Qwen latent (5D [B,C,T,H,W]) ──────
        # Qwen process_img requires 5D input
        # Generate 5D same as EmptyQwenImageLayeredLatentImage
        # C: model img_in.weight[in_feat] / patch_size²
        _latent_channels = 4   # Qwen default (patch_size=4 → 4×16=64)
        _latent_block    = 8   # VAE spatial compression
        _latent_layers   = 1   # T dimension (default 1 frame)
        try:
            diff_model = model.get_model_object("diffusion_model")
            img_in_w   = diff_model.img_in.weight          # shape [out, in_feat]
            in_feat    = img_in_w.shape[1]                 # e.g. 64
            patch_size = getattr(diff_model, 'patch_size', 4)
            _latent_channels = in_feat // (patch_size * patch_size)
            _dbg(f"  [RPKSamplerQwen] img_in.in={in_feat}, "
                  f"patch_size={patch_size} → C={_latent_channels}")
        except Exception as e:
            _dbg(f"  [RPKSamplerQwen] channel auto-detect failed({e}) → C={_latent_channels}")

        iw = (width  // 16) * 16
        ih = (height // 16) * 16
        lw = iw // _latent_block
        lh = ih // _latent_block

        # 5D: [B, C, T, H, W]  (same as EmptyQwenImageLayeredLatentImage)
        samples = torch.zeros(
            (1, _latent_channels, _latent_layers, lh, lw),
            device=mm.intermediate_device()
        ).to(device)
        h, w = lh, lw
        print(f"\n[RPKSamplerQwen] latent generated: {iw}×{ih}px → "
              f"[1,{_latent_channels},{_latent_layers},{lh},{lw}] 5D")

        # ── 2. Apply AuraFlow sigma shift ───────────────────────
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
                _dbg(f"  [RPKSamplerQwen] AuraFlow sigma shift={shift} applied")
            except Exception as e:
                _dbg(f"  [RPKSamplerQwen] shift failed: {e} → using default sigma")
                try: del _tmp
                except: pass

        # ── 3. LoRA weight adjustment ──────────────────────
        col_lora_map = regional_lora_map if regional_lora_map else {}
        if lora_weight_adj > 0 and col_lora_map:
            rate = lora_weight_adj / 100.0
            col_lora_map = {
                d: {n: wt * rate for n, wt in loras.items()}
                for d, loras in col_lora_map.items()
            }
            _dbg(f"  [RP LoRA] Weight Adj {lora_weight_adj}% → ×{rate:.2f}")

        # ── 4. Parse prompts + merge with position natural language ─
        # merge structure:
        #   {BASE},
        #   {COMMON},
        #   In the LEFT of the image, {DIV0_col_only},
        #   In the CENTER of the image, {DIV1_col_only},
        #   In the RIGHT of the image, {DIV2_col_only},
        if isinstance(regional_prompts_nolora, dict):
            nolora_list = regional_prompts_nolora["nolora"]
            common_text = regional_prompts_nolora.get("common", "")
            col_texts   = regional_prompts_nolora.get("col_texts", [])
            has_base    = regional_prompts_nolora.get("has_base", False)
        else:
            nolora_list = list(regional_prompts_nolora)
            common_text = ""
            col_texts   = nolora_list
            has_base    = False

        # BASE: nolora_list[0]=BASE+COL0 full → col_texts[0]=BASE col_only
        # DIV: each col_texts[i] for nolora_list[start_i:] is col_only
        start_i   = 1 if has_base else 0
        base_only = _RE_LORA.sub("", col_texts[0]).strip() if (has_base and col_texts) else ""
        div_col_texts = [
            _RE_LORA.sub("", col_texts[i]).strip() if i < len(col_texts) else ""
            for i in range(start_i, len(nolora_list))
        ]

        # regional_col_n_row → DIV position natural language
        div_indices  = []
        n_cols_total = 1
        n_rows_total = 1

        if regional_col_n_row:
            _is_v = "Ver" in (divide_mode or "Horizontal")
            rs = regional_col_n_row

            def _is_region_row(obj):
                return hasattr(obj, 'cols') and hasattr(obj, 'st') and hasattr(obj, 'ed')

            if _is_v:
                if rs and _is_region_row(rs[0]):
                    n_cols_total = len(rs)
                    n_rows_total = max(len(r.cols) for r in rs) if rs else 1
                    for ci, col_row in enumerate(rs):
                        for ri in range(len(col_row.cols)):
                            div_indices.append((ci, ri))
            else:
                if rs and _is_region_row(rs[0]):
                    n_rows_total = len(rs)
                    n_cols_total = max(len(r.cols) for r in rs) if rs else 1
                    for ri, row in enumerate(rs):
                        for ci in range(len(row.cols)):
                            div_indices.append((ci, ri))

        def _pos_label(col_idx, row_idx, n_cols, n_rows):
            H_labels = {
                1: ["CENTER"], 2: ["LEFT","RIGHT"],
                3: ["LEFT","CENTER","RIGHT"],
                4: ["LEFT","SECOND LEFT","FIRST RIGHT","RIGHT"],
                5: ["MORE LEFT","LEFT","CENTER","RIGHT","MORE RIGHT"],
                6: ["MORE LEFT","LEFT","CENTER TO LEFT",
                    "CENTER TO RIGHT","RIGHT","MORE RIGHT"],
            }
            V_labels = {1: ["MID"], 2: ["UPPER","LOWER"], 3: ["UPPER","MID","LOWER"]}
            h_list = H_labels.get(n_cols, [f"COL{col_idx}"])
            h_part = h_list[col_idx] if col_idx < len(h_list) else f"COL{col_idx}"
            if n_rows == 1:
                return h_part
            v_list = V_labels.get(n_rows)
            if v_list:
                v_part = v_list[row_idx] if row_idx < len(v_list) else f"ROW{row_idx}"
            else:
                mid_n = n_rows - 2
                if   row_idx == 0:          v_part = "UPPER"
                elif row_idx == n_rows - 1: v_part = "LOWER"
                else:
                    mi = row_idx - 1
                    if   mid_n == 1: v_part = "MID"
                    elif mid_n == 2: v_part = ["MID TO UPPER","MID TO LOWER"][mi]
                    elif mid_n == 3: v_part = ["MID TO UPPER","MID","MID TO LOWER"][mi]
                    else:            v_part = f"MID{mi}"
            return f"{h_part} AND {v_part}"

        # Build merge
        # Structure:
        # Scene : {COMMON}, ({DIV0}) on the [loc] side and ({DIV1}) on the [loc] side,
        #         interacting naturally in the same scene, seamless composition
        # Detail : {BASE}
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
        V_side = {
            "UPPER": "upper",
            "MID":   "middle",
            "LOWER": "lower",
        }

        div_phrases = []
        for i, col_only in enumerate(div_col_texts):
            if not col_only:
                continue
            if i < len(div_indices):
                ci, ri = div_indices[i]
                if regional_col_n_row and not _is_v:
                    _rs = regional_col_n_row
                    row_ncols_i = len(_rs[ri].cols) if ri < len(_rs) and _is_region_row(_rs[ri]) else n_cols_total
                    row_nrows_i = n_rows_total
                elif regional_col_n_row and _is_v:
                    _rs = regional_col_n_row
                    row_ncols_i = n_cols_total
                    row_nrows_i = len(_rs[ci].cols) if ci < len(_rs) and _is_region_row(_rs[ci]) else n_rows_total
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

                phrase = f"({col_only}) on the {side_str} side"
                _dbg(f"  [DIV[{ri},{ci}]] side='{side_str}'  '{col_only[:50]}'")
            else:
                phrase = f"({col_only})"
                _dbg(f"  [DIV{i}] (no region)  '{col_only[:50]}'")
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

        # Scene line
        # Remove person count from common_text (prevents double-counting with DIVs)
        # e.g. "1boy and 2girls", "1girl", "2boys" patterns
        import re as _re
        _COUNT_RE = _re.compile(
            r'\b\d+\s*(?:boy|girl|boys|girls|man|woman|men|women|person|people)s?\b'
            r'|(?:and\s+)?\b\d+\s*(?:boy|girl|boys|girls|man|woman|men|women)s?\b',
            _re.IGNORECASE
        )
        def _strip_count(text):
            cleaned = _COUNT_RE.sub('', text)
            cleaned = _re.sub(r',\s*,', ',', cleaned)      # consecutive commas
            cleaned = _re.sub(r'\s{2,}', ' ', cleaned)     # multiple spaces
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

        # Detail line
        detail_line = base_only.strip() if base_only else ""

        # Final merge
        if scene_line and detail_line:
            merged_text = f"Scene : {scene_line}\nDetail : {detail_line}"
        elif scene_line:
            merged_text = f"Scene : {scene_line}"
        else:
            merged_text = detail_line

        _dbg(f"  [COMMON] '{common_text[:80]}'")
        _dbg(f"  [BASE]   '{base_only[:80]}'")
        _dbg(f"  [merged]\n{merged_text[:800]}")

        tok = clip.tokenize(merged_text)
        cond, pooled = _enc(clip, tok)

        _dbg(f"  cond.shape={tuple(cond.shape)}")

        # ── 5. Apply merged LoRA ─────────────────────────────────
        sample_model = model_shifted
        if col_lora_map and _COMFY_OK:
            from core.lora_manager import _apply_loras
            merged_lora: dict = {}
            for div_loras in col_lora_map.values():
                for n, wt in div_loras.items():
                    merged_lora[n] = (merged_lora[n] + wt) / 2 if n in merged_lora else wt
            if merged_lora:
                sample_model, _ = _apply_loras(model_shifted, clip, merged_lora)
                _dbg(f"  [RP LoRA] merged: {merged_lora}")

        # ── 6. Qwen conditioning (no cross_attn key) ──────────
        positive = [[cond, {"pooled_output": pooled}]]

        # ── 7. Sampling ───────────────────────────────────
        print(f"\n[RPKSamplerQwen] sampling  steps={steps}  cfg={cfg}  denoise={denoise}")
        noise  = comfy.sample.prepare_noise(samples, seed, None)
        try:
            sample_model.unpatch_model()
        except Exception:
            pass
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
        output_tensor = output["samples"] if isinstance(output, dict) else output
        print(f"[RPKSamplerQwen] done  shape={output_tensor.shape}")

        # ── 8. VAE decode ─────────────────────────────────
        # Qwen VAE: try decoding 5D [B,C,T,H,W] as-is
        # On failure: remove T dim and retry as 4D
        image = None
        for decode_tensor, desc in [
            (output_tensor,                           "5D as-is"),
            (output_tensor[:,:,0,:,:] if output_tensor.ndim==5 else output_tensor, "5D→4D slice"),
        ]:
            try:
                image = vae.decode(decode_tensor)
                print(f"[RPKSamplerQwen] VAE decode({desc})  shape={image.shape}")
                break
            except Exception as e:
                print(f"[RPKSamplerQwen] VAE decode({desc}) failed: {e}")

        if image is None:
            import torch as _t
            image = _t.zeros(1, h * _latent_block, w * _latent_block, 3)

        # Release model reference (prevent circular ref / memory leak)
        try:
            sample_model.patches = {}
            sample_model.object_patches = {}
            sample_model.object_patches_backup = {}
            del sample_model
        except Exception:
            pass
        try:
            if _cloned and model_shifted is not model:
                model_shifted.patches = {}
                model_shifted.object_patches = {}
                model_shifted.object_patches_backup = {}
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
