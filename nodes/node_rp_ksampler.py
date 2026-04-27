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

class RPKSampler:
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
                "threshold":               ("RP_THRESHOLD",),
                "latent_image":            ("LATENT",),
                "seed":         ("INT",   {"default": 0,
                                           "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps":        ("INT",   {"default": 20, "min": 1, "max": 200}),
                "cfg":          ("FLOAT", {"default": 7.0, "min": 0.0,
                                           "max": 30.0, "step": 0.1}),
                "sampler_name": (_SAMPLERS,),
                "scheduler":    (_SCHEDULERS,),
                "denoise":      ("FLOAT", {"default": 1.0, "min": 0.0,
                                           "max": 1.0, "step": 0.01}),
                "use_base":     ("BOOLEAN", {"default": False,
                                 "tooltip": "Set the same as use_base in RPPromptParser."}),
                "use_common":   ("BOOLEAN", {"default": True}),
                "base_ratio":   ("STRING",  {"default": "0.2",
                                             "tooltip": "BASE:REGION blend ratio. 0.2 → 20% BASE + 80% REGION. "
                                                        "Per-region: '0.2,0.3,0.5'"}),
                "lora_weight_adj": ("INT", {"default": 0, "min": 0, "max": 500,
                                                   "tooltip": "LoRA weight multiplier (%). "
                                                              "0=disabled, 100=original, 50=half, 200=double"}),
                "debug":           ("BOOLEAN", {"default": False,
                                   "tooltip": "Print debug log when enabled."}),
            },
            "optional": {
                "divide_mode":  ("RP_DIV_MODE", {"default": "Horizontal",
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

    def execute(self, model, clip, vae, regional_prompts_nolora, regional_col_n_row,
                regional_lora_map, negative, threshold, latent_image,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                divide_mode="Horizontal",
                use_base=False, use_common=True,
                base_ratio="0.2", lora_weight_adj=0, debug=False):

        _dbg = print if debug else lambda *a, **kw: None
        if not _COMFY_OK:
            raise RuntimeError("comfy module not found.")

        import torch
        import comfy.sample
        import comfy.model_management as mm

        # Normalize internal variable names (backward compat)
        region_rows  = regional_col_n_row
        col_lora_map = regional_lora_map if regional_lora_map else {}
        mode         = divide_mode

        # LoRA weight multiplier: weight * (lora_weight_adj / 100)
        # 0=disabled, 100=original, 50=half, 200=double
        if lora_weight_adj > 0 and col_lora_map:
            rate = lora_weight_adj / 100.0
            col_lora_map = {
                div_idx: {name: weight * rate for name, weight in loras.items()}
                for div_idx, loras in col_lora_map.items()
            }
            _dbg(f"  [RP LoRA] Weight Adj {lora_weight_adj}% → ×{rate:.2f}")

        # Parse base_ratio: "0.2" or "0.2,0.3,0.5" (per-region)
        # base_ratio_list[i] = BASE contribution ratio for COL i
        # formula: result = base_ratio*BASE + (1-base_ratio)*COL
        def _parse_base_ratios(s, n_cols):
            try:
                parts = [float(x.strip()) for x in str(s).split(",") if x.strip()]
                parts = [max(0.0, min(1.0, v)) for v in parts]
                if len(parts) == 0:
                    parts = [0.2]
                if len(parts) == 1:
                    parts = parts * n_cols
                elif len(parts) < n_cols:
                    parts = parts + [parts[-1]] * (n_cols - len(parts))
                else:
                    parts = parts[:n_cols]
                return parts
            except Exception:
                return [0.2] * n_cols

        device  = mm.get_torch_device()
        samples = latent_image["samples"].to(device)
        C, h, w = samples.shape[1], samples.shape[-2], samples.shape[-1]

        # ── Model architecture compatibility check ────────────
        _model_type_name = ""
        _area_conditioning_supported = True
        _expected_channels = C
        _is_5d_latent = samples.ndim == 5  # 5D latent e.g. Qwen [B,C,T,H,W]
        try:
            diff_model = model.get_model_object("diffusion_model")
            _model_type_name = type(diff_model).__module__ + "." + type(diff_model).__name__
            _unsupported_keywords = ("lumina", "dit", "flux", "sd3", "aura", "qwen")
            if any(k in _model_type_name.lower() for k in _unsupported_keywords):
                _area_conditioning_supported = False

            # Check model expected in_channels
            _in_ch = getattr(diff_model, "in_channels", None)
            if _in_ch is None:
                _in_ch = getattr(diff_model, "img_in", None)
                if _in_ch is not None:
                    _in_ch = getattr(_in_ch, "in_features", None)
            if _in_ch and isinstance(_in_ch, int):
                _expected_channels = _in_ch

        except Exception:
            pass

        # Detect channel mismatch (skip check for 5D latent)
        if not _is_5d_latent and _expected_channels != C:
            raise RuntimeError(
                f"[RPKSampler] Latent channel mismatch!\n"
                f"  Model expected channels: {_expected_channels}ch\n"
                f"  Input latent channels: {C}ch (shape={tuple(samples.shape)})\n"
                f"\n"
                f"  Fix: Use EmptySD3LatentImage node with\n"
                f"  channels={_expected_channels}.\n"
                f"  (EmptyLatentImage=4ch, Z-Image requires 16ch)"
            )
        if _is_5d_latent:
            _dbg(f"  [RPKSampler] 5D latent detected: {tuple(samples.shape)} → single conditioning")

        if not _area_conditioning_supported:
            _dbg(f"  ⚠ Model '{_model_type_name}' → area conditioning not supported, using single conditioning")
        else:
            _dbg(f"  Model type: {_model_type_name or 'unknown'}")

        print(f"\n[RPKSampler] ══════════════════════════════════")
        _dbg(f"  steps={steps}  cfg={cfg}  mode={mode}")
        _dbg(f"  use_base={use_base}  use_common={use_common}"
              f"  base_ratio={base_ratio}")

        # Convert threshold → per-region sigma range
        # threshold_data: {start_list: [0,0,...], end_list: [0.4,0.6,...]}
        # start fixed to 0, only end is per-region
        import comfy.samplers as _cs
        _model_sampling = model.get_model_object("model_sampling")
        _all_sigmas = _cs.calculate_sigmas(_model_sampling, scheduler, steps)

        def _th_to_sigma(ratio):
            """Convert step ratio(0~1) → sigma value. Higher ratio = later step (lower sigma)."""
            idx = int(max(0.0, min(1.0, ratio)) * (steps - 1))
            return _all_sigmas[idx].item()

        if threshold is not None and isinstance(threshold, dict):
            end_list   = threshold.get("end_list",   [1.0])
            start_list = threshold.get("start_list", [0.0])  # always 0
        else:
            end_list   = [1.0]
            start_list = [0.0]

        # Per-region sigma list (by COL index)
        # sigma_max_list[i] = start sigma for COL i conditioning (start=0 → max sigma)
        # sigma_min_list[i] = end sigma for COL i conditioning (based on end ratio)
        def _expand_list(lst, n):
            if len(lst) == 0: lst = [lst[0] if lst else 1.0]
            if len(lst) < n:  lst = lst + [lst[-1]] * (n - len(lst))
            return lst[:n]

        _dbg(f"  threshold  end_list={end_list}  (start=0 fixed)")
        _dbg(f"  latent shape={samples.shape}")
        _dbg(f"  vae={'connected' if vae else 'none'}")
        _dbg(f"  col_lora_map={col_lora_map}")

        # ── CLIP encode (Prompt-EX) ──────────────────────────
        # Prompt-EX: encode common and col separately, then concat on dim=1
        # → CLIP attention processes common and col tokens as separate chunks
        # → col-specific features reflected strongly without dilution by common
        #
        # Normal: encode("common, col") → single 77-token (common dilutes col)
        # Prompt-EX: encode("common") + encode("col") → 154-token concat

        # regional_prompts_nolora: dict(Prompt-EX) or list(legacy compat)
        if isinstance(regional_prompts_nolora, dict):
            nolora_list  = regional_prompts_nolora["nolora"]
            common_text  = regional_prompts_nolora.get("common", "")
            col_texts    = regional_prompts_nolora.get("col_texts", [])
            _is_2d       = regional_prompts_nolora.get("is_2d", False)
            _row_struct  = regional_prompts_nolora.get("row_structure", [])
            # has_base/has_common: keyword detection result from prompt
            # Effective only when RPKSampler use_base/use_common is True
            # → Both must be True to activate (user intent + prompt structure match)
            _has_base    = regional_prompts_nolora.get("has_base",   use_base)
            _has_common  = regional_prompts_nolora.get("has_common", use_common)
            use_base     = use_base   and _has_base
            # use_common: widget value takes priority - active when common_text exists and use_common=True
            # has_common is structure detection, but use_common widget=True overrides
            _use_com     = use_common
            prompt_ex    = bool(common_text and _use_com)
        elif isinstance(regional_prompts_nolora, str):
            nolora_list  = [regional_prompts_nolora]
            common_text  = ""
            col_texts    = nolora_list
            _is_2d       = False
            _row_struct  = []
            prompt_ex    = False
        else:
            nolora_list  = list(regional_prompts_nolora)
            common_text  = ""
            col_texts    = nolora_list
            prompt_ex    = False

        _dbg(f"  [Prompt-EX] {'✓ active' if prompt_ex else '✗ inactive (no common or use_common=False)'}")
        if prompt_ex:
            _dbg(f"    common='{common_text[:60]}'")

        import torch as _t

        # ── Batch CLIP encode (load/unload CLIP model once) ─────
        # Batch tokenize all col texts → encode once → slice results
        # common_text is shared across all DIVs, encode only once
        #
        # Processing order:
        #   1. Batch tokenize N col texts
        #   2. encode_from_tokens(batch) → cond[N, seq, dim], pooled[N, dim]
        #   3. If prompt_ex: concat common cond with each DIV cond
        #   4. Slice results into conds list

        # Prepare texts to encode
        encode_texts = []  # col_only texts to encode
        full_texts   = []  # full texts for logging (LoRA removed)
        for i, ft in enumerate(nolora_list):
            ft_clean = _RE_LORA.sub("", ft).strip()
            full_texts.append(ft_clean)
            if prompt_ex and i < len(col_texts):
                ct = _RE_LORA.sub("", col_texts[i]).strip()
                encode_texts.append(ct if ct else ft_clean)
            else:
                encode_texts.append(ft_clean)

        # Encode common_text once
        _c_com = None
        if prompt_ex and common_text:
            tok_com = clip.tokenize(common_text)
            _c_com, _ = _enc(clip, tok_com)


        # Batch tokenize col texts
        # clip.tokenize(text) → dict{key: [[chunk1_tokens], [chunk2_tokens], ...]}
        # Each text may have different chunk count (multi-chunk for >77 tokens)
        tok_list = [clip.tokenize(t) for t in encode_texts]

        # Calculate chunk count per text (using first key)
        _first_key = list(tok_list[0].keys())[0]
        chunks_per = [len(tok[_first_key]) for tok in tok_list]  # e.g. [1, 1, 2, ...]

        # Build batch: concat chunks of all texts per key
        batch_tokens = {}
        for key in tok_list[0].keys():
            combined = []
            for tok in tok_list:
                combined.extend(tok[key])
            batch_tokens[key] = combined

        # Batch encode (load CLIP model once)
        batch_cond, batch_pooled = _enc(clip, batch_tokens)

        # batch_cond:   [sum(chunks_per), seq, dim]
        # batch_pooled: [sum(chunks_per), dim]

        # Slice correct chunk range per text to build conds
        conds = []
        offset = 0
        for i, (ft_clean, et) in enumerate(zip(full_texts, encode_texts)):
            n_ch = chunks_per[i]
            c_col = batch_cond[offset:offset + n_ch]      # [n_ch, seq, dim]
            p_col = batch_pooled[offset:offset + n_ch]    # [n_ch, dim]
            # pooled from last chunk (ComfyUI standard)
            p_col = p_col[-1:] if n_ch > 1 else p_col
            offset += n_ch

            if prompt_ex and _c_com is not None and et and c_col.shape[0] > 0:
                cond   = _t.cat([_c_com, c_col], dim=1)  # [1, 154+, dim]
                pooled = p_col
                mode_str = "EX"
            elif c_col.shape[0] > 0:
                cond   = c_col
                pooled = p_col
                mode_str = "full"
            else:
                # fallback: individual encode
                tok = clip.tokenize(et or ft_clean)
                cond, pooled = _enc(clip, tok)
                mode_str = "full(fallback)"

            conds.append([cond, {"pooled_output": pooled}])

            if use_base and i == 0:
                label = "BASE"
            elif _is_2d and _row_struct:
                col_seq = i - (1 if use_base else 0)
                ri, ci, acc = 0, 0, 0
                for r, n in enumerate(_row_struct):
                    if col_seq < acc + n:
                        ri, ci = r, col_seq - acc
                        break
                    acc += n
                label = f"DIV[{ri},{ci}]"
            elif use_base:
                label = f"DIV[0,{i-1}]"
            else:
                label = f"DIV[0,{i}]"
            _dbg(f"  cond[{label}] [{mode_str}] shape={tuple(cond.shape)}"
                  f"  col='{et[:50]}'  full='{ft_clean[:30]}'")

        actual_areas = len(conds)
        if _is_2d:
            _dbg(f"  actual_areas={actual_areas}  [2D] row_structure={_row_struct}"
                  f"  (rows={len(_row_struct)}, cols_per_row={_row_struct})")
        else:
            _dbg(f"  actual_areas={actual_areas}  [1D]")

        # ── Build filters ───────────────────────────────
        raw_filters = make_filters(
            region_rows=region_rows, h=h, w=w,
            mode=mode, usebase=use_base, device=str(device),
        )
        filters = [f.expand(C, h, w).clone() for f in raw_filters]

        # Fix mismatch: pad filters if filters != actual_areas
        if len(filters) != actual_areas:
            _dbg(f"  ⚠️  filters({len(filters)}) != actual_areas({actual_areas}) → patching")
            if len(filters) > actual_areas:
                filters = filters[:actual_areas]
            else:
                # pad missing filters with full mask (keep front mapping)
                while len(filters) < actual_areas:
                    filters.append(torch.ones(C, h, w, device=device))

        for fi, fil in enumerate(filters):
            _dbg(f"  filter[{fi}] coverage={fil[0].float().mean().item():.1%}")

        # ── LoRA manager: pre-build model clone dict per division ──
        has_lora = (
            col_lora_map is not None
            and isinstance(col_lora_map, dict)
            and len(col_lora_map) > 0
            and _COMFY_OK
        )

        if has_lora:
            _dbg(f"\n  [RP LoRA] col_lora_map analysis:")
            lora_mgr = LoRADivisionManager()

            # Build col_idx → DIV[r,c] or BASE label function
            def _make_div_label(col_idx, use_base=use_base,
                                is_2d=_is_2d, row_struct=_row_struct):
                if use_base and col_idx == 0:
                    return "BASE"
                div_i = col_idx - (1 if use_base else 0)
                if is_2d and row_struct:
                    ri, ci, acc = 0, 0, 0
                    for r, n in enumerate(row_struct):
                        if div_i < acc + n:
                            return f"DIV[{r},{div_i - acc}]"
                        acc += n
                return f"DIV[0,{div_i}]"

            lora_mgr.setup(
                col_lora_map=col_lora_map, division_count=actual_areas,
                base_model=model, base_clip=clip,
                div_label_fn=_make_div_label
            )
            lora_mgr.prebuild_cache()

            # Log: distinguish BASE/COL
            from core.lora_manager import _make_addnet_key
            _lora_col_seq = 0
            for i in range(actual_areas):
                if use_base and i == 0:
                    label = "BASE"
                elif _is_2d and _row_struct:
                    ri2, ci2, acc2 = 0, 0, 0
                    for r2, n2 in enumerate(_row_struct):
                        if _lora_col_seq < acc2 + n2:
                            ri2, ci2 = r2, _lora_col_seq - acc2
                            break
                        acc2 += n2
                    label = f"DIV[{ri2},{ci2}]"
                    _lora_col_seq += 1
                else:
                    label = f"DIV[0,{_lora_col_seq}]"
                    _lora_col_seq += 1
                loras_i = col_lora_map.get(i, {})
                akey = _make_addnet_key(loras_i)
                cached = lora_mgr._cache.get(akey)
                if loras_i:
                    _dbg(f"    [{label}] LoRA={loras_i} → {'cache_hit' if cached else 'not_built'}")
                else:
                    _dbg(f"    [{label}] no LoRA → base model")
        else:
            _dbg(f"  [RP LoRA] no LoRA")

        _dbg(f"══════════════════════════════════════════════\n")

        # ── Apply Area Conditioning ─────────────────────────
        # SD-WebUI original formula:
        #   result = base_ratio * BASE + (1 - base_ratio) * REGION
        # → BASE strength = base_ratio  (full area)
        # → COL  strength = 1-base_ratio (per area)
        # base_ratio=0.2 → BASE(0.2,full) + COL(0.8,area) → sum=1.0

        n_cols = actual_areas - (1 if use_base else 0)
        base_ratio_list = _parse_base_ratios(base_ratio, max(n_cols, 1))
        _dbg(f"  base_ratio parsed: '{base_ratio}' → {base_ratio_list}")

        # Per-region threshold sigma list (by COL)
        _end_list = _expand_list(list(end_list), max(n_cols, 1))
        _sigma_max_global = _all_sigmas[0].item()
        _col_sigma_mins = [_th_to_sigma(e) for e in _end_list]
        _dbg(f"  threshold per-region sigma_min={[f'{s:.3f}' for s in _col_sigma_mins]}")

        # overlap: x-direction boundary blending (small value preserves region character)
        _overlap = max(2, min(w // (len(conds) * 4), 8))
        _overlap_y = _overlap
        _overlap_x = _overlap

        area_conds = []
        col_idx = 0
        for a, (cond_item, fil) in enumerate(zip(conds, filters)):
            cond_tensor, cond_dict = cond_item
            new_dict = cond_dict.copy()

            is_base = (use_base and a == 0)

            if is_base:
                # BASE: strength=1.0 (full area, strong)
                # COL strength=(1-base_ratio) for natural blend
                new_dict['strength']  = 1.0
                new_dict['min_sigma'] = 0.0
                new_dict['max_sigma'] = 99.0
                _dbg(f"  [area] BASE  strength=1.0  sigma=[0~99] (full)")
            else:
                br            = base_ratio_list[col_idx] if col_idx < len(base_ratio_list) else 0.2
                sigma_min_col = _col_sigma_mins[col_idx] if col_idx < len(_col_sigma_mins) else 0.0

                # Calculate 2D label
                if _is_2d and _row_struct:
                    ri2, ci2, acc2 = 0, 0, 0
                    for r2, n2 in enumerate(_row_struct):
                        if col_idx < acc2 + n2:
                            ri2, ci2 = r2, col_idx - acc2
                            break
                        acc2 += n2
                    area_label = f"DIV[{ri2},{ci2}]"
                else:
                    area_label = f"DIV[0,{col_idx}]"

                col_idx += 1

                spatial = fil[0]
                rows = spatial.any(dim=1).nonzero(as_tuple=True)[0]
                cols_px = spatial.any(dim=0).nonzero(as_tuple=True)[0]

                if len(rows) > 0 and len(cols_px) > 0:
                    y0, y1 = rows[0].item(), rows[-1].item() + 1
                    x0, x1 = cols_px[0].item(), cols_px[-1].item() + 1

                    if mode == "Horizontal":
                        # Horizontal: overlap both row(y) and col(x) boundaries
                        x0_ov = max(0, x0 - _overlap_x)
                        x1_ov = min(w, x1 + _overlap_x)
                        y0_ov = max(0, y0 - _overlap_y)
                        y1_ov = min(h, y1 + _overlap_y)
                    else:
                        # Vertical: overlap both col(x) and row(y) boundaries
                        x0_ov = max(0, x0 - _overlap_x)
                        x1_ov = min(w, x1 + _overlap_x)
                        y0_ov = max(0, y0 - _overlap_y)
                        y1_ov = min(h, y1 + _overlap_y)

                    area_h = y1_ov - y0_ov
                    area_w = x1_ov - x0_ov

                    new_dict['area']      = (area_h, area_w, y0_ov, x0_ov)
                    new_dict['strength']  = 1.0 - br   # SD-WebUI formula: 1 - base_ratio
                    new_dict['min_sigma'] = sigma_min_col
                    new_dict['max_sigma'] = _sigma_max_global
                    _dbg(f"  [area] {area_label}"
                          f"  area=({area_h},{area_w},{y0_ov},{x0_ov})"
                          f"  strength={1.0-br:.2f}  base_ratio={br:.2f}"
                          f"  sigma=[{sigma_min_col:.3f}~{_sigma_max_global:.3f}]"
                          f"  coverage={spatial.float().mean().item():.1%}")
                else:
                    new_dict['strength']  = 1.0 - br
                    new_dict['min_sigma'] = sigma_min_col
                    new_dict['max_sigma'] = _sigma_max_global
                    _dbg(f"  [area] {area_label}  area calc failed → full")

            area_conds.append([cond_tensor, new_dict])

        # ── sampling ─────────────────────────────────────────────
        import comfy.model_management as _mm

        # Log negative conditioning check
        neg_len = len(negative) if negative else 0
        if neg_len > 0:
            neg_cond = negative[0][0]
            _dbg(f"\n  [negative] conditioning check: {neg_len} items"
                  f"  shape={tuple(neg_cond.shape)}"
                  f"  → applied to full area (no area) ✓")
        else:
            _dbg(f"\n  [negative] ⚠️  no conditioning → CFG may malfunction")

        # Model does not support area conditioning → single conditioning fallback
        if not _area_conditioning_supported:
            _fb_cond   = conds[0][0]
            _fb_pooled = conds[0][1].get("pooled_output")
            _fb_extra  = {k: v for k, v in conds[0][1].items()
                          if k not in ("pooled_output",)}  # preserve original extra keys

            # Qwen: keep original conditioning structure without cross_attn key
            # Lumina2: requires cross_attn key
            _is_qwen = "qwen" in _model_type_name.lower()
            if _is_qwen:
                _fallback_cond = [[_fb_cond, {"pooled_output": _fb_pooled, **_fb_extra}]]
                _dbg(f"  [fallback] Qwen single conditioning  shape={tuple(_fb_cond.shape)}")
            else:
                _fallback_cond = [[_fb_cond, {
                    "pooled_output": _fb_pooled,
                    "cross_attn":    _fb_cond,
                    **_fb_extra,
                }]]
                _dbg(f"  [fallback] single conditioning  shape={tuple(_fb_cond.shape)}  cross_attn key used")
            positive_for_sample = _fallback_cond
        else:
            positive_for_sample = area_conds

        _dbg(f"  [positive] {len(positive_for_sample)} items"
              f"  {'(single fallback)' if not _area_conditioning_supported else '(BASE+COL area)'}")
        _dbg(f"  cfg={cfg}  → denoised = uncond + {cfg}*(cond-uncond)")
        print(f"\n[RPKSampler] sampling start")
        noise = comfy.sample.prepare_noise(samples, seed, None)

        # ── LoRA division wrapper ──────────────────────────────────
        sample_model = model
        if has_lora:
            _n_div      = actual_areas
            _total      = _n_div + 1
            _global_cnt = [0]

            def _rp_lora_wrapper(apply_model_fn, args):
                cnt      = _global_cnt[0]
                _global_cnt[0] += 1

                call_in  = cnt % _total
                div_idx  = call_in

                input_x  = args["input"]
                timestep = args["timestep"]
                c        = args["c"]

                if div_idx >= _n_div:
                    return apply_model_fn(input_x, timestep, **c)

                div_result = lora_mgr.get_model_for_division(div_idx)
                if div_result is not None:
                    div_model, _ = div_result
                    try:
                        return apply_model_fn(input_x, timestep, **c)
                    except Exception as e:
                        _dbg(f"[RP LoRA] wrapper DIV[{div_idx}] error: {e}")

                return apply_model_fn(input_x, timestep, **c)

            sample_model = model.clone()
            sample_model.set_model_unet_function_wrapper(_rp_lora_wrapper)
            _dbg(f"  [RP LoRA] wrapper applied  actual_areas={actual_areas}")

        output = comfy.sample.sample(
            model        = sample_model,
            noise        = noise,
            steps        = steps,
            cfg          = cfg,
            sampler_name = sampler_name,
            scheduler    = scheduler,
            positive     = positive_for_sample,
            negative     = negative,
            latent_image = samples,
            denoise      = denoise,
            seed         = seed,
        )
        print(f"[RPKSampler] sampling done")

        import torch as _torch
        if isinstance(output, dict):
            output_tensor = output["samples"]
        else:
            output_tensor = output
        print(f"[RPKSampler] done  output.shape={output_tensor.shape}\n")

        # ── VAE decode ───────────────────────────────
        image = None
        if vae is not None:
            try:
                image = vae.decode(output_tensor)
                print(f"[RPKSampler] VAE decode  image.shape={image.shape}")
            except Exception as e:
                print(f"[RPKSampler] VAE decode failed: {e}")

        if image is None:
            image = _torch.zeros(1, h * 8, w * 8, 3)

        # Release model reference (prevent circular ref)
        try:
            if sample_model is not model:
                if hasattr(sample_model, "patches"):
                    sample_model.patches.clear()
                del sample_model
        except Exception:
            pass
        # Release conditioning/noise references
        try:
            del conds, area_conds, positive_for_sample, noise
        except Exception:
            pass
        import gc; gc.collect()
        try:
            import comfy.model_management as _cmm
            _cmm.soft_empty_cache()
        except Exception:
            pass

        return ({"samples": output_tensor}, image)


