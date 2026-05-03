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

# 4. RPRegionalDetailer
#    - Detect persons in each regional area (person_yolov8m-seg.pt)
#    - Extract bbox of largest person per area
#    - Inpaint bbox with regional prompt + LoRA
# ══════════════════════════════════════════════════════
class RPRegionalDetailer:
    CATEGORY = "Regional Prompter"
    cnr_id  = "ComfyUI_RP_Cast"

    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically load bbox model list
        bbox_models = []
        try:
            import folder_paths
            bbox_dir = os.path.join(folder_paths.models_dir, "ultralytics", "bbox")
            segm_dir = os.path.join(folder_paths.models_dir, "ultralytics", "segm")
            for d in [bbox_dir, segm_dir]:
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.endswith(".pt"):
                            prefix = "bbox/" if "bbox" in d else "segm/"
                            bbox_models.append(prefix + f)
        except Exception:
            pass
        if not bbox_models:
            bbox_models = ["bbox/person_yolov8m-seg.pt", "segm/person_yolov8m-seg.pt"]

        return {
            "required": {
                "image":                    ("IMAGE",),
                "model":                    ("MODEL",),
                "clip":                     ("CLIP",),
                "vae":                      ("VAE",),
                "regional_prompts_nolora":  ("RP_SUBPROMPTS",),
                "regional_lora_map":        ("RP_LORA_MAP",),
                "regional_col_n_row":       ("RP_REGIONS",),
                "negative":                 ("CONDITIONING",),
                "divide_mode":              ("RP_DIV_MODE", {"default": "Horizontal",
                                                        "forceInput": True}),
                "seed":         ("INT",   {"default": 0,
                                           "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps":        ("INT",   {"default": 20, "min": 1, "max": 200}),
                "cfg":          ("FLOAT", {"default": 5.0, "min": 0.0,
                                           "max": 30.0, "step": 0.1}),
                "sampler_name": (_SAMPLERS,),
                "scheduler":    (_SCHEDULERS,),
                "denoise":      ("FLOAT", {"default": 0.5, "min": 0.0,
                                           "max": 1.0, "step": 0.01}),
                "feather":       ("INT",  {"default": 0, "min": 0, "max": 64,
                                           "tooltip": "Feather (soft edge) radius in pixels. "
                                                      "Blends mask edges smoothly. Applied after dilation."}),
                "noise_mask":    ("BOOLEAN", {"default": True,
                                              "tooltip": "Apply noise mask to latent before sampling. "
                                                         "Helps preserve areas outside the inpainting region."}),
                "force_inpaint": ("BOOLEAN", {"default": True,
                                              "tooltip": "Force inpainting mode regardless of model type. "
                                                         "Recommended for best inpainting results."}),
                "bbox_model":   (bbox_models,),
                "detect_threshold":  ("FLOAT", {"default": 0.3, "min": 0.1,
                                           "max": 1.0, "step": 0.05,
                                           "tooltip": "YOLO detection confidence threshold."}),
                "drop_size": ("INT", {"default": 10, "min": 1, "max": 16384,
                                      "step": 1,
                                      "tooltip": "Minimum detection size (px). Bboxes with short side below this are ignored."}),
                "mask_padding": ("INT",   {"default": 32, "min": 0, "max": 256,
                                           "tooltip": "Mask padding in pixels (image space)."}),
                "mask_blur":    ("INT",   {"default": 8,  "min": 0, "max": 64,
                                           "tooltip": "Blur radius for mask edges."}),
                "mask_dilation": ("INT",  {"default": 4, "min": 0, "max": 64,
                                           "tooltip": "Mask dilation in pixels. "
                                                      "Expands the mask boundary to widen the inpainting area."}),
                "scale_to_pixel": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32,
                                           "tooltip": "Target pixel size for upscaling. "
                                                      "Masks smaller than this are upscaled before inpainting. "
                                                      "Masks larger than this are processed as-is."}),

                "use_base":     ("BOOLEAN", {"default": False,
                                             "tooltip": "Prepend BASE prompt to each COL prompt during processing."}),
                "use_common":   ("BOOLEAN", {"default": True}),

                "debug":           ("BOOLEAN", {"default": False,
                                   "tooltip": "Print debug log when enabled."}),
            },
            "optional": {                "divide_ratio": ("RP_DIV_RATIO",  {"default": "1,1",
                                             "forceInput": True,
                                             "tooltip": "Connect divide_ratio from RPPromptParser."}),            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("image", "debug_image")
    FUNCTION      = "execute"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def execute(self, image, model, clip, vae, regional_prompts_nolora,
                regional_lora_map, regional_col_n_row, negative,
                divide_mode, seed, steps, cfg, sampler_name, scheduler,
                denoise, feather=0, noise_mask=True, force_inpaint=True,
                bbox_model=None, detect_threshold=0.3, drop_size=10,
                mask_padding=32, mask_blur=8, mask_dilation=4,
                scale_to_pixel=1024,
                use_base=False, use_common=True, divide_ratio="1,1", debug=False):

        _dbg = print if debug else lambda *a, **kw: None
        import torch
        import comfy.sample
        import comfy.model_management as mm
        import numpy as np

        if not _COMFY_OK:
            raise RuntimeError("comfy module not found.")

        # ── 1. Input image → numpy, PIL ───────────────────────
        # image: [B, H, W, C] float32 0~1
        img_tensor = image[0]                              # [H, W, C]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)  # [H,W,C] uint8
        img_h, img_w = img_np.shape[:2]

        print(f"\n[RPRegionalDetailer] start  image={img_w}×{img_h}  model={bbox_model}")

        # ── 2. YOLO detection ───────────────────────────────────
        try:
            from ultralytics import YOLO
            import folder_paths as fp
            import torch

            # Search model path
            prefix, fname = (bbox_model.split("/", 1) + [""])[:2]
            if not fname:
                fname = prefix; prefix = "bbox"
            search_dirs = [
                os.path.join(fp.models_dir, "ultralytics", "bbox"),
                os.path.join(fp.models_dir, "ultralytics", "segm"),
                os.path.join(fp.models_dir, "ultralytics"),
            ]
            model_path = None
            for d in search_dirs:
                p = os.path.join(d, fname)
                if os.path.isfile(p):
                    model_path = p
                    break
            if model_path is None:
                raise FileNotFoundError(f"Model file '{fname}' not found. "
                                        f"Search paths: {search_dirs}")

            # Fully bypass torch_wrapper from impact-subpack
            # Temporarily restore original torch.load and torch.serialization._load
            import torch
            import torch.serialization as _ts

            _patched_load = torch.load

            def _bypass_load(f, map_location=None, **kwargs):
                # Force weights_only=False to allow ultralytics model loading
                kwargs.pop('weights_only', None)
                return _TORCH_LOAD_ORIG(f, map_location=map_location,
                                        weights_only=False, **kwargs)

            torch.load = _bypass_load
            try:
                yolo = YOLO(model_path)
            finally:
                torch.load = _patched_load  # restore

            _dbg(f"  [YOLO] model loaded: {model_path}")

        except ImportError:
            raise RuntimeError("ultralytics package required. "
                               "`pip install ultralytics`")

        # ── 3. Per-area detection → largest person bbox ─────────
        from core.regions import make_filters

        region_rows = regional_col_n_row
        latent_h    = img_h // 8
        latent_w    = img_w // 8
        filters     = make_filters(
            region_rows=region_rows, h=latent_h, w=latent_w,
            mode=divide_mode, usebase=use_base, device="cpu",
        )

        # Extract prompts data
        if isinstance(regional_prompts_nolora, dict):
            nolora_list  = regional_prompts_nolora["nolora"]
            common_text  = regional_prompts_nolora.get("common", "")
            col_texts    = regional_prompts_nolora.get("col_texts", [])
            _use_com     = regional_prompts_nolora.get("use_common", use_common)
        else:
            nolora_list  = [regional_prompts_nolora] if isinstance(regional_prompts_nolora, str) else list(regional_prompts_nolora)
            common_text  = ""
            col_texts    = nolora_list
            _use_com     = False

        col_lora_map = regional_lora_map if regional_lora_map else {}

        # YOLO full-image inference (once)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_np)
        yolo_results = yolo(pil_img, conf=detect_threshold, classes=[0], verbose=False)
        # classes=[0] → COCO person class

        # Collect all bbox + scores (xyxy, conf)
        all_boxes = []  # [(x1,y1,x2,y2,conf,area)]
        if yolo_results and len(yolo_results[0].boxes) > 0:
            boxes_data = yolo_results[0].boxes
            for i in range(len(boxes_data)):
                x1, y1, x2, y2 = boxes_data.xyxy[i].cpu().numpy()
                conf = float(boxes_data.conf[i].cpu())
                area = (x2 - x1) * (y2 - y1)
                all_boxes.append((float(x1), float(y1), float(x2), float(y2), conf, area))

        _dbg(f"  [YOLO] Total persons detected: {len(all_boxes)}")

        # ── 4. ZImage-style: assign bbox to region by center coordinate ──
        # Single full-image YOLO pass → bbox center → divide_ratio boundary comparison
        n_regions = len(filters) - (1 if use_base else 0)
        region_detections = {}

        def _build_boundaries(ratio_str, is_vertical, w, h):
            """Build pixel boundary lists from divide_ratio string."""
            if not ratio_str:
                return None
            try:
                segs = ratio_str.split(";")
                seg_vals = []
                for s in segs:
                    sub = [max(0.0, float(v.strip())) for v in s.split(",") if v.strip()]
                    if sub:
                        seg_vals.append(sub)
            except Exception:
                return None
            if not seg_vals:
                return None
            def to_px(vals, size):
                total = sum(vals) or 1
                bounds = [0]
                acc = 0.0
                for v in vals:
                    acc += v / total
                    bounds.append(round(acc * size))
                return bounds
            if not is_vertical:
                row_h_vals = [sum(s) for s in seg_vals]
                y_bounds = to_px(row_h_vals, h)
                x_bounds_per_row = [to_px(s, w) for s in seg_vals]
                return ("H", y_bounds, x_bounds_per_row)
            else:
                col_w_vals = [sum(s) for s in seg_vals]
                x_bounds = to_px(col_w_vals, w)
                y_bounds_per_col = [to_px(s, h) for s in seg_vals]
                return ("V", x_bounds, y_bounds_per_col)

        def _bbox_to_col_i(x1, y1, x2, y2, bounds, is_vertical, n_regions):
            """Return col_i (0-based region index) for a bbox by center coordinate."""
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if bounds is None:
                # No ratio info: equal split by center
                if not is_vertical:
                    return min(int(cx / img_w * n_regions), n_regions - 1)
                else:
                    return min(int(cy / img_h * n_regions), n_regions - 1)
            mode = bounds[0]
            if mode == "H":
                y_bounds, x_bounds_per_row = bounds[1], bounds[2]
                row_n = len(y_bounds) - 2
                for ri in range(len(y_bounds) - 1):
                    if y_bounds[ri] <= cy < y_bounds[ri + 1]:
                        row_n = ri; break
                xb = x_bounds_per_row[row_n] if row_n < len(x_bounds_per_row) else [0, img_w]
                col_n = len(xb) - 2
                for ci in range(len(xb) - 1):
                    if xb[ci] <= cx < xb[ci + 1]:
                        col_n = ci; break
                # Flatten (row_n, col_n) → linear col_i
                col_i = 0
                for ri in range(row_n):
                    rb = x_bounds_per_row[ri] if ri < len(x_bounds_per_row) else [0, img_w]
                    col_i += max(1, len(rb) - 1)
                col_i += col_n
                return min(col_i, n_regions - 1)
            else:  # V
                x_bounds, y_bounds_per_col = bounds[1], bounds[2]
                col_n = len(x_bounds) - 2
                for ci in range(len(x_bounds) - 1):
                    if x_bounds[ci] <= cx < x_bounds[ci + 1]:
                        col_n = ci; break
                yb = y_bounds_per_col[col_n] if col_n < len(y_bounds_per_col) else [0, img_h]
                row_n = len(yb) - 2
                for ri in range(len(yb) - 1):
                    if yb[ri] <= cy < yb[ri + 1]:
                        row_n = ri; break
                # Flatten (col_n, row_n) → linear col_i
                col_i = 0
                for ci in range(col_n):
                    yb2 = y_bounds_per_col[ci] if ci < len(y_bounds_per_col) else [0, img_h]
                    col_i += max(1, len(yb2) - 1)
                col_i += row_n
                return min(col_i, n_regions - 1)

        _is_vertical = "Ver" in (divide_mode or "Horizontal")
        _ratio_str   = (divide_ratio or "").strip()
        _bounds      = _build_boundaries(_ratio_str, _is_vertical, img_w, img_h)
        _dbg(f"  [assign] divide_mode={divide_mode}  ratio='{_ratio_str}'  "
             f"n_regions={n_regions}")

        # Assign each detected bbox to a region by center coordinate
        # If multiple persons in same region → keep largest area, rest → fallback
        fallback_boxes = []  # persons not selected for any COL → base prompt
        displaced = {}       # region → previously assigned (smaller) bbox
        for bx1, by1, bx2, by2, bconf, barea in all_boxes:
            bw, bh = bx2 - bx1, by2 - by1
            if min(bw, bh) < drop_size:
                continue
            col_i = _bbox_to_col_i(bx1, by1, bx2, by2, _bounds, _is_vertical, n_regions)
            if col_i not in region_detections:
                region_detections[col_i] = (bx1, by1, bx2, by2, bconf, barea)
                _dbg(f"  [region{col_i}] assigned "
                     f"bbox=({bx1:.0f},{by1:.0f},{bx2:.0f},{by2:.0f})"
                     f"  conf={bconf:.2f}  area={barea:.0f}px²")
            elif barea > region_detections[col_i][5]:
                # New bbox is larger → displace current to fallback
                old_box = region_detections[col_i]
                fallback_boxes.append(old_box[:5])
                _dbg(f"  [region{col_i}] replaced by larger bbox → old to fallback")
                region_detections[col_i] = (bx1, by1, bx2, by2, bconf, barea)
            else:
                # Current bbox is smaller → goes to fallback
                fallback_boxes.append((bx1, by1, bx2, by2, bconf))
                _dbg(f"  [fallback] bbox=({bx1:.0f},{by1:.0f},{bx2:.0f},{by2:.0f})"
                     f"  conf={bconf:.2f} → base prompt")

        _dbg(f"  [region_detections] {len(region_detections)} regions assigned, "
             f"{len(fallback_boxes)} fallback persons")

        # Strip area from stored tuple to match downstream format (x1,y1,x2,y2,conf)
        region_detections = {
            k: v[:5] for k, v in region_detections.items()
        }

        # ── 5. Build debug image (bbox visualization) ──────────────
        debug_np = img_np.copy()
        colors = [(255,80,80),(80,255,80),(80,80,255),(255,255,80),(255,80,255),(80,255,255)]
        for col_i, (x1,y1,x2,y2,conf) in region_detections.items():
            c = colors[col_i % len(colors)]
            ix1,iy1,ix2,iy2 = int(x1),int(y1),int(x2),int(y2)
            for t in range(3):
                debug_np[max(0,iy1-t):iy1+t+1, ix1:ix2] = c
                debug_np[iy2-t:min(img_h,iy2+t+1), ix1:ix2] = c
                debug_np[iy1:iy2, max(0,ix1-t):ix1+t+1] = c
                debug_np[iy1:iy2, ix2-t:min(img_w,ix2+t+1)] = c
        debug_tensor = torch.from_numpy(debug_np.astype(np.float32) / 255.0).unsqueeze(0)

        if not region_detections:
            _dbg("  [RPRegionalDetailer] no persons detected → return original")
            return (image, debug_tensor)

        # ── 6. Per-area inpainting ──────────────────────────────
        result_img = img_tensor.clone()  # [H,W,C]

        device = mm.get_torch_device()

        for col_i, (x1, y1, x2, y2, conf) in region_detections.items():
            # Apply padding
            px1 = max(0, int(x1) - mask_padding)
            py1 = max(0, int(y1) - mask_padding)
            px2 = min(img_w, int(x2) + mask_padding)
            py2 = min(img_h, int(y2) + mask_padding)

            # Align crop coordinates to multiples of 8
            # → Ensures VAE encode/decode sizes match exactly
            px1 = (px1 // 8) * 8
            py1 = (py1 // 8) * 8
            px2 = min(img_w, ((px2 + 7) // 8) * 8)
            py2 = min(img_h, ((py2 + 7) // 8) * 8)

            crop_w = px2 - px1
            crop_h = py2 - py1
            if crop_w <= 0 or crop_h <= 0:
                continue

            # Build mask (image size)
            mask_np = np.zeros((img_h, img_w), dtype=np.float32)
            mask_np[py1:py2, px1:px2] = 1.0

            # Apply in order: dilation → blur → feather
            try:
                import cv2
                if mask_dilation > 0:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (mask_dilation * 2 + 1, mask_dilation * 2 + 1)
                    )
                    mask_np = cv2.dilate(mask_np, kernel)
                if mask_blur > 0:
                    ksize = mask_blur * 2 + 1
                    mask_np = cv2.GaussianBlur(mask_np, (ksize, ksize), mask_blur)
                if feather > 0:
                    ksize = feather * 2 + 1
                    mask_np = cv2.GaussianBlur(mask_np, (ksize, ksize), feather)
                    mask_np = np.clip(mask_np, 0.0, 1.0)
            except ImportError:
                pass  # skip dilation/blur/feather if cv2 not available

            # ── crop → scale_to_pixel upscale → inpaint → downscale ──
            # mask < scale_to_pixel: upscale long edge → inpaint → restore
            # mask >= scale_to_pixel: skip upscale → inpaint at original size
            crop_np = img_np[py1:py2, px1:px2]          # [crop_h, crop_w, 3]
            orig_crop_h, orig_crop_w = crop_np.shape[:2]
            max_side = max(orig_crop_w, orig_crop_h)

            if max_side < scale_to_pixel:
                # upscale: scale so long edge equals scale_to_pixel
                scale_ratio = scale_to_pixel / max_side
                up_w = (int(orig_crop_w * scale_ratio) // 8) * 8
                up_h = (int(orig_crop_h * scale_ratio) // 8) * 8
                up_w = max(64, min(4096, up_w))
                up_h = max(64, min(4096, up_h))
                print(f"  [area{col_i}] crop={orig_crop_w}×{orig_crop_h} → up={up_w}×{up_h} (target={scale_to_pixel}px)")
            else:
                # skip upscale: keep original size (align to 8)
                up_w = (orig_crop_w // 8) * 8
                up_h = (orig_crop_h // 8) * 8
                up_w = max(64, up_w)
                up_h = max(64, up_h)
                print(f"  [area{col_i}] crop={orig_crop_w}×{orig_crop_h} → upscale skip (max_side={max_side}>={scale_to_pixel}px)")

            # Upscale crop image (LANCZOS)
            from PIL import Image as _PILUp
            crop_pil = _PILUp.fromarray(crop_np)
            crop_up  = crop_pil.resize((up_w, up_h), _PILUp.LANCZOS)

            # Upscale mask together
            mask_crop_np = (mask_np[py1:py2, px1:px2] * 255).astype(np.uint8)
            mask_pil_up  = _PILUp.fromarray(mask_crop_np).resize((up_w, up_h), _PILUp.LANCZOS)

            # Upscaled crop → tensor → VAE encode
            crop_t = torch.from_numpy(
                np.array(crop_up).astype(np.float32) / 255.0
            ).unsqueeze(0).to(device)                    # [1,up_h,up_w,3]

            latent_crop = vae.encode(crop_t)
            if isinstance(latent_crop, dict):
                latent_crop = latent_crop["samples"]

            # Select prompt: use_base=True → nolora_list[0]=BASE, DIV=[1:]
            prompt_idx  = col_i + (1 if use_base else 0)
            full_prompt = nolora_list[prompt_idx] if prompt_idx < len(nolora_list) else ""
            col_text    = col_texts[prompt_idx]   if prompt_idx < len(col_texts)   else ""

            # BASE text: col_texts[0] = _base_part (between ADDCOMM and ADDBASE)
            # use_base=True: prepend BASE text when encoding each DIV
            base_text = col_texts[0] if (use_base and col_texts) else ""

            # CLIP encode
            # use_common: combine COMMON + (BASE +) DIV
            # use_base only: combine BASE + DIV
            # otherwise: use full_prompt as-is
            _use_com_final = bool(common_text and _use_com)
            if _use_com_final and use_base and base_text and col_text:
                # COMMON + BASE + DIV
                tok_com  = clip.tokenize(common_text)
                tok_base = clip.tokenize(base_text)
                tok_col  = clip.tokenize(col_text)
                c_com,  _       = _enc(clip, tok_com) 
                c_base, _       = _enc(clip, tok_base)
                c_col,  pooled  = _enc(clip, tok_col) 
                cond = torch.cat([c_com, c_base, c_col], dim=1)
            elif _use_com_final and col_text:
                # COMMON + DIV
                tok_com = clip.tokenize(common_text)
                tok_col = clip.tokenize(col_text)
                c_com, _       = _enc(clip, tok_com)
                c_col, pooled  = _enc(clip, tok_col)
                cond = torch.cat([c_com, c_col], dim=1)
            elif use_base and base_text and col_text:
                # BASE + DIV
                tok_base = clip.tokenize(base_text)
                tok_col  = clip.tokenize(col_text)
                c_base, _       = _enc(clip, tok_base)
                c_col,  pooled  = _enc(clip, tok_col) 
                cond = torch.cat([c_base, c_col], dim=1)
            else:
                tok = clip.tokenize(col_text if col_text else full_prompt)
                cond, pooled = _enc(clip, tok)

            positive_cond = [[cond, {"pooled_output": pooled}]]

            # Apply LoRA for this area
            sample_model = model
            loras_for_col = col_lora_map.get(prompt_idx, {})
            if loras_for_col and _COMFY_OK:
                from core.lora_manager import _apply_loras
                sample_model, _ = _apply_loras(model, clip, loras_for_col)
                _dbg(f"  [area{col_i}] LoRA applied: {loras_for_col}")

            # noise + sampling
            noise = comfy.sample.prepare_noise(latent_crop, seed + col_i, None)

            # Build noise_mask tensor (resample to upscaled latent resolution)
            _noise_mask_t = None
            if noise_mask:
                import torch.nn.functional as _F
                # Resample upscaled mask (mask_pil_up) to latent size
                lh, lw = latent_crop.shape[2], latent_crop.shape[3]
                mask_up_np = np.array(mask_pil_up).astype(np.float32) / 255.0
                mask_up_t  = torch.from_numpy(mask_up_np).unsqueeze(0).unsqueeze(0)
                _noise_mask_t = _F.interpolate(
                    mask_up_t.float(), size=(lh, lw), mode="bilinear",
                    align_corners=False
                ).squeeze(0).to(latent_crop.device)   # [1, lh, lw]

            try:
                sample_model.unpatch_model()
            except Exception:
                pass
            inpaint_output = comfy.sample.sample(
                model        = sample_model,
                noise        = noise,
                steps        = steps,
                cfg          = cfg,
                sampler_name = sampler_name,
                scheduler    = scheduler,
                positive     = positive_cond,
                negative     = negative,
                latent_image = latent_crop,
                denoise      = denoise,
                seed         = seed + col_i,
                noise_mask   = _noise_mask_t,
            )

            if isinstance(inpaint_output, dict):
                inpaint_lat = inpaint_output["samples"]
            else:
                inpaint_lat = inpaint_output

            # VAE decode
            inpaint_img = vae.decode(inpaint_lat)
            inpaint_np  = (inpaint_img[0].cpu().numpy() * 255).astype(np.uint8)

            # Fix decode size if it differs from up_w/h
            if inpaint_np.shape[:2] != (up_h, up_w):
                inpaint_np = np.array(
                    _PILUp.fromarray(inpaint_np).resize((up_w, up_h), _PILUp.LANCZOS)
                )

            # RPH method: downscale → restore to original crop size
            inpaint_down = np.array(
                _PILUp.fromarray(inpaint_np).resize(
                    (orig_crop_w, orig_crop_h), _PILUp.LANCZOS)
            )

            # Downscale upscaled mask for blending
            mask_blend = np.array(
                mask_pil_up.resize((orig_crop_w, orig_crop_h), _PILUp.LANCZOS)
            ).astype(np.float32) / 255.0                 # [ch, cw] 0~1
            mask_blend = mask_blend[:, :, np.newaxis]    # [ch, cw, 1]

            original_crop = (result_img[py1:py2, px1:px2].cpu().numpy() * 255).astype(np.uint8)
            blended = (inpaint_down * mask_blend + original_crop * (1 - mask_blend)).astype(np.uint8)
            result_img[py1:py2, px1:px2] = torch.from_numpy(
                blended.astype(np.float32) / 255.0
            )

            _dbg(f"  [area{col_i}] inpainting done  "
                  f"crop={orig_crop_w}×{orig_crop_h} → up={up_w}×{up_h}"
                  f"({'skip' if max_side >= scale_to_pixel else f'→{scale_to_pixel}px'})")

            # Release sample_model immediately (cloned per loop → prevent leak)
            try:
                if sample_model is not model:
                    sample_model.patches = {}
                    sample_model.object_patches = {}
                    sample_model.object_patches_backup = {}
                    del sample_model
            except Exception:
                pass
            # Release conditioning/inpaint tensor references
            try:
                del positive_cond, inpaint_lat, inpaint_img
            except Exception:
                pass

        result_tensor = result_img.unsqueeze(0)  # [1,H,W,C]


        # ── Fallback: base prompt inpainting for displaced persons ──────────
        # Persons not assigned to any COL region → inpaint with base prompt
        if fallback_boxes:
            print(f"[RPRegionalDetailer] fallback: {len(fallback_boxes)} persons → base prompt")

            # Build base conditioning: common + base_text (same as col loop)
            _fb_text = ""
            _use_com_fb = bool(common_text and _use_com)
            if _use_com_fb:
                _fb_text += common_text.strip()
            _base_t = col_texts[0] if (use_base and col_texts) else ""
            if _base_t:
                _fb_text = (_fb_text + ", " + _base_t).strip(", ")
            if not _fb_text:
                _fb_text = col_texts[0] if col_texts else (nolora_list[0] if nolora_list else "")
            print(f"[RPRegionalDetailer] fallback base prompt: {_fb_text[:100]}")

            if _fb_text:
                tok_fb     = clip.tokenize(_fb_text)
                cond_fb, pooled_fb = _enc(clip, tok_fb)
                pos_fb     = [[cond_fb, {"pooled_output": pooled_fb}]]

                from PIL import Image as _PILFb
                import torch.nn.functional as _FFb

                for _fb_i, (_fx1, _fy1, _fx2, _fy2, _fconf) in enumerate(fallback_boxes):
                    # Apply padding (same as col loop)
                    fpx1 = max(0, int(_fx1) - mask_padding)
                    fpy1 = max(0, int(_fy1) - mask_padding)
                    fpx2 = min(img_w, int(_fx2) + mask_padding)
                    fpy2 = min(img_h, int(_fy2) + mask_padding)
                    fcw, fch = fpx2 - fpx1, fpy2 - fpy1
                    if fcw < 8 or fch < 8:
                        continue

                    try:
                        # Dilation mask (full bbox area)
                        _fb_mask_np = np.ones((fch, fcw), dtype=np.float32)

                        # Apply feather via blur
                        if feather > 0:
                            try:
                                import cv2 as _cv2fb
                                _fb_mask_np = _cv2fb.GaussianBlur(
                                    _fb_mask_np, (0, 0), feather)
                            except Exception:
                                pass

                        # Crop from current result
                        _fb_crop_np = (result_img[fpy1:fpy2, fpx1:fpx2].cpu().numpy() * 255).astype(np.uint8)

                        # Upscale if needed (same logic as col loop)
                        _fb_max = max(fcw, fch)
                        if _fb_max < scale_to_pixel:
                            _sc = scale_to_pixel / _fb_max
                            _fup_w = max(64, round(fcw * _sc) // 8 * 8)
                            _fup_h = max(64, round(fch * _sc) // 8 * 8)
                        else:
                            _fup_w = max(64, fcw // 8 * 8)
                            _fup_h = max(64, fch // 8 * 8)

                        _fb_crop_pil = _PILFb.fromarray(_fb_crop_np)
                        _fb_crop_up  = _fb_crop_pil.resize((_fup_w, _fup_h), _PILFb.LANCZOS)
                        _fb_mask_pil = _PILFb.fromarray((_fb_mask_np * 255).astype(np.uint8))
                        _fb_mask_up  = _fb_mask_pil.resize((_fup_w, _fup_h), _PILFb.LANCZOS)

                        # VAE encode
                        _fb_crop_t = torch.from_numpy(
                            np.array(_fb_crop_up).astype(np.float32) / 255.0
                        ).unsqueeze(0).to(device)
                        _fb_lat = vae.encode(_fb_crop_t)
                        if isinstance(_fb_lat, dict):
                            _fb_lat = _fb_lat["samples"]

                        # Noise mask
                        _fb_noise_mask = None
                        if noise_mask:
                            _lh, _lw = _fb_lat.shape[2], _fb_lat.shape[3]
                            _mn = np.array(_fb_mask_up).astype(np.float32) / 255.0
                            _mt = torch.from_numpy(_mn).unsqueeze(0).unsqueeze(0)
                            _fb_noise_mask = _FFb.interpolate(
                                _mt.float(), size=(_lh, _lw), mode="bilinear",
                                align_corners=False
                            ).squeeze(0).to(_fb_lat.device)

                        # Sample
                        _fb_noise_t = comfy.sample.prepare_noise(_fb_lat, seed + 1000 + _fb_i, None)
                        try:
                            model.unpatch_model()
                        except Exception:
                            pass
                        _fb_out = comfy.sample.sample(
                            model        = model,
                            noise        = _fb_noise_t,
                            steps        = steps,
                            cfg          = cfg,
                            sampler_name = sampler_name,
                            scheduler    = scheduler,
                            positive     = pos_fb,
                            negative     = negative,
                            latent_image = _fb_lat,
                            denoise      = denoise,
                            seed         = seed + 1000 + _fb_i,
                            noise_mask   = _fb_noise_mask,
                        )
                        _fb_lat_out = _fb_out["samples"] if isinstance(_fb_out, dict) else _fb_out

                        # VAE decode
                        _fb_dec = vae.decode(_fb_lat_out)
                        _fb_dec_np = (_fb_dec[0].cpu().numpy() * 255).astype(np.uint8)

                        # Resize back to original crop size
                        if _fb_dec_np.shape[:2] != (fch, fcw):
                            _fb_dec_np = np.array(
                                _PILFb.fromarray(_fb_dec_np).resize((fcw, fch), _PILFb.LANCZOS)
                            )

                        # Blend with mask
                        _fb_mask_blend = np.array(
                            _fb_mask_up.resize((fcw, fch), _PILFb.LANCZOS)
                        ).astype(np.float32) / 255.0
                        _fb_mask_blend = _fb_mask_blend[:, :, np.newaxis]
                        _fb_orig = (result_img[fpy1:fpy2, fpx1:fpx2].cpu().numpy() * 255).astype(np.uint8)
                        _fb_blended = (_fb_dec_np * _fb_mask_blend + _fb_orig * (1 - _fb_mask_blend)).astype(np.uint8)
                        result_img[fpy1:fpy2, fpx1:fpx2] = torch.from_numpy(
                            _fb_blended.astype(np.float32) / 255.0
                        )
                        print(f"[RPRegionalDetailer] fallback{_fb_i} inpainted ✓  "
                              f"bbox=({int(_fx1)},{int(_fy1)},{int(_fx2)},{int(_fy2)})")

                    except Exception as _fe:
                        import traceback
                        print(f"[RPRegionalDetailer] fallback{_fb_i} failed: {_fe}")
                        traceback.print_exc()

        # Update debug image with latest result
        debug_final = debug_tensor  # keep bbox visualization

        import gc; gc.collect()
        try:
            import comfy.model_management as _cmm
            _cmm.soft_empty_cache()
        except Exception:
            pass
        print(f"[RPRegionalDetailer] done")
        return (result_tensor, debug_final)
