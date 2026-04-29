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

# 4b. RPRegionalDetailerZImage
#     - Z-Image(Lumina2/NextDiT) dedicated Regional Detailer
#     - YOLO area detection → crop → 16ch zeros latent → AuraFlow sampling
#     - Detail Daemon sigma manipulation integrated
#     - Pass Lumina2 conditioning via cross_attn key
#     - Completely separate from RPRegionalDetailer
# ══════════════════════════════════════════════════════
class RPRegionalDetailerZImage:
    CATEGORY = "Regional Prompter"
    cnr_id  = "ComfyUI_RP_Cast"
    _debug_mode = False  # set from debug param in execute, referenced by WD14 methods

    @classmethod
    def INPUT_TYPES(cls):
        bbox_models = []
        try:
            import folder_paths
            for sub in ["bbox", "segm"]:
                d = os.path.join(folder_paths.models_dir, "ultralytics", sub)
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.endswith(".pt"):
                            bbox_models.append(f"{sub}/{f}")
        except Exception:
            pass
        if not bbox_models:
            bbox_models = ["bbox/person_yolov8m-seg.pt", "segm/person_yolov8m-seg.pt"]

        return {
            "required": {
                "image":                   ("IMAGE",),
                "model":                   ("MODEL",),
                "clip":                    ("CLIP",),
                "vae":                     ("VAE",),
                "regional_prompts_nolora": ("RP_SUBPROMPTS",),
                "regional_lora_map":       ("RP_LORA_MAP",),
                "negative":                ("CONDITIONING",),
                "seed":    ("INT",   {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps":   ("INT",   {"default": 8, "min": 1, "max": 50,
                                      "tooltip": "Recommended for Z-Image Turbo: 6~10."}),
                "cfg":     ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                      "tooltip": "Recommended for Z-Image Turbo: 1.0."}),
                "sampler_name": (_SAMPLERS,),
                "scheduler":    (_SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "img2img denoise strength. 0.35~0.55 recommended."}),
                "feather":           ("INT",   {"default": 0,  "min": 0, "max": 64,
                                                "tooltip": "Feather (soft edge) radius in pixels. "
                                                           "Blends mask edges smoothly. Applied after dilation."}),
                "noise_mask":        ("BOOLEAN", {"default": True,
                                                  "tooltip": "Apply noise mask to latent before sampling. "
                                                             "Helps preserve areas outside the inpainting region."}),
                "force_inpaint":     ("BOOLEAN", {"default": True,
                                                  "tooltip": "Force inpainting mode regardless of model type. "
                                                             "Recommended for best inpainting results."}),
                "shift":   ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5,
                                      "tooltip": "AuraFlow sigma shift. Z-Image Turbo recommended: 3~6."}),
                "bbox_model":        (bbox_models,),
                "detect_threshold":  ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0,
                                                "step": 0.05}),
                "drop_size":         ("INT",   {"default": 10, "min": 1, "max": 16384,
                                                "step": 1,
                                                "tooltip": "Minimum detection size (px). "
                                                           "Bboxes with short side (min(w,h)) below this are ignored. "
                                                           "Same criterion as Impact Pack FaceDetailer."}),
                "mask_padding":      ("INT",   {"default": 32, "min": 0, "max": 256}),
                "mask_blur":         ("INT",   {"default": 8,  "min": 0, "max": 64}),
                "mask_dilation":     ("INT",   {"default": 4,  "min": 0, "max": 64}),
                "scale_to_pixel":    ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32,
                                     "tooltip": "Target pixel size for upscaling. "
                                                "Masks smaller than this are upscaled before inpainting. "
                                                "Masks larger than this are processed as-is."}),

                "use_base":     ("BOOLEAN", {"default": False,
                                             "tooltip": "Prepend BASE prompt to each COL prompt during processing."}),
                "use_common":   ("BOOLEAN", {"default": True}),

                "debug":           ("BOOLEAN", {"default": False,
                                   "tooltip": "Print debug log when enabled."}),
            },
            "optional": {                "divide_mode":  ("RP_DIV_MODE",  {"default": "Horizontal",
                                             "forceInput": True,
                                             "tooltip": "Connect divide_mode from RPPromptParser."}),
                "divide_ratio": ("RP_DIV_RATIO",  {"default": "",
                                             "forceInput": True,
                                             "tooltip": "Connect divide_ratio from RPPromptParser."}),            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("image", "debug_image")
    FUNCTION      = "execute"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    # ── Internal helper: load YOLO ───────────────────────────
    @staticmethod
    def _load_yolo(bbox_model):
        from ultralytics import YOLO
        import folder_paths as fp
        prefix, fname = (bbox_model.split("/", 1) + [""])[:2]
        if not fname:
            fname = prefix
        for sub in ["bbox", "segm", ""]:
            p = os.path.join(fp.models_dir, "ultralytics", sub, fname) if sub \
                else os.path.join(fp.models_dir, "ultralytics", fname)
            if os.path.isfile(p):
                import torch
                _orig = torch.load
                def _bypass(f, map_location=None, **kw):
                    kw.pop('weights_only', None)
                    return _TORCH_LOAD_ORIG(f, map_location=map_location,
                                            weights_only=False, **kw)
                torch.load = _bypass
                try:
                    return YOLO(p)
                finally:
                    torch.load = _orig
        raise FileNotFoundError(f"YOLO model '{fname}' not found.")

    # ── Internal helper: detect all persons in full image ────
    @staticmethod
    def _detect_persons(yolo, pil_img, img_h, img_w,
                        detect_threshold, drop_size=10):
        """Detect all persons in the full image.
        drop_size: exclude bbox if min(w,h) < drop_size (same as Impact FaceDetailer)
        Returns: list of (x1,y1,x2,y2,conf,area) sorted left→right
        """
        res = yolo(pil_img, conf=detect_threshold, classes=[0], verbose=False)
        persons = []
        if res and len(res[0].boxes) > 0:
            for i in range(len(res[0].boxes)):
                x1,y1,x2,y2 = res[0].boxes.xyxy[i].cpu().numpy()
                conf = float(res[0].boxes.conf[i].cpu())
                w, h = x2-x1, y2-y1
                area = w * h
                # apply drop_size filter only
                if min(w, h) < drop_size:
                    continue
                persons.append((float(x1),float(y1),float(x2),float(y2),conf,area))

        # Sort left→right (by x1)
        persons.sort(key=lambda p: p[0])
        return persons

    # ── WD14 Tagger gender classification (Danbooru tag) ──────
    # Model cache (class variable)
    _wd14_session  = None
    _wd14_tags     = None
    _wd14_boy_idx  = None
    _wd14_girl_idx = None

    @classmethod
    def _load_wd14(cls):
        """Load WD14 tagger ONNX session (once, cached thereafter).
        Model: SmilingWolf/wd-v1-4-swinv2-tagger-v2
        Uses ComfyUI-WD14-Tagger cache if available, else downloads from HuggingFace.
        """
        if cls._wd14_session is not None:
            return

        import csv
        import numpy as np
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "onnxruntime required: pip install onnxruntime"
            )

        # WD14 model storage / search paths:
        # 1. custom_nodes/ComfyUI_RP_Cast/data/wd14/  ← default cache
        # 2. ComfyUI-WD14-Tagger cache
        # 3. folder_paths.models_dir/wd14
        # 4. Auto-download from HuggingFace → save to data/wd14/
        model_name = "wd-v1-4-swinv2-tagger-v2"
        onnx_name  = "model.onnx"
        csv_name   = "selected_tags.csv"
        onnx_path, csv_path = None, None

        # data/ directory: relative to __init__.py
        _this_dir  = os.path.dirname(os.path.abspath(__file__))
        _data_dir  = os.path.join(_this_dir, "data", "wd14")
        os.makedirs(_data_dir, exist_ok=True)

        search_dirs = [_data_dir]  # search data/wd14/ first
        try:
            import folder_paths as _fp
            search_dirs += [
                os.path.join(_fp.base_path, "custom_nodes", "ComfyUI-WD14-Tagger", "models"),
                os.path.join(_fp.models_dir, "wd14"),
                os.path.join(_fp.models_dir, "tagger"),
            ]
        except Exception:
            pass

        for d in search_dirs:
            op = os.path.join(d, onnx_name)
            cp = os.path.join(d, csv_name)
            if os.path.isfile(op) and os.path.isfile(cp):
                onnx_path, csv_path = op, cp
                (print if RPRegionalDetailerZImage._debug_mode else lambda *a,**k: None)(
                    f"  [WD14] local cache found: {d}")
                break

        if onnx_path is None:
            # Auto-download from HuggingFace → save to data/wd14/
            try:
                from huggingface_hub import hf_hub_download
                repo_id = f"SmilingWolf/{model_name}"
                onnx_path = hf_hub_download(repo_id=repo_id,
                                             filename="model.onnx",
                                             local_dir=_data_dir)
                csv_path  = hf_hub_download(repo_id=repo_id,
                                             filename="selected_tags.csv",
                                             local_dir=_data_dir)
                (print if RPRegionalDetailerZImage._debug_mode else lambda *a,**k: None)(
                    f"  [WD14] download complete: {_data_dir}")
            except ImportError:
                raise RuntimeError(
                    "huggingface_hub required: pip install huggingface_hub\n"
                    "or install ComfyUI-WD14-Tagger."
                )

        # Create ONNX session
        providers = ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort2
            if "CUDAExecutionProvider" in ort2.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        cls._wd14_session = ort.InferenceSession(onnx_path, providers=providers)

        # Load tags CSV
        tags = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tags.append(row.get("name", row.get("tag_id", "")))
        cls._wd14_tags = tags

        # Find 1boy / 1girl indices
        cls._wd14_boy_idx  = next((i for i,t in enumerate(tags) if t == "1boy"),  None)
        cls._wd14_girl_idx = next((i for i,t in enumerate(tags) if t == "1girl"), None)
        (print if RPRegionalDetailerZImage._debug_mode else lambda *a,**k: None)(f"  [WD14] load complete  tags={len(tags)}  1boy_idx={cls._wd14_boy_idx}  1girl_idx={cls._wd14_girl_idx}")

    @classmethod
    def _classify_gender(cls, clip_model, crop_pil):
        """Classify gender of crop image using WD14 ONNX tagger.
        Returns: 'boy', 'girl', 'unknown'
        """
        import numpy as np

        try:
            cls._load_wd14()
        except Exception as e:
            (print if RPRegionalDetailerZImage._debug_mode else lambda *a,**k: None)(f"    WD14 load failed: {e} → 'unknown'")
            return "unknown"

        # Preprocess: resize to 448×448, RGB, normalize
        img = crop_pil.convert("RGB").resize((448, 448))
        img_np = np.array(img, dtype=np.float32)
        # Convert BGR + normalize (WD14 input format)
        img_np = img_np[:, :, ::-1]          # RGB → BGR
        img_np = np.expand_dims(img_np, 0)   # [1, 448, 448, 3]

        # Inference
        input_name = cls._wd14_session.get_inputs()[0].name
        preds = cls._wd14_session.run(None, {input_name: img_np})[0][0]

        # Compare 1boy / 1girl scores
        score_boy  = float(preds[cls._wd14_boy_idx])  if cls._wd14_boy_idx  is not None else 0.0
        score_girl = float(preds[cls._wd14_girl_idx]) if cls._wd14_girl_idx is not None else 0.0

        (print if RPRegionalDetailerZImage._debug_mode else lambda *a,**k: None)(f"    WD14 gender: 1boy={score_boy:.3f}  1girl={score_girl:.3f}  ", end="")

        THRESH = 0.35  # minimum confidence
        if score_boy >= THRESH or score_girl >= THRESH:
            gender = "boy" if score_boy > score_girl else "girl"
        else:
            gender = "unknown"
        (print if RPRegionalDetailerZImage._debug_mode else lambda *a,**k: None)(f"→ {gender}")
        return gender

    # ── Internal helper: parse gender keywords from prompt ────
    @staticmethod
    def _parse_gender(text):
        """Parse gender from prompt text: 'boy', 'girl', or 'unknown'.
        - majority vote: count boy-related vs girl-related keywords
        - 'unknown' if tied or both zero
        """
        t = text.lower()

        # count boy-related keywords
        boy_keywords  = ["1boy", "2boys", "3boys", "4boys", "5boys",
                         "male", "man", "men", "boy", "he ", "his "]
        girl_keywords = ["1girl", "2girls", "3girls", "4girls", "5girls",
                         "female", "woman", "women", "girl", "she ", "her "]

        boy_count  = sum(t.count(k) for k in boy_keywords)
        girl_count = sum(t.count(k) for k in girl_keywords)

        if boy_count == 0 and girl_count == 0:
            return "unknown"
        if boy_count > girl_count:
            return "boy"
        if girl_count > boy_count:
            return "girl"
        # tie
        return "unknown"

    def execute(self, image, model, clip, vae,
                regional_prompts_nolora, regional_lora_map,
                negative, seed, steps, cfg, sampler_name, scheduler,
                denoise, feather=0, noise_mask=True, force_inpaint=True,
                shift=3.0, bbox_model=None,
                detect_threshold=0.3, drop_size=10, mask_padding=32, mask_blur=8,
                mask_dilation=4, scale_to_pixel=1024,
                use_base=False, use_common=True,
                divide_mode="Horizontal", divide_ratio="", debug=False):

        _dbg = print if debug else lambda *a, **kw: None
        RPRegionalDetailerZImage._debug_mode = debug  # referenced by WD14 methods
        import torch
        import comfy.sample
        import comfy.model_management as mm
        import numpy as np

        if not _COMFY_OK:
            raise RuntimeError("comfy module not found.")

        # ── 1. Prepare image ─────────────────────────────────
        img_tensor = image[0]
        img_np  = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img_h, img_w = img_np.shape[:2]
        print(f"\n[RPRegionalDetailerZImage] start  {img_w}×{img_h}  model={bbox_model}")

        # ── 2. Load YOLO ─────────────────────────────────
        try:
            yolo = self._load_yolo(bbox_model)
        except ImportError:
            raise RuntimeError("ultralytics required: pip install ultralytics")

        # ── 3. Parse prompts → extract COL list ─────────────
        from PIL import Image as _PIL

        if isinstance(regional_prompts_nolora, dict):
            nolora_list = regional_prompts_nolora["nolora"]
            col_texts   = regional_prompts_nolora.get("col_texts", [])
        else:
            nolora_list = list(regional_prompts_nolora)
            col_texts   = nolora_list

        col_lora_map = regional_lora_map if regional_lora_map else {}

        # use_base=True → nolora_list[0]=BASE → COL starts from [1:]
        start_i       = 1 if use_base else 0
        col_list      = nolora_list[start_i:]
        col_text_list = col_texts[start_i:]
        n_cols        = len(col_list)

        if n_cols == 0:
            _dbg("  no COL prompts → return original")
            return (image, image)

        _dbg(f"  DIV prompts: {n_cols}")
        for i, t in enumerate(col_list):
            col_only = col_text_list[i] if i < len(col_text_list) else t
            g = self._parse_gender(col_only)
            # DIV[row,col] label based on divide_ratio/divide_mode
            _dr = (divide_ratio or "").strip()
            _iv = "Ver" in (divide_mode or "Horizontal")
            def _div_tag(idx, ratio_str, is_v):
                if not ratio_str: return f"DIV[0,{idx}]" if not is_v else f"DIV[{idx},0]"
                try:
                    rows=[]; 
                    for rs in ratio_str.split(";"):
                        cols=[v.strip() for v in rs.split(",") if v.strip()]
                        if cols: rows.append(len(cols))
                except: return f"DIV[{idx}]"
                pos=0
                if not is_v:
                    for ri,nc in enumerate(rows):
                        for ci in range(nc):
                            if pos==idx: return f"DIV[{ri},{ci}]"
                            pos+=1
                else:
                    for ci,nr in enumerate(rows):
                        for ri in range(nr):
                            if pos==idx: return f"DIV[{ri},{ci}]"
                            pos+=1
                return f"DIV[{idx}]"
            tag = _div_tag(i, _dr, _iv)
            _dbg(f"    {tag}: gender={g}  col_only='{col_only[:60]}'  full='{t[:40]}'")
        # Gender list (used for matching)
        col_genders = [
            self._parse_gender(col_text_list[i] if i < len(col_text_list) else col_list[i])
            for i in range(n_cols)
        ]
        # ── 4. Detect all persons (no limit) ───────────────
        pil_img = _PIL.fromarray(img_np)
        persons = self._detect_persons(
            yolo, pil_img, img_h, img_w,
            detect_threshold, drop_size
        )
        _dbg(f"  total detected: {len(persons)}  (drop_size={drop_size}px  DIV={n_cols})")
        if not persons:
            _dbg("  no persons → return original")
            return (image, image)

        # ── 5. Classify gender of all masks (WD14 tagger) ───────
        person_genders = []
        for i, (x1,y1,x2,y2,conf,area) in enumerate(persons):
            cx1 = max(0, int(x1)); cy1 = max(0, int(y1))
            cx2 = min(img_w, int(x2)); cy2 = min(img_h, int(y2))
            crop_pil = pil_img.crop((cx1, cy1, cx2, cy2))
            g = self._classify_gender(clip, crop_pil)
            person_genders.append(g)
            _dbg(f"  mask{i}: x={cx1}~{cx2}  gender={g}"
                  f"  conf={conf:.2f}  area={area/img_h/img_w*100:.1f}%")

        # ── 6. divide_mode + divide_ratio → sort direction → order matching ──
        #
        # divide_mode:
        #   Horizontal: "," → left→right, ";" → new row then left→right
        #               mask sort: x(left→right) primary, y(top→bottom) secondary
        #   Vertical:   "," → top→down,  ";" → new col then top→down
        #               mask sort: y(top→bottom) primary, x(left→right) secondary
        #
        # Matching formula:
        #   1. Count required persons per gender
        #   2. Per gender: sort by score(conf×area) desc → take needed count
        #   3. Sort extracted masks by divide_mode direction
        #   4. Assign masks to COLs in order by gender
        #   5. unknown COL → assign remaining masks in order

        _is_vertical = "Ver" in (divide_mode or "Horizontal")
        _ratio_str   = (divide_ratio or "").strip()

        def _score(m_i):
            _, _, _, _, conf, area = persons[m_i]
            return conf * area

        # divide_ratio → compute x/y boundaries (for sorting)
        def _build_boundaries(ratio_str, is_vertical, img_w, img_h):
            """
            Horizontal: ';'→row_boundary(y), ','→col_boundary(x)
            Vertical:   ';'→col_boundary(x), ','→row_boundary(y)
            Returns: (col_x_bounds, row_y_bounds_per_col)
              col_x_bounds: [0, x1, x2, ...] col boundaries (Vertical: ';' based)
              row_y_bounds: row boundary list per col (Vertical: ',' based)
            """
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
                # Horizontal: ';'→row(y), ','→col(x) per row
                row_h_vals = [sum(s) for s in seg_vals]
                y_bounds = to_px(row_h_vals, img_h)
                x_bounds_per_row = [to_px(s, img_w) for s in seg_vals]
                return ("H", y_bounds, x_bounds_per_row)
            else:
                # Vertical: ';'→col(x), ','→row(y) per col
                col_w_vals = [sum(s) for s in seg_vals]
                x_bounds = to_px(col_w_vals, img_w)
                y_bounds_per_col = [to_px(s, img_h) for s in seg_vals]
                return ("V", x_bounds, y_bounds_per_col)

        _bounds = _build_boundaries(_ratio_str, _is_vertical, img_w, img_h)

        def _sort_key(m_i):
            """
            Sort based on divide_ratio boundaries:
            Horizontal: row number(y boundary) first → then col number(x boundary)
            Vertical:   col number(x boundary) first → then row number(y boundary)
            No boundary info: H=(cy,cx), V=(cx,cy)
            """
            x1, y1, x2, y2, _, _ = persons[m_i]
            cx, cy = (x1+x2)/2, (y1+y2)/2

            if _bounds is None:
                return (cy, cx) if not _is_vertical else (cx, cy)

            mode = _bounds[0]
            if mode == "H":
                y_bounds, x_bounds_per_row = _bounds[1], _bounds[2]
                # row number
                row_n = 0
                for ri in range(len(y_bounds)-1):
                    if y_bounds[ri] <= cy < y_bounds[ri+1]:
                        row_n = ri; break
                # col number (x boundary of the row)
                xb = x_bounds_per_row[row_n] if row_n < len(x_bounds_per_row) else [0, img_w]
                col_n = 0
                for ci in range(len(xb)-1):
                    if xb[ci] <= cx < xb[ci+1]:
                        col_n = ci; break
                return (row_n, col_n, cx)  # x as tiebreaker
            else:  # V
                x_bounds, y_bounds_per_col = _bounds[1], _bounds[2]
                # col number
                col_n = 0
                for ci in range(len(x_bounds)-1):
                    if x_bounds[ci] <= cx < x_bounds[ci+1]:
                        col_n = ci; break
                # row number (y boundary of the col)
                yb = y_bounds_per_col[col_n] if col_n < len(y_bounds_per_col) else [0, img_h]
                row_n = 0
                for ri in range(len(yb)-1):
                    if yb[ri] <= cy < yb[ri+1]:
                        row_n = ri; break
                return (col_n, row_n, cy)  # y as tiebreaker

        dir_label = "top→bottom/col-reset" if _is_vertical else "left→right/row-reset"
        _dbg(f"  sort direction: [{dir_label}]  (divide_ratio='{_ratio_str}')")

        # COL index → DIV[row, col] conversion helper
        # Calculate which row/col each COL belongs to from divide_ratio structure
        def _col_to_div(col_i, ratio_str, is_vertical):
            """Convert COL index to DIV[row,col] format.
            No divide_ratio: Horizontal=DIV[0,col_i], Vertical=DIV[col_i,0]
            """
            if not ratio_str:
                return (0, col_i) if not is_vertical else (col_i, 0)
            try:
                row_strs = ratio_str.split(";")
                rows = []
                for rs in row_strs:
                    cols = [v.strip() for v in rs.split(",") if v.strip()]
                    if cols:
                        rows.append(len(cols))
            except Exception:
                return (0, col_i)
            # Convert COL index to row/col coordinates
            idx = 0
            if not is_vertical:
                # Horizontal: row→y, col→x
                for ri, ncols in enumerate(rows):
                    for ci in range(ncols):
                        if idx == col_i:
                            return (ri, ci)
                        idx += 1
            else:
                # Vertical: row→x(col), col→y(row)
                for ri, nrows in enumerate(rows):
                    for ci in range(nrows):
                        if idx == col_i:
                            return (ci, ri)
                        idx += 1
            return (0, col_i)

        def _div_label(col_i):
            r, c = _col_to_div(col_i, _ratio_str, _is_vertical)
            return f"DIV[{r},{c}]"

        # Count required count per gender
        need_count = {}
        for pg in col_genders:
            need_count[pg] = need_count.get(pg, 0) + 1
        _dbg(f"  required genders: {need_count}")

        # Per gender: sort by score desc → take needed → sort by direction
        selected        = {}
        used_for_select = set()
        for pg, needed in need_count.items():
            if pg == "unknown":
                continue
            cands = sorted(
                [m_i for m_i, m_g in enumerate(person_genders)
                 if m_g == pg or m_g == "unknown"],
                key=_score, reverse=True
            )[:needed]
            cands.sort(key=_sort_key)   # sort by position direction
            selected[pg] = list(cands)
            used_for_select.update(cands)
            _dbg(f"  selected({pg}×{needed}): "
                  + ", ".join(f"mask{m}(score={_score(m):.0f})" for m in cands))

        # Assign to COLs in order
        ptr     = {pg: 0 for pg in selected}
        matched = {}
        for col_i, pg in enumerate(col_genders):
            if pg in selected and ptr[pg] < len(selected[pg]):
                pick = selected[pg][ptr[pg]]
                ptr[pg] += 1
                matched[col_i] = pick
                x1,y1,x2,y2,conf,area = persons[pick]
                _dbg(f"  match: {_div_label(col_i)}({pg}) ↔ mask{pick}"
                      f"({person_genders[pick]})"
                      f"  x={int((x1+x2)/2)}  y={int((y1+y2)/2)}"
                      f"  score={conf*area:.0f}")
            else:
                _dbg(f"  match fail: {_div_label(col_i)}({pg}) → insufficient masks")

        # unknown COL → assign remaining masks in order
        remaining = sorted(
            [m for m in range(len(persons))
             if m not in used_for_select and m not in matched.values()],
            key=_sort_key
        )
        for col_i in range(n_cols):
            if col_i in matched:
                continue
            if remaining:
                pick = remaining.pop(0)
                matched[col_i] = pick
                x1,y1,x2,y2,conf,area = persons[pick]
                _dbg(f"  match(fallback): {_div_label(col_i)}(unknown) ↔ mask{pick}"
                      f"  x={int((x1+x2)/2)}  y={int((y1+y2)/2)}")

        if not matched:
            _dbg("  no matches → return original")
            return (image, image)

        # ── 7. Build debug image ─────────────────────────────
        debug_np = img_np.copy()
        colors = [(255,80,80),(80,255,80),(80,80,255),
                  (255,255,80),(255,80,255),(80,255,255)]
        for col_i, m_i in matched.items():
            x1,y1,x2,y2,*_ = persons[m_i]
            c = colors[col_i % len(colors)]
            ix1,iy1,ix2,iy2 = int(x1),int(y1),int(x2),int(y2)
            for t in range(3):
                debug_np[max(0,iy1-t):iy1+t+1, ix1:ix2] = c
                debug_np[iy2-t:min(img_h,iy2+t+1), ix1:ix2] = c
                debug_np[iy1:iy2, max(0,ix1-t):ix1+t+1] = c
                debug_np[iy1:iy2, ix2-t:min(img_w,ix2+t+1)] = c
        debug_tensor = torch.from_numpy(debug_np.astype(np.float32)/255.0).unsqueeze(0)

        # ── 8. Prepare ModelSamplingAuraFlow(shift) ────────────
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
                _dbg(f"  AuraFlow sigma shift={shift} applied")
            except Exception as e:
                _dbg(f"  AuraFlow shift failed: {e}")
                try: del _tmp
                except: pass

        device  = mm.get_torch_device()
        result_img = img_tensor.clone()

        # ── 9. Inpaint per matched COL ──────────────────────────
        for col_i, m_i in matched.items():
            x1,y1,x2,y2,conf,area = persons[m_i]

            # padding + align to 8
            px1 = (max(0,      int(x1) - mask_padding) // 8) * 8
            py1 = (max(0,      int(y1) - mask_padding) // 8) * 8
            px2 = min(img_w, ((int(x2) + mask_padding + 7) // 8) * 8)
            py2 = min(img_h, ((int(y2) + mask_padding + 7) // 8) * 8)
            crop_w, crop_h = px2-px1, py2-py1
            if crop_w <= 0 or crop_h <= 0:
                continue

            # Upscale based on scale_to_pixel
            # mask < scale_to_pixel: upscale by long edge
            # mask >= scale_to_pixel: skip upscale
            max_side = max(crop_w, crop_h)
            if max_side < scale_to_pixel:
                scale_ratio = scale_to_pixel / max_side
                up_w = max(32, (int(crop_w * scale_ratio) // _ZIMAGE_GRID_SIZE) * _ZIMAGE_GRID_SIZE)
                up_h = max(32, (int(crop_h * scale_ratio) // _ZIMAGE_GRID_SIZE) * _ZIMAGE_GRID_SIZE)
                up_w = min(4096, up_w)
                up_h = min(4096, up_h)
            else:
                up_w = max(32, (crop_w // _ZIMAGE_GRID_SIZE) * _ZIMAGE_GRID_SIZE)
                up_h = max(32, (crop_h // _ZIMAGE_GRID_SIZE) * _ZIMAGE_GRID_SIZE)

            crop_np  = img_np[py1:py2, px1:px2]
            crop_up  = _PIL.fromarray(crop_np).resize((up_w, up_h), _PIL.LANCZOS)
            _dbg(f"  [{_div_label(col_i)}↔mask{m_i}] crop={crop_w}×{crop_h} → up={up_w}×{up_h}"
                 f"({'skip' if max_side >= scale_to_pixel else f'→{scale_to_pixel}px'})")

            # VAE encode (16ch)
            crop_t = torch.from_numpy(
                np.array(crop_up).astype(np.float32)/255.0
            ).unsqueeze(0).to(device)
            try:
                latent_crop = vae.encode(crop_t)
                if isinstance(latent_crop, dict):
                    latent_crop = latent_crop["samples"]
                _dbg(f"    VAE encode: {tuple(latent_crop.shape)}")
            except Exception as e:
                _dbg(f"    VAE encode failed: {e} → zeros latent")
                lh = up_h // _ZIMAGE_LATENT_BLOCK
                lw = up_w // _ZIMAGE_LATENT_BLOCK
                latent_crop = torch.zeros(
                    (1,_ZIMAGE_LATENT_CHANNELS,lh,lw),
                    device=mm.intermediate_device()
                )

            # CLIP encode
            # col_texts[0] = _base_part (between ADDCOMM and ADDBASE) = BASE text
            col_full    = col_list[col_i]
            col_only    = col_text_list[col_i] if col_i < len(col_text_list) else col_full
            common_text = regional_prompts_nolora.get("common", "") \
                          if isinstance(regional_prompts_nolora, dict) else ""
            all_col_texts = regional_prompts_nolora.get("col_texts", []) \
                            if isinstance(regional_prompts_nolora, dict) else []
            # BASE = col_texts[0] (if use_base=True, nolora_list[0]=BASE)
            base_text   = all_col_texts[0] if (use_base and all_col_texts) else ""
            _use_com    = use_common and bool(common_text)

            if _use_com and use_base and base_text and col_only:
                # COMMON + BASE + DIV
                tok_com  = clip.tokenize(common_text)
                tok_base = clip.tokenize(base_text)
                tok_col  = clip.tokenize(col_only)
                c_com,  _       = _enc(clip, tok_com) 
                c_base, _       = _enc(clip, tok_base)
                c_col,  pooled  = _enc(clip, tok_col) 
                cond = torch.cat([c_com, c_base, c_col], dim=1)
            elif _use_com and col_only:
                # COMMON + DIV
                tok_com = clip.tokenize(common_text)
                tok_col = clip.tokenize(col_only)
                c_com, _       = _enc(clip, tok_com)
                c_col, pooled  = _enc(clip, tok_col)
                cond = torch.cat([c_com, c_col], dim=1)
            elif use_base and base_text and col_only:
                # BASE + DIV
                tok_base = clip.tokenize(base_text)
                tok_col  = clip.tokenize(col_only)
                c_base, _       = _enc(clip, tok_base)
                c_col,  pooled  = _enc(clip, tok_col) 
                cond = torch.cat([c_base, c_col], dim=1)
            else:
                tok = clip.tokenize(col_only if col_only else col_full)
                cond, pooled = _enc(clip, tok)
            positive_cond = [[cond, {"pooled_output": pooled, "cross_attn": cond}]]

            # Apply LoRA
            pidx = col_i + start_i
            sample_model = model_shifted
            loras_for_col = col_lora_map.get(pidx, {})
            if loras_for_col and _COMFY_OK:
                from core.lora_manager import _apply_loras
                sample_model, _ = _apply_loras(model_shifted, clip, loras_for_col)

            # Build mask_np (needed before sampling: shared by noise_mask and blending)
            mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            bx1 = max(0, int(x1)-px1); by1 = max(0, int(y1)-py1)
            bx2 = min(crop_w, int(x2)-px1); by2 = min(crop_h, int(y2)-py1)
            mask_np[by1:by2, bx1:bx2] = 1.0
            try:
                import cv2
                if mask_dilation > 0:
                    k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,(mask_dilation*2+1,mask_dilation*2+1))
                    mask_np = cv2.dilate(mask_np, k)
                if mask_blur > 0:
                    ks = mask_blur*2+1
                    mask_np = cv2.GaussianBlur(mask_np,(ks,ks),mask_blur)
                if feather > 0:
                    ks = feather*2+1
                    mask_np = cv2.GaussianBlur(mask_np,(ks,ks),feather)
                    mask_np = np.clip(mask_np, 0.0, 1.0)
            except ImportError:
                pass

            # sampling
            noise = comfy.sample.prepare_noise(latent_crop, seed + col_i, None)
            _dbg(f"    sampling  denoise={denoise}  steps={steps}  cfg={cfg}")

            # Build noise_mask tensor (resample to latent resolution)
            _noise_mask_t = None
            if noise_mask:
                import torch.nn.functional as _F
                lh, lw = latent_crop.shape[2], latent_crop.shape[3]
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
                _noise_mask_t = _F.interpolate(
                    mask_t.float(), size=(lh, lw), mode="bilinear", align_corners=False
                ).squeeze(0).to(latent_crop.device)

            out = comfy.sample.sample(
                model=sample_model, noise=noise,
                steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive_cond, negative=negative,
                latent_image=latent_crop,
                denoise=denoise, seed=seed + col_i,
                noise_mask=_noise_mask_t,
            )
            out_lat = out["samples"] if isinstance(out, dict) else out

            # VAE decode
            inpaint_img = vae.decode(out_lat)
            # Qwen VAE may return 5D [B,T,H,W,C] → normalize to [B,H,W,C]
            if inpaint_img.ndim == 5:
                inpaint_img = inpaint_img[:, 0, :, :, :]
            inpaint_t   = inpaint_img[0].cpu().float()
            if torch.isnan(inpaint_t).any() or torch.isinf(inpaint_t).any():
                _dbg(f"    ⚠ NaN/inf → clamping")
                inpaint_t = torch.nan_to_num(inpaint_t, nan=0.0, posinf=1.0, neginf=0.0)
            inpaint_t   = inpaint_t.clamp(0.0, 1.0)
            inpaint_np  = (inpaint_t.numpy() * 255).astype(np.uint8)

            # Fix size + downscale
            if inpaint_np.shape[:2] != (up_h, up_w):
                inpaint_np = np.array(
                    _PIL.fromarray(inpaint_np).resize((up_w,up_h),_PIL.LANCZOS))
            inpaint_down = np.array(
                _PIL.fromarray(inpaint_np).resize((crop_w,crop_h),_PIL.LANCZOS))

            # Mask (inside bbox = 1)
            mask_blend = mask_np[:,:,np.newaxis]

            # blending
            orig_crop = (result_img[py1:py2,px1:px2].cpu().numpy()*255).astype(np.uint8)
            blended   = (inpaint_down*mask_blend + orig_crop*(1-mask_blend)).astype(np.uint8)
            result_img[py1:py2,px1:px2] = torch.from_numpy(
                blended.astype(np.float32)/255.0)
            _dbg(f"    inpainting done")
            # Release model reference used for sampling
            try:
                if sample_model is not model_shifted:
                    if hasattr(sample_model, 'patches'):
                        sample_model.patches.clear()
                    if hasattr(sample_model, 'object_patches'):
                        sample_model.object_patches.clear()
                    if hasattr(sample_model, 'object_patches_backup'):
                        sample_model.object_patches_backup.clear()
                    del sample_model
                del latent_crop, out_lat, positive_cond
                del inpaint_img, inpaint_t, inpaint_np, inpaint_down
            except Exception:
                pass

        # Release model_shifted reference (prevent circular ref / memory leak)
        try:
            if _cloned and model_shifted is not model:
                # Clear all patches to break circular references
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

        print(f"[RPRegionalDetailerZImage] done")
        return (result_img.unsqueeze(0), debug_tensor)
