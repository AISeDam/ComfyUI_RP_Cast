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

# Import RPRegionalDetailerZImage for delegation
def _get_zi_detailer():
    import importlib.util
    _key = "nodes.node_rp_detailer_zimage"
    if _key in sys.modules:
        return sys.modules[_key]
    _spec = importlib.util.spec_from_file_location(
        _key, os.path.join(_HERE, "node_rp_detailer_zimage.py"))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_key] = _mod
    _spec.loader.exec_module(_mod)
    return _mod

_zi_mod = _get_zi_detailer()
RPRegionalDetailerZImage = _zi_mod.RPRegionalDetailerZImage
del _zi_mod, _get_zi_detailer

# 9. RPRegionalDetailerQwen
#    - Qwen Image dedicated Regional Detailer
#    - Same behavior as RPRegionalDetailerZImage
#    - 5D latent handling, conditioning without cross_attn
# ══════════════════════════════════════════════════════
class RPRegionalDetailerQwen:
    CATEGORY = "Regional Prompter"
    cnr_id  = "ComfyUI_RP_Cast"

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
                "steps":   ("INT",   {"default": 20, "min": 1, "max": 100,
                                      "tooltip": "Qwen recommended: 15~20 steps."}),
                "cfg":     ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                      "tooltip": "CFG=1.0 recommended for Qwen distilled."}),
                "sampler_name": (_SAMPLERS,),
                "scheduler":    (_SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "img2img denoise strength. 0.35~0.55 recommended."}),
                "feather":       ("INT",   {"default": 0,  "min": 0, "max": 64,
                                            "tooltip": "Feather radius in pixels. Applied after dilation."}),
                "noise_mask":    ("BOOLEAN", {"default": True,
                                              "tooltip": "Apply noise mask to latent before sampling."}),
                "force_inpaint": ("BOOLEAN", {"default": True,
                                              "tooltip": "Force inpainting mode."}),
                "shift":   ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5,
                                      "tooltip": "AuraFlow sigma shift for Qwen."}),
                "bbox_model":        (bbox_models,),
                "detect_threshold":  ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0,
                                                "step": 0.05,
                                                "tooltip": "YOLO detection confidence threshold."}),
                "drop_size":         ("INT",   {"default": 10, "min": 1, "max": 16384,
                                                "step": 1,
                                                "tooltip": "Minimum detection size (px)."}),
                "mask_padding":      ("INT",   {"default": 32, "min": 0, "max": 256}),
                "mask_blur":         ("INT",   {"default": 8,  "min": 0, "max": 64}),
                "mask_dilation":     ("INT",   {"default": 4,  "min": 0, "max": 64}),
                "scale_to_pixel":    ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32,
                                     "tooltip": "Target pixel size for upscaling. "
                                                "Masks smaller than this are upscaled before inpainting. "
                                                "Masks larger than this are processed as-is."}),
            },
            "optional": {
                "use_base":     ("BOOLEAN", {"default": False,
                                             "tooltip": "Prepend BASE prompt to each COL prompt."}),
                "use_common":   ("BOOLEAN", {"default": True}),
                "divide_mode":  ("RP_DIV_MODE",  {"default": "Horizontal",
                                             "forceInput": True,
                                             "tooltip": "Connect divide_mode from RPPromptParser."}),
                "divide_ratio": ("RP_DIV_RATIO",  {"default": "",
                                             "forceInput": True,
                                             "tooltip": "Connect divide_ratio from RPPromptParser."}),
                "debug":           ("BOOLEAN", {"default": False,
                                   "tooltip": "Print debug log when enabled."}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("image", "debug_image")
    FUNCTION      = "execute"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def execute(self, image, model, clip, vae,
                regional_prompts_nolora, regional_lora_map,
                negative, seed, steps, cfg, sampler_name, scheduler,
                denoise, feather=0, noise_mask=True, force_inpaint=True,
                shift=3.0, bbox_model=None,
                detect_threshold=0.3, drop_size=10, mask_padding=32, mask_blur=8,
                mask_dilation=4, scale_to_pixel=1024,
                use_base=False, use_common=True,
                divide_mode="Horizontal", divide_ratio="", debug=False):

        # Delegate to RPRegionalDetailerZImage.execute
        _dbg = print if debug else lambda *a, **kw: None
        import torch
        import comfy.sample
        import comfy.model_management as mm
        import numpy as np

        if not _COMFY_OK:
            raise RuntimeError("comfy module not found.")

        # Qwen VAE decode may return 5D [B,T,H,W,C]
        # ZImage detailer expects [B,H,W,C] 4D → normalize
        _image = image
        if image.ndim == 5:
            _image = image[:, 0, :, :, :]
            _dbg(f"[RPRegionalDetailerQwen] image 5D→4D: {tuple(image.shape)} → {tuple(_image.shape)}")
        elif image.ndim == 4 and image.shape[1] == 1:
            _image = image.squeeze(1)
            _dbg(f"[RPRegionalDetailerQwen] image squeeze: {tuple(image.shape)} → {tuple(_image.shape)}")

        zi_detailer = RPRegionalDetailerZImage()
        _dbg(f"[RPRegionalDetailerQwen] → delegating to RPRegionalDetailerZImage (Qwen compat)")

        result = zi_detailer.execute(
            image=_image, model=model, clip=clip, vae=vae,
            regional_prompts_nolora=regional_prompts_nolora,
            regional_lora_map=regional_lora_map,
            negative=negative, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler,
            denoise=denoise, feather=feather,
            noise_mask=noise_mask, force_inpaint=force_inpaint,
            shift=shift, bbox_model=bbox_model,
            detect_threshold=detect_threshold, drop_size=drop_size,
            mask_padding=mask_padding, mask_blur=mask_blur,
            mask_dilation=mask_dilation, scale_to_pixel=scale_to_pixel,
            use_base=use_base, use_common=use_common,
            divide_mode=divide_mode, divide_ratio=divide_ratio,
            debug=debug,
        )
        del zi_detailer
        return result


