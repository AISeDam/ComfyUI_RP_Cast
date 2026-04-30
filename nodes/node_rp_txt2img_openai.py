"""
RP Txt2Img (OpenAI) - Regional Prompter to GPT Image API
Version: 0.5.59
"""
from __future__ import annotations
import json, urllib.request, urllib.error
import base64
from ._rp_txt2img_common import (
    _is_rp_prompt, _convert_rp_to_natural, _bytes_to_tensor,
    _get_setting, _regions_to_col_n_row,
)

# ══════════════════════════════════════════════════════════════════════════════
# ComfyUI Node Class
# ══════════════════════════════════════════════════════════════════════════════
class RPTxt2ImgOpenAI:
    """Generate images from RP prompts via OpenAI GPT Image API."""

    CATEGORY = "RP Cast"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":   (["gpt-image-2", "gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini"],
                           {"default": "gpt-image-2"}),
                "prompt":  ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "RP prompt with ADDCOMM/ADDBASE/ADDCOL/ADDROW syntax, or plain text.",
                }),
                "size":    (["1024x1024", "1536x1024", "1024x1536", "auto"], {"default": "1536x1024"}),
                "quality": (["auto", "high", "medium", "low"], {"default": "high"}),
                "debug":   ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "regional_col_n_row": ("RP_REGIONS", {
                    "tooltip": (
                        "Connect regional_col_n_row output from RPRatioParser.\n"
                        "If not connected: auto-detected from ADDROW presence (default: Horizontal)."
                    ),
                }),
                "divide_mode": ("RP_DIV_MODE", {
                    "tooltip": "Connect divide_mode output from RPPromptParser.",
                }),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "execute"
    OUTPUT_NODE   = True

    def execute(
        self,
        model: str = "gpt-image-2",
        prompt: str = "",
        size: str = "1536x1024",
        quality: str = "high",
        debug: bool = False,
        regional_col_n_row=None,
        divide_mode: str = "Horizontal",
        background: str = "auto",
    ):
        import torch, base64, struct, zlib
        from io import BytesIO

        # Load API key
        api_key = _get_setting("ComfyUI-RP-Cast.Configuration.openai_api_key").strip()
        if not api_key:
            raise RuntimeError(
                "OpenAI API Key is not set.\n"
                "Settings > ComfyUI-RP-Cast > Configuration > openai_api_key"
            )

        # Extract col_n_row string from RP_REGIONS
        # Horizontal: len(region_rows)=n_rows, len(r.cols)=n_cols
        # Vertical  : len(region_rows)=n_cols, len(r.cols)=n_rows (swapped)
        col_n_row_str = ""
        if regional_col_n_row is not None:
            try:
                region_rows = regional_col_n_row  # RP_REGIONS = List[RegionRow]
                rr_len = len(region_rows)
                rc_len = max(len(r.cols) for r in region_rows) if rr_len > 0 else 1
                if divide_mode == "Vertical":
                    # Vertical: region_rows=cols, r.cols=rows
                    n_cols = rr_len
                    n_rows = rc_len
                else:
                    # Horizontal: region_rows=rows, r.cols=cols
                    n_rows = rr_len
                    n_cols = rc_len
                col_n_row_str = f"{n_cols}x{n_rows}"
                if debug:
                    print(f"[RPTxt2ImgOpenAI] RP_REGIONS → {n_rows}rows x {n_cols}cols  mode={divide_mode}")
            except Exception as e:
                print(f"[RPTxt2ImgOpenAI] regional_col_n_row parse failed: {e}")

        # Convert prompt
        if _is_rp_prompt(prompt):
            final_prompt = _convert_rp_to_natural(
                prompt,
                regional_col_n_row=col_n_row_str,
                divide_mode=divide_mode,
                debug=debug,
            )
        else:
            final_prompt = prompt

        print(f"[RPTxt2ImgOpenAI] model={model}  size={size}  quality={quality}")
        print(f"[RPTxt2ImgOpenAI] final_prompt:\n{final_prompt}")

        # Call API
        body = {
            "model": model,
            "prompt": final_prompt,
            "n": 1,
            "size": size,
            "quality": quality,
        }
        if background != "auto":
            body["background"] = background

        req = urllib.request.Request(
            "https://api.openai.com/v1/images/generations",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenAI API Error {e.code}: {e.read().decode()}")

        # base64 → Tensor
        tensors = []
        for item in data.get("data", []):
            b64 = item.get("b64_json", "")
            if not b64:
                continue
            img_bytes = base64.b64decode(b64)
            tensor = _bytes_to_tensor(img_bytes)
            if tensor is not None:
                tensors.append(tensor)

        if not tensors:
            raise RuntimeError("API response contains no image data.")

        return (torch.cat(tensors, dim=0),)




NODE_CLASS_MAPPINGS = {"RPTxt2ImgOpenAI": RPTxt2ImgOpenAI}
NODE_DISPLAY_NAME_MAPPINGS = {"RPTxt2ImgOpenAI": "RP Txt2Img (OpenAI)"}
