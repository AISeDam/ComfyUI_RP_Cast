"""
RP Txt2Img (Gemini) - Google Gemini Image Generation API
Version: 0.5.59

REST API: https://ai.google.dev/gemini-api/docs/image-generation
"""
from __future__ import annotations
import json, urllib.request, urllib.error, base64
from ._rp_txt2img_common import (
    _is_rp_prompt, _convert_rp_to_natural, _bytes_to_tensor,
    _get_setting, _regions_to_col_n_row,
)

_GEMINI_IMAGE_MODELS = [
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
]


class RPTxt2ImgGemini:
    CATEGORY = "RP Cast"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":        (_GEMINI_IMAGE_MODELS,
                                 {"default": "gemini-3.1-flash-image-preview"}),
                "prompt":       ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "RP prompt with ADDCOMM/ADDBASE/ADDCOL/ADDROW syntax, or plain text.",
                }),
                "aspect_ratio": (
                    ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5",
                     "5:4", "9:16", "16:9", "21:9"],
                    {"default": "16:9"}
                ),
                "image_size":   (["1K", "2K", "4K"], {"default": "1K"}),
                "debug":        ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "regional_col_n_row": ("RP_REGIONS",
                    {"tooltip": "Connect regional_col_n_row output from RPRatioParser."}),
                "divide_mode": ("RP_DIV_MODE",
                    {"tooltip": "Connect divide_mode output from RPPromptParser."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "execute"
    OUTPUT_NODE  = True

    def execute(self, model="gemini-3.1-flash-image-preview", prompt="",
                aspect_ratio="16:9", image_size="1K", debug=False,
                regional_col_n_row=None, divide_mode="Horizontal"):
        import torch

        api_key = _get_setting("ComfyUI-RP-Cast.Configuration.gemini_api_key").strip()
        if not api_key:
            raise RuntimeError(
                "Gemini API Key is not set.\n"
                "Settings > ComfyUI-RP-Cast > Configuration > gemini_api_key\n"
                "Get API Key: https://aistudio.google.com/apikey")

        col_n_row_str = _regions_to_col_n_row(regional_col_n_row, divide_mode, debug, "RPTxt2ImgGemini")
        final_prompt  = (_convert_rp_to_natural(prompt, col_n_row_str, divide_mode, debug)
                         if _is_rp_prompt(prompt) else prompt)

        print(f"[RPTxt2ImgGemini] model={model}  aspect_ratio={aspect_ratio}  image_size={image_size}")
        print(f"[RPTxt2ImgGemini] final_prompt:\n{final_prompt}")

        url = (f"https://generativelanguage.googleapis.com/v1beta"
               f"/models/{model}:generateContent")
        body = {
            "contents": [{"parts": [{"text": final_prompt}], "role": "user"}],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize":   image_size,
                },
            },
        }

        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Gemini API Error {e.code}: {e.read().decode()}")

        tensors = []
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part and debug:
                    print(f"[RPTxt2ImgGemini] text: {part['text'][:100]}")
                inline = part.get("inlineData", {})
                if inline.get("data"):
                    mime = inline.get("mimeType", "")
                    if debug:
                        print(f"[RPTxt2ImgGemini] mimeType={mime}  data_len={len(inline['data'])}")
                    tensor = _bytes_to_tensor(base64.b64decode(inline["data"]), mime)
                    if tensor is not None:
                        tensors.append(tensor)

        if not tensors:
            raise RuntimeError(
                f"Gemini API response contains no image data.\n"
                f"Response: {json.dumps(data, ensure_ascii=False)[:300]}")

        return (torch.cat(tensors, dim=0),)


NODE_CLASS_MAPPINGS        = {"RPTxt2ImgGemini": RPTxt2ImgGemini}
NODE_DISPLAY_NAME_MAPPINGS = {"RPTxt2ImgGemini": "RP Txt2Img (Gemini)"}
