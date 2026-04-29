"""
RP Txt2Img (Grok) - Regional Prompter to xAI Grok Image API
Converts ADDCOMM/ADDBASE/ADDCOL/ADDROW prompt syntax to natural language
and generates images via xAI Grok Image Generation API.
API: https://docs.x.ai/developers/rest-api-reference/inference/images
"""
from __future__ import annotations
import json, urllib.request, urllib.error, base64
from ._rp_txt2img_common import (
    _is_rp_prompt, _convert_rp_to_natural, _bytes_to_tensor,
    _get_setting, _regions_to_col_n_row,
)

_GROK_IMAGE_MODELS = [
    "grok-imagine-image",
    "grok-imagine-image-pro",
]

_ASPECT_RATIOS = [
    "1:1", "3:4", "4:3", "9:16", "16:9",
    "2:3", "3:2", "9:19.5", "19.5:9",
    "9:20", "20:9", "1:2", "2:1", "auto",
]


class RPTxt2ImgGrok:
    """RP 프롬프트를 xAI Grok 이미지 생성 API로 이미지를 생성하는 노드."""
    CATEGORY = "RP Cast"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":        (_GROK_IMAGE_MODELS, {"default": "grok-imagine-image"}),
                "prompt":       ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "ADDCOMM/ADDBASE/ADDCOL/ADDROW 포함 RP 프롬프트 또는 일반 프롬프트",
                }),
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "16:9"}),
                "quality":      (["low", "medium", "high"], {"default": "medium"}),
                "resolution":   (["1k", "2k"], {"default": "1k"}),
                "debug":        ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "regional_col_n_row": ("RP_REGIONS",
                    {"tooltip": "RPRatioParser의 regional_col_n_row 출력과 연결."}),
                "divide_mode": ("RP_DIV_MODE",
                    {"tooltip": "RP Prompt Parser의 divide_mode 출력과 연결."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "execute"
    OUTPUT_NODE  = True

    def execute(self, model="grok-imagine-image", prompt="",
                aspect_ratio="16:9", quality="medium", resolution="1k",
                debug=False, regional_col_n_row=None, divide_mode="Horizontal"):
        import torch

        api_key = _get_setting("ComfyUI-RP-Cast.Configuration.grok_api_key").strip()
        if not api_key:
            raise RuntimeError(
                "Grok API Key가 설정되지 않았습니다.\n"
                "Settings > ComfyUI-RP-Cast > Configuration > grok_api_key")

        col_n_row_str = _regions_to_col_n_row(regional_col_n_row, divide_mode, debug, "RPTxt2ImgGrok")
        final_prompt  = (_convert_rp_to_natural(prompt, col_n_row_str, divide_mode, debug)
                         if _is_rp_prompt(prompt) else prompt)

        print(f"[RPTxt2ImgGrok] model={model}  aspect_ratio={aspect_ratio}  quality={quality}  resolution={resolution}")
        print(f"[RPTxt2ImgGrok] final_prompt:\n{final_prompt}")

        body = {
            "model":           model,
            "prompt":          final_prompt,
            "n":               1,
            "aspect_ratio":    aspect_ratio,
            "quality":         quality,
            "resolution":      resolution,
            "response_format": "b64_json",
        }

        req = urllib.request.Request(
            "https://api.x.ai/v1/images/generations",
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
            raise RuntimeError(f"Grok API Error {e.code}: {e.read().decode()}")

        tensors = []
        for item in data.get("data", []):
            b64 = item.get("b64_json", "")
            if not b64:
                continue
            mime = item.get("mime_type", "")
            if debug:
                print(f"[RPTxt2ImgGrok] mime_type={mime}")
            tensor = _bytes_to_tensor(base64.b64decode(b64), mime)
            if tensor is not None:
                tensors.append(tensor)

        if not tensors:
            raise RuntimeError("Grok API 응답에 이미지 데이터가 없습니다.")

        return (torch.cat(tensors, dim=0),)


NODE_CLASS_MAPPINGS        = {"RPTxt2ImgGrok": RPTxt2ImgGrok}
NODE_DISPLAY_NAME_MAPPINGS = {"RPTxt2ImgGrok": "RP Txt2Img (Grok)"}
