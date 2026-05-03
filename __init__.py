# SPDX-License-Identifier: AGPL-3.0-or-later
#
# ComfyUI_RP_Cast
# Copyright (C) 2024-2026  ComfyUI_RP_Cast Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Third-party references (algorithm/pattern, not direct code copy):
#   - sd-webui-regional-prompter (hako-mikan) : AGPL-3.0
#   - ComfyUI-Impact-Pack (ltdrdata)           : AGPL-3.0
#   - ComfyUI-ZImagePowerNodes (martin-rizzo)  : MIT
#   - ComfyUI-WD14-Tagger (pythongosssss)      : MIT
#
"""
ComfyUI_RP_Cast  —  Regional Prompter for ComfyUI
Version: 0.5.59
"""
import os, sys, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Load node modules via absolute path (avoids package name conflicts) ────────
def _load_module(name, rel_path):
    """Load a module from a path relative to this package directory."""
    abs_path = os.path.join(_HERE, rel_path)
    spec = importlib.util.spec_from_file_location(
        f"ComfyUI_RP_Cast.{name}", abs_path,
        submodule_search_locations=[os.path.dirname(abs_path)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ComfyUI_RP_Cast.{name}"] = mod
    # Also register under short name for cross-module imports
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load shared utilities first
_shared   = _load_module("nodes._shared",                  "nodes/_shared.py")
_common   = _load_module("nodes._rp_txt2img_common",       "nodes/_rp_txt2img_common.py")
_pp       = _load_module("nodes.node_rp_prompt_parser",    "nodes/node_rp_prompt_parser.py")
_ks       = _load_module("nodes.node_rp_ksampler",         "nodes/node_rp_ksampler.py")
_det      = _load_module("nodes.node_rp_detailer",         "nodes/node_rp_detailer.py")
_det_zi   = _load_module("nodes.node_rp_detailer_zimage",  "nodes/node_rp_detailer_zimage.py")
_conv     = _load_module("nodes.node_rp_converter",       "nodes/node_rp_converter.py")
_det_qw   = _load_module("nodes.node_rp_detailer_qwen",    "nodes/node_rp_detailer_qwen.py")
_oai      = _load_module("nodes.node_rp_txt2img_openai",   "nodes/node_rp_txt2img_openai.py")
_gem      = _load_module("nodes.node_rp_txt2img_gemini",   "nodes/node_rp_txt2img_gemini.py")
_grok     = _load_module("nodes.node_rp_txt2img_grok",     "nodes/node_rp_txt2img_grok.py")

# ── Node class references ──────────────────────────────────────────────────────
RPPromptParser           = _pp._RPPromptParser           if hasattr(_pp, "_RPPromptParser") else _pp.RPPromptParser
RPRatioParser            = _pp.RPRatioParser
RPKSampler               = _ks.RPKSampler
RPRegionalDetailer       = _det.RPRegionalDetailer
RPRegionalDetailerZImage = _det_zi.RPRegionalDetailerZImage
RPConverter              = _conv.RPConverter
RPRegionalDetailerQwen   = _det_qw.RPRegionalDetailerQwen

# For test compatibility
_auto_aratios  = _pp._auto_aratios
_count_addcol  = _pp._count_addcol

# ── Registration ──────────────────────────────────────────────────────────────
_VERSION = "0.5.59"

NODE_CLASS_MAPPINGS = {
    "RPPromptParser":           RPPromptParser,
    "RPRatioParser":            RPRatioParser,
    "RPKSampler":               RPKSampler,
    "RPRegionalDetailer":       RPRegionalDetailer,
    "RPRegionalDetailerZImage": RPRegionalDetailerZImage,
    "RPConverter":             RPConverter,
    "RPRegionalDetailerQwen":   RPRegionalDetailerQwen,
    "RPTxt2ImgOpenAI":          _oai.RPTxt2ImgOpenAI,
    "RPTxt2ImgGemini":          _gem.RPTxt2ImgGemini,
    "RPTxt2ImgGrok":            _grok.RPTxt2ImgGrok,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RPPromptParser":           "RP Prompt Parser",
    "RPRatioParser":            "RP Ratio Parser",
    "RPKSampler":               "RP KSampler (SDXL)",
    "RPRegionalDetailer":       "RP Regional Detailer (SDXL)",
    "RPRegionalDetailerZImage": "RP Regional Detailer (Z-Image)",
    "RPConverter":             "RP Converter",
    "RPRegionalDetailerQwen":   "RP Regional Detailer (Qwen)",
    "RPTxt2ImgOpenAI":          "RP Txt2Img (OpenAI)",
    "RPTxt2ImgGemini":          "RP Txt2Img (Gemini)",
    "RPTxt2ImgGrok":            "RP Txt2Img (Grok)",
}

print(f"[ComfyUI_RP_Cast v{_VERSION}] {len(NODE_CLASS_MAPPINGS)} nodes registered: "
      f"{list(NODE_CLASS_MAPPINGS.keys())}")

WEB_DIRECTORY = "./web"

# ── Settings API route ────────────────────────────────────────────────────────
try:
    from server import PromptServer as _PromptServer
    from aiohttp import web as _web
    import json as _json

    @_PromptServer.instance.routes.get("/rp_cast/settings")
    async def _rp_cast_settings(request):
        """Return RP Cast settings stored in ComfyUI settings file."""
        setting_key = "ComfyUI-RP-Cast.Configuration.openai_api_key"
        api_key = ""
        try:
            settings_path = os.path.join(
                os.path.dirname(os.path.dirname(_HERE)),
                "user", "default", "comfy.settings.json"
            )
            if os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    all_settings = _json.load(f)
                api_key = all_settings.get(setting_key, "")
        except Exception as _se:
            print(f"[ComfyUI_RP_Cast] Failed to read settings: {_se}")
        return _web.Response(
            text=_json.dumps({setting_key: api_key}),
            content_type="application/json"
        )

    print(f"[ComfyUI_RP_Cast] Route registered: /rp_cast/settings")
except Exception as _e:
    print(f"[ComfyUI_RP_Cast] Route registration failed: {_e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
