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
Version: 0.5.40
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
_pp       = _load_module("nodes.node_rp_prompt_parser",    "nodes/node_rp_prompt_parser.py")
_ks       = _load_module("nodes.node_rp_ksampler",         "nodes/node_rp_ksampler.py")
_det      = _load_module("nodes.node_rp_detailer",         "nodes/node_rp_detailer.py")
_det_zi   = _load_module("nodes.node_rp_detailer_zimage",  "nodes/node_rp_detailer_zimage.py")
_ks_zi    = _load_module("nodes.node_rp_ksampler_zimage",  "nodes/node_rp_ksampler_zimage.py")
_ks_qw    = _load_module("nodes.node_rp_ksampler_qwen",    "nodes/node_rp_ksampler_qwen.py")
_det_qw   = _load_module("nodes.node_rp_detailer_qwen",    "nodes/node_rp_detailer_qwen.py")

# ── Node class references ──────────────────────────────────────────────────────
RPPromptParser           = _pp._RPPromptParser           if hasattr(_pp, "_RPPromptParser") else _pp.RPPromptParser
RPRatioParser            = _pp.RPRatioParser
RPKSampler               = _ks.RPKSampler
RPRegionalDetailer       = _det.RPRegionalDetailer
RPRegionalDetailerZImage = _det_zi.RPRegionalDetailerZImage
RPKSamplerZImage         = _ks_zi.RPKSamplerZImage
RPKSamplerQwen           = _ks_qw.RPKSamplerQwen
RPRegionalDetailerQwen   = _det_qw.RPRegionalDetailerQwen

# For test compatibility
_auto_aratios  = _pp._auto_aratios
_count_addcol  = _pp._count_addcol

# ── Registration ──────────────────────────────────────────────────────────────
_VERSION = "0.5.40"

NODE_CLASS_MAPPINGS = {
    "RPPromptParser":           RPPromptParser,
    "RPRatioParser":            RPRatioParser,
    "RPKSampler":               RPKSampler,
    "RPRegionalDetailer":       RPRegionalDetailer,
    "RPRegionalDetailerZImage": RPRegionalDetailerZImage,
    "RPKSamplerZImage":         RPKSamplerZImage,
    "RPKSamplerQwen":           RPKSamplerQwen,
    "RPRegionalDetailerQwen":   RPRegionalDetailerQwen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RPPromptParser":           "RP Prompt Parser",
    "RPRatioParser":            "RP Ratio Parser",
    "RPKSampler":               "RP KSampler",
    "RPRegionalDetailer":       "RP Regional Detailer",
    "RPRegionalDetailerZImage": "RP Regional Detailer (Z-Image)",
    "RPKSamplerZImage":         "RP KSampler (Z-Image)",
    "RPKSamplerQwen":           "RP KSampler (Qwen)",
    "RPRegionalDetailerQwen":   "RP Regional Detailer (Qwen)",
}

print(f"[ComfyUI_RP_Cast v{_VERSION}] {len(NODE_CLASS_MAPPINGS)} nodes registered: "
      f"{list(NODE_CLASS_MAPPINGS.keys())}")

WEB_DIRECTORY = "./web"

# ── Inject script tag into index.html for ComfyUI 0.17 compatibility ──────────
try:
    import server as _server
    from aiohttp import web as _web

    _JS_PATH = os.path.join(_HERE, "web", "js", "rp_nodes.js")

    # Serve rp_nodes.js via a dedicated route
    @_server.PromptServer.instance.routes.get("/rp_cast/rp_nodes.js")
    async def _rp_nodes_js(request):
        with open(_JS_PATH, "r", encoding="utf-8") as f:
            code = f.read()
        return _web.Response(text=code, content_type="application/javascript")

    # Override GET / to inject <script type="module"> into index.html
    async def _patched_root(request):
        # Find ComfyUI's web root
        import pathlib
        for res in _server.PromptServer.instance.app.router.routes():
            if hasattr(res, 'resource') and hasattr(res.resource, 'canonical'):
                if res.resource.canonical in ('/', ''):
                    break
        # Serve from ComfyUI's own web directory
        comfyui_web = pathlib.Path(_server.PromptServer.instance.app['comfyui_web_root'] if 'comfyui_web_root' in _server.PromptServer.instance.app else '.')
        index = comfyui_web / 'index.html'
        if not index.exists():
            # fallback: find ComfyUI root
            import comfy
            comfyui_web = pathlib.Path(comfy.__file__).parent.parent / 'web' / 'dist'
            index = comfyui_web / 'index.html'
        if index.exists():
            html = index.read_text(encoding='utf-8')
            tag = '<script type="module" src="/rp_cast/rp_nodes.js"></script>'
            if tag not in html:
                html = html.replace('</head>', f'{tag}\n</head>', 1)
            return _web.Response(text=html, content_type='text/html')
        raise _web.HTTPNotFound()

    print(f"[ComfyUI_RP_Cast] Routes registered: /rp_cast/rp_nodes.js")
except Exception as _e:
    print(f"[ComfyUI_RP_Cast] Route registration failed: {_e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
