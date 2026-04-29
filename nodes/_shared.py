"""
Shared utilities and constants for ComfyUI_RP_Cast nodes.
"""
import os, sys, re

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Capture the original torch.load before impact-subpack or similar patches it
import torch as _torch_module
_TORCH_LOAD_ORIG = _torch_module.load

from core.prompt_parser import parse_prompt, get_2d_structure
from core.regions       import parse_regions, make_filters
from core.filters       import RPLatentCompositor
from core.lora_manager  import LoRADivisionManager

_RE_LORA = re.compile(r"<lora:[^>]+>", re.IGNORECASE)

# ── Z-Image shared constants ───────────────────────────────────────────────────
_ZIMAGE_LATENT_CHANNELS  = 16   # Z-Image VAE channel count
_ZIMAGE_LATENT_BLOCK     = 8    # pixels per latent block (8×8)
_ZIMAGE_GRID_SIZE        = 32   # image size must be multiple of 32

# AuraFlow sigma schedule class — module-level (prevent circular ref)
try:
    import comfy.model_sampling as _ms_global
    class _ZImageAuraFlow(_ms_global.ModelSamplingDiscreteFlow, _ms_global.CONST):
        """AuraFlow sigma schedule for Z-Image Turbo."""
        pass
    del _ms_global
except Exception:
    _ZImageAuraFlow = None

def _enc(clip, tokens):
    """Encode tokens and detach/clone result to release TE model reference (prevents memory leak)."""
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    cond   = cond.detach().clone()   if cond   is not None else cond
    pooled = pooled.detach().clone() if pooled is not None else pooled
    return cond, pooled

try:
    import comfy.samplers as _cs
    _SAMPLERS   = _cs.KSampler.SAMPLERS
    _SCHEDULERS = _cs.KSampler.SCHEDULERS
    _COMFY_OK   = True
except Exception:
    _SAMPLERS   = ["euler","euler_ancestral","dpm_2","dpm_2_ancestral",
                   "heun","dpm_fast","dpm_adaptive","lms",
                   "dpmpp_2s_ancestral","dpmpp_sde","dpmpp_2m","ddim","uni_pc"]
    _SCHEDULERS = ["normal","karras","exponential","sgm_uniform",
                   "simple","ddim_uniform"]
    _COMFY_OK   = False
