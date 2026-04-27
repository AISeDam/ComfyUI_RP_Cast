"""
Regional Prompter - LoRA Manager (P2)
ComfyUI port based on AddNet/RP latent.py caching approach.

Original design (A1111):
  - _reload_loras_for_col(): LoRA switching per division
  - cache key: str "rp_col_{idx}_{loras}" (type identifies cache state)
  - cache hit: isinstance(key, str) and key == addnet_key

ComfyUI port:
  - Cannot replace LoRA weights directly (ModelPatcher based)
  - Pre-build ModelPatcher clone per division
  - Call separate apply_model per division in model_function_wrapper
  - cache: {addnet_key_str: (model_patcher_clone, clip_clone)}
"""
from __future__ import annotations
import os
from typing import Dict, Optional, Tuple

_COMFY_AVAILABLE = False
try:
    import comfy.sd
    import comfy.utils
    import folder_paths
    _COMFY_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────
# Build cache key (str type: port of A1111 addnet_key approach)
# ─────────────────────────────────────────────────────────
def _make_addnet_key(loras: Dict[str, float]) -> str:
    """
    LoRA set identifier (independent of division index).
    A1111: "rp_col_{loras_sorted}"
    Same LoRA set → same key → cache reuse
    """
    return "rp_col_" + ",".join(f"{k}:{v:.4f}" for k, v in sorted(loras.items()))


def _make_col_idx_key(col_idx: int, loras: Dict[str, float]) -> str:
    """
    Division unique identifier (includes and_idx).
    A1111: "rp_col_{and_idx}_{loras_sorted}"
    Used for full-hit check.
    """
    return f"rp_col_{col_idx}_" + ",".join(f"{k}:{v:.4f}" for k, v in sorted(loras.items()))


# Backward-compat alias
def _make_cache_key(col_idx: int, loras: Dict[str, float]) -> str:
    return _make_col_idx_key(col_idx, loras)


def _find_lora_path(lora_name: str) -> Optional[str]:
    if not _COMFY_AVAILABLE:
        return None
    try:
        # Normalize path separator (Windows \ → /)
        lora_name_norm = lora_name.replace("\\", "/")

        # Remove unnecessary prefix like models/loras/
        for prefix in ("models/loras/", "models\\loras\\", "loras/", "loras\\"):
            pfx = prefix.replace("\\", "/")
            if lora_name_norm.startswith(pfx):
                lora_name_norm = lora_name_norm[len(pfx):]
                break

        lora_stem_norm   = os.path.splitext(lora_name_norm)[0]   # remove extension
        lora_basename    = os.path.basename(lora_stem_norm)       # filename only
        lora_stem_lower  = lora_stem_norm.lower()
        lora_base_lower  = lora_basename.lower()

        # Get loras folder path list
        try:
            lora_dirs = folder_paths.get_folder_paths("loras")
        except Exception:
            lora_dirs = []

        # 1. folder_paths.get_filename_list method (ComfyUI newer versions with recursive support)
        try:
            candidates = folder_paths.get_filename_list("loras")
            for fname in candidates:
                fname_norm  = fname.replace("\\", "/")
                fstem_norm  = os.path.splitext(fname_norm)[0]
                # exact match including path
                if fstem_norm == lora_stem_norm or fstem_norm.lower() == lora_stem_lower:
                    full = folder_paths.get_full_path("loras", fname)
                    if full:
                        print(f"[RP LoRA] '{lora_name}' → '{fname}'")
                        return full
                # filename-only match
                fbase = os.path.basename(fstem_norm)
                if fbase == lora_basename or fbase.lower() == lora_base_lower:
                    full = folder_paths.get_full_path("loras", fname)
                    if full:
                        print(f"[RP LoRA] '{lora_name}' → '{fname}' (basename)")
                        return full
        except Exception:
            pass

        # 2. Direct recursive scan of loras folder
        #    → covers cases where get_filename_list does not support subdirs
        lora_exts = {".safetensors", ".pt", ".ckpt", ".bin"}
        for lora_dir in lora_dirs:
            if not os.path.isdir(lora_dir):
                continue
            for root, _dirs, files in os.walk(lora_dir):
                for fname in files:
                    if os.path.splitext(fname)[1].lower() not in lora_exts:
                        continue
                    full_path  = os.path.join(root, fname)
                    # relative path from lora_dir (/ separator)
                    rel_path   = os.path.relpath(full_path, lora_dir).replace("\\", "/")
                    rel_stem   = os.path.splitext(rel_path)[0]
                    file_stem  = os.path.splitext(fname)[0]

                    # exact match including path
                    if rel_stem == lora_stem_norm or rel_stem.lower() == lora_stem_lower:
                        print(f"[RP LoRA] '{lora_name}' → '{rel_path}' (recursive)")
                        return full_path
                    # filename-only match
                    if file_stem == lora_basename or file_stem.lower() == lora_base_lower:
                        print(f"[RP LoRA] '{lora_name}' → '{rel_path}' (recursive basename)")
                        return full_path

    except Exception as e:
        print(f"[RP LoRA] folder_paths error: {e}")

    print(f"[RP LoRA] '{lora_name}' file not found")
    return None


def _apply_loras(base_model, base_clip, loras: Dict[str, float]):
    """
    ComfyUI official method: comfy.sd.load_lora_for_models
    (same as nodes.py LoraLoader)
    """
    model = base_model.clone() if hasattr(base_model, "clone") else base_model
    clip  = base_clip.clone()  if (
        base_clip is not None and hasattr(base_clip, "clone")
    ) else base_clip

    for lora_name, weight in loras.items():
        path = _find_lora_path(lora_name)
        if path is None:
            continue
        try:
            lora_data = comfy.utils.load_torch_file(path, safe_load=True)
            model, clip = comfy.sd.load_lora_for_models(
                model, clip, lora_data,
                strength_model=float(weight),
                strength_clip=float(weight),
            )
            print(f"[RP LoRA] '{lora_name}' weight={weight:.3f} applied")
        except Exception as e:
            print(f"[RP LoRA] '{lora_name}' apply failed: {e}")

    return model, clip


class LoRADivisionManager:
    """
    ComfyUI port of AddNet/_reload_loras_for_col caching approach.

    Cache structure:
      _cache: {addnet_key(str): (ModelPatcher, CLIPModel)}
        - addnet_key is str type → identified by isinstance(key, str)
        - Same LoRA set reuses same ModelPatcher across divisions

      _loaded_col_idx: str|None
        - Last activated col_idx_key
        - For full-hit check (maps to A1111 _loaded_col_idx)

      _loaded_cache_key: str|None
        - Currently active addnet_key
        - None forces reload (already_loaded=False)
    """

    def __init__(self):
        self.col_lora_map:     Dict[int, Dict[str, float]] = {}
        self.division_count:   int        = 0
        self._base_model                  = None
        self._base_clip                   = None

        # Cache state variables (A1111 style)
        self._cache:           Dict[str, Tuple] = {}  # addnet_key → (model, clip)
        self._col_key_map:     Dict[int, str]   = {}  # col_idx → col_idx_key
        self._loaded_col_idx:  Optional[str]    = None  # last col_idx_key
        self._loaded_cache_key: Optional[str]   = None  # last addnet_key (str|None)

        # Runtime state
        self.u_count: int = 0
        self.step:    int = 0

    def setup(self, col_lora_map, division_count, base_model, base_clip,
              div_label_fn=None):
        self.col_lora_map    = col_lora_map
        self.division_count  = division_count
        self._base_model     = base_model
        self._base_clip      = base_clip
        self._div_label_fn   = div_label_fn  # col_idx → "DIV[r,c]" or "BASE"
        self.u_count         = 0
        self.step            = 0
        self._cache          = {}
        self._col_key_map    = {}
        self._loaded_col_idx  = None
        self._loaded_cache_key = None

    def _div_label(self, col_idx: int) -> str:
        """Convert col_idx to DIV[r,c] or BASE label."""
        if self._div_label_fn is not None:
            return self._div_label_fn(col_idx)
        return f"DIV[0,{col_idx}]"

    def prebuild_cache(self):
        """
        Pre-build ModelPatcher per division before sampling.

        Performs A1111 _reload_loras_for_col step 3 (cache miss → load)
        in advance during prebuild phase.

        Cache key: addnet_key (str) → identifiable as RP cache
        Same LoRA set reuses clone without rebuilding.
        """
        if not _COMFY_AVAILABLE:
            print("[RP LoRA] ComfyUI not available → skip cache")
            return

        print("[RP LoRA] === Division LoRA cache build ===")
        for col_idx in range(self.division_count):
            loras        = self.col_lora_map.get(col_idx, {})
            addnet_key   = _make_addnet_key(loras)
            col_idx_key  = _make_col_idx_key(col_idx, loras)
            self._col_key_map[col_idx] = col_idx_key
            label = self._div_label(col_idx)

            if addnet_key in self._cache:
                print(f"[RP LoRA] {label} same LoRA set reused: {addnet_key}")
                continue

            if not loras:
                self._cache[addnet_key] = (self._base_model, self._base_clip)
                print(f"[RP LoRA] {label} no LoRA")
            else:
                print(f"[RP LoRA] {label} build: {loras}")
                pm, pc = _apply_loras(self._base_model, self._base_clip, loras)
                self._cache[addnet_key] = (pm, pc)

        print("[RP LoRA] === cache complete ===")

    def get_model_for_division(self, col_idx: int) -> Tuple:
        """
        Cache decision logic of _reload_loras_for_col() (A1111 port).

        Step 1: col_idx_key full hit → return (model, clip)
        Step 2: same LoRA set, different division → update _loaded_col_idx and return
        Step 3: cache miss → build and return

        Returns: (model_patcher, clip_model)
        """
        loras       = self.col_lora_map.get(col_idx, {})
        addnet_key  = _make_addnet_key(loras)
        col_idx_key = self._col_key_map.get(col_idx)
        if col_idx_key is None:
            col_idx_key = _make_col_idx_key(col_idx, loras)

        cached_key  = self._loaded_cache_key  # str|None

        # Determine already_loaded (A1111 style)
        # str type check: handles mix of tuple(regular) and str(rp_col) cache entries
        already_loaded = (
            isinstance(cached_key, str) and cached_key == addnet_key
        )

        if already_loaded:
            if self._loaded_col_idx == col_idx_key:
                # Step 1: full hit
                pass
            else:
                # Step 2: same LoRA set, different division → update _loaded_col_idx only
                self._loaded_col_idx = col_idx_key
        else:
            # Step 3: cache miss → build (should already be built in prebuild)
            if addnet_key not in self._cache:
                if not loras:
                    self._cache[addnet_key] = (self._base_model, self._base_clip)
                elif _COMFY_AVAILABLE:
                    pm, pc = _apply_loras(self._base_model, self._base_clip, loras)
                    self._cache[addnet_key] = (pm, pc)
            self._loaded_cache_key = addnet_key
            self._loaded_col_idx   = col_idx_key

        return self._cache.get(addnet_key, (self._base_model, self._base_clip))

    def reset_step(self, step: int):
        self.step    = step
        self.u_count = 0

    def u_start(self) -> Optional[Tuple]:
        """
        Calculate division index on each UNet forward → call get_model_for_division.
        uncond (last slot) → returns None.
        """
        if self.division_count == 0:
            return None

        n_div   = self.division_count + 1  # divisions + uncond
        raw_idx = self.u_count % n_div

        div_idx = raw_idx

        self.u_count += 1

        if div_idx >= self.division_count:
            return None  # uncond → use base model

        return self.get_model_for_division(div_idx)

    # Backward compat
    def _get_cached(self, col_idx: int) -> Optional[Tuple]:
        return self.get_model_for_division(col_idx)
