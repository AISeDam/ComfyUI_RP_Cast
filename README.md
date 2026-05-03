# ComfyUI_RP_Cast

Generate images with **different prompts per region** — left/right, top/bottom, or grid layouts.
Supports SDXL, Z-Image, and Qwen models.

**Version: 0.6.00** | [GitHub](https://github.com/AISeDam/ComfyUI_RP_Cast)

---

## About

This project is a **partial port and adaptation** of
[sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter)
by hako-mikan, rewritten for the ComfyUI environment.

Draw a scene where the left side has one character and the right side has another —
each with their own prompt, LoRA, and style. No masking or manual selection needed.

The core prompt-division concept (ADDCOMM / ADDBASE / ADDCOL / ADDROW),
divide_ratio syntax, and regional latent blending algorithm are derived from the original work.
The following have been **added or modified** for ComfyUI and extended model support:

- ComfyUI node architecture (modular Python nodes, JS frontend extension)
- Z-Image support (RPKSamplerZImage, RPRegionalDetailerZImage)
- Qwen scene-composition support (RPKSamplerQwen, RPRegionalDetailerQwen)
- LoRA division management per region
- YOLO-based regional detailer nodes

> For the original prompt syntax and algorithm details, refer to:
> **→ https://github.com/hako-mikan/sd-webui-regional-prompter**

---

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/AISeDam/ComfyUI_RP_Cast
```

Restart ComfyUI — 8 nodes will appear automatically.

---

## Which nodes do I use?

### Step 1 — Always start with these two

| Node | What it does |
|------|--------------|
| `RPPromptParser` | Write your multi-region prompt here. Splits it into regions automatically. |
| `RPRatioParser` | Takes `divide_ratio`, `divide_mode`, and `threshold` as input. Outputs region data (`regional_col_n_row`) and threshold. Connect after RPPromptParser. |

### Step 2 — Pick the sampler and detailer that matches your model

| Your model | Sampler | Detailer |
|------------|---------|----------|
| SDXL / SD 1.x | `RP KSampler (SDXL)` | `RP Regional Detailer (SDXL)` |
| Z-Image / Qwen | *(use standard ComfyUI KSampler)* | `RP Regional Detailer (Z-Image)` |
| Qwen | *(use standard ComfyUI KSampler)* | `RP Regional Detailer (Qwen)` |

### Optional — RP Converter

| Node | What it does |
|------|--------------|
| `RP Converter` | Converts a natural language scene description into an RP-structured prompt using a local Ollama LLM (`gemma3:12b` recommended). Splits input at COSPLAY/person keyword boundary, appends style keywords, and adds random LoRA tags per COL section. |

### Alternative — Generate via external API (no GPU needed)

| API | Node | API Key |
|-----|------|---------|
| OpenAI GPT Image | `RP Txt2Img (OpenAI)` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Google Gemini | `RP Txt2Img (Gemini)` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| xAI Grok | `RP Txt2Img (Grok)` | [x.ai](https://console.x.ai/) |

Set your API key in **Settings → ComfyUI-RP-Cast → Configuration**.

---

## SDXL / SD1.x — RPKSampler

Connect an `EmptyLatentImage` node for the canvas size.

| Widget | What to set |
|--------|-------------|
| `use_base` | Same as `use_base` in RPPromptParser |
| `use_common` | Applies the COMMON prompt to all regions |
| `base_ratio` | How much BASE blends into each region. `0.2` = 20% base. Try `0.2,0.3` for per-region control. |
| `lora_weight_adj` | Scale all LoRA weights. `100` = unchanged, `50` = half, `200` = double. `0` = off. |

> **Why per-region LoRA isolation is not fully implemented**
>
> ComfyUI's model execution pipeline does not support safely swapping LoRA weights
> per-region mid-sample. Internally, RPKSampler registers a
> `set_model_unet_function_wrapper` callback that intercepts each UNet call and
> identifies which region (division) is active at that step. However, actually
> hot-swapping the LoRA-patched model weights at that point causes conditioning
> misalignment and sampling instability, so the implementation falls back to a
> single shared model for all regions.
>
> As a result, all region LoRAs are **averaged into one** before sampling,
> applied uniformly across the entire image.
> Use `lora_weight_adj` to scale this combined weight up or down globally.
>
> True per-region character reinforcement is handled at the **Detailer stage**:
> each YOLO-detected bbox is inpainted individually with its own region prompt
> and LoRA, which is where character-specific features are most effectively
> strengthened.

**Basic connection:**

```
CheckpointLoader ──────────────────→ RPKSampler
EmptyLatentImage → RPKSampler (latent_image)
RPPromptParser → RPRatioParser → RPKSampler
```

---

## Detailer nodes

Run after the sampler to refine each detected person separately using inpainting.
Requires a YOLO model (e.g. `bbox/person_yolov8m-seg.pt`). The model list is auto-detected from your `models/ultralytics/` folder.

| Node | For which model |
|------|----------------|
| `RP Regional Detailer (SDXL)` | SDXL / SD1.x |
| `RP Regional Detailer (Z-Image)` | Z-Image / Qwen |
| `RP Regional Detailer (Qwen)` | Qwen (delegates to Z-Image detailer) |

**Fallback inpainting (all detailers)**

When multiple persons are detected in the same region, the largest-area person is
assigned to the COL region prompt. All remaining (displaced) persons are inpainted
using the **base prompt** (`COMMON + BASE text`) so they still receive refinement.

### How each Detailer works

**RP Regional Detailer (SDXL)**

1. Runs YOLO on the **full image once**, then assigns each detected person to a region
   by comparing the **bbox center coordinate** to the divide_ratio boundaries (ZImage-style).
   Supports Horizontal, Vertical, and 2D grid layouts.
2. If multiple persons land in the same region, the largest-area person wins;
   the rest are queued for fallback inpainting with the base prompt.
3. For each assigned bbox: applies mask dilation + blur → upscales crop (LANCZOS)
   → VAE encode → inpaints with the region's own prompt (COMMON + BASE + DIV combined)
   → VAE decode → blends back with mask.
4. Each region loads its own LoRA independently before inpainting —
   true per-region LoRA isolation is fully achieved at the Detailer stage.

**RP Regional Detailer (Z-Image)**

1. Runs YOLO on the **full image once**, assigns persons to regions by bbox center
   coordinate vs divide_ratio boundaries.
2. Optionally runs **WD14 ONNX gender classification** (boy/girl score) per detected
   bbox to automatically select the best-matching region prompt.
3. Inpaints each bbox with a Z-Image latent (16-channel, crop-size auto-generated).
   Prompt encoding uses the same **COMMON + BASE + DIV** combination as the SDXL detailer
   — not the merged scene-narrative format.
   Per-region LoRA isolation is identical to the SDXL detailer.
4. Displaced persons (not assigned to any COL) are inpainted with the base prompt.

**RP Regional Detailer (Qwen)**

Delegates entirely to `RP Regional Detailer (Z-Image)` with one structural difference:
Qwen's VAE may return a 5D tensor `[B, T, H, W, C]`, which is normalized to 4D
`[B, H, W, C]` before being passed through.
Region prompts use the same **COMMON + BASE + DIV** encoding — no scene-narrative merging.


---

## Txt2Img API Nodes

Generate images from Regional Prompter prompts using external image generation APIs.
No GPU required — prompts are converted to natural language and sent to the API.

> These nodes share the same `RPPromptParser → RPRatioParser` connection as other samplers.
> Connect `regional_col_n_row` (from `RPRatioParser`) and `divide_mode` (from `RPPromptParser`)
> to the node for automatic grid layout calculation.

### Prompt conversion

The RP prompt structure is converted to natural language before sending to the API:

```
(scene text), (1girl) on the upper-left side, (1boy) on the upper-right side,
interacting naturally in the same scene, seamless composition, (style tags)
```

Position labels are computed from the actual region grid:
- **Horizontal 2×2**: upper-left / upper-right / lower-left / lower-right
- **Vertical 2×2** (C0=1row, C1=2rows, C2=1row): left / upper-center / lower-center / right
- **Asymmetric** (2+1 rows): upper-left / upper-right / lower-center

---

### RP Txt2Img (OpenAI)

Generates images using the [OpenAI Image Generation API](https://platform.openai.com/docs/api-reference/images).

| Widget | Description |
|--------|-------------|
| `model` | `gpt-image-2` / `gpt-image-1.5` / `gpt-image-1` / `gpt-image-1-mini` |
| `size` | `1024×1024` / `1536×1024` / `1024×1536` / `auto` |
| `quality` | `auto` / `high` / `medium` / `low` |
| `debug` | Print converted prompt to console |

**Optional inputs** (connect from RPRatioParser / RPPromptParser):
`regional_col_n_row` · `divide_mode` · `background`

**Settings key**: `ComfyUI-RP-Cast.Configuration.openai_api_key`

---

### RP Txt2Img (Gemini)

Generates images using the [Google Gemini Image Generation API](https://ai.google.dev/gemini-api/docs/image-generation).

| Widget | Description |
|--------|-------------|
| `model` | `gemini-3.1-flash-image-preview` / `gemini-3-pro-image-preview` / `gemini-2.5-flash-image` |
| `aspect_ratio` | `1:1` / `3:4` / `4:3` / `9:16` / `16:9` / `4:5` / `5:4` / `2:3` / `3:2` / `21:9` |
| `image_size` | `1K` / `2K` / `4K` |
| `debug` | Print converted prompt and API response info to console |

**Settings key**: `ComfyUI-RP-Cast.Configuration.gemini_api_key`

---

### RP Txt2Img (Grok)

Generates images using the [xAI Grok Image Generation API](https://docs.x.ai/developers/rest-api-reference/inference/images).

| Widget | Description |
|--------|-------------|
| `model` | `grok-imagine-image` / `grok-imagine-image-pro` |
| `aspect_ratio` | `1:1` / `3:4` / `4:3` / `9:16` / `16:9` / `2:3` / `3:2` / `9:19.5` / `19.5:9` / `9:20` / `20:9` / `1:2` / `2:1` / `auto` |
| `quality` | `low` / `medium` / `high` |
| `resolution` | `1k` / `2k` |
| `debug` | Print converted prompt to console |

**Settings key**: `ComfyUI-RP-Cast.Configuration.grok_api_key`

---

## Update History

### v0.6.00 (2026-05-03)

**Added**
- `RP Converter` node: converts natural language scene descriptions to RP-structured prompts via Ollama LLM
  - `style_prompt`: keywords appended between ADDCOMM and ADDBASE after Ollama conversion
  - `lora_directory` + `lora_auto_apply`: auto-append random LoRA tags (with trigger words) per COL section (AGP-style)
  - Ollama model list loaded at ComfyUI startup; retried at execute time if initially unavailable
  - `gemma3:12b` recommended; `llama3.2:3b` and `qwen3` also supported (`qwen3` uses `/no_think` directive)
  - Model unloaded from VRAM after conversion (`keep_alive=0`)
  - WD14 tag-matching fallback when Ollama output is invalid (auto-downloaded if missing)
  - 1-retry on invalid RP structure (missing or duplicate ADDCOMM/ADDBASE)
- `RPPromptParser`: `NL_prompts` STRING output added for direct CLIP encode use
- `RPKSampler (SDXL)`: `steps_add_per_div` and `cfg_add_per_div` widgets added above `lora_weight_adj`
  - Total steps = steps + (n_div × steps_add_per_div); total cfg = cfg + (n_div × cfg_add_per_div)
  - Default value `0` — no-op when unused
- `RPRegionalDetailer` + `RPRegionalDetailerZImage`: fallback inpainting for displaced persons
  - Persons not selected for any COL region are inpainted with base prompt (common + base_text)
  - Uses full crop→upscale→VAE encode→sample→blend pipeline (same as main COL loop)
- `RPRegionalDetailer`: switched to ZImage-style single full-image YOLO pass with bbox-center region assignment
  - Replaces per-region crop YOLO; supports Horizontal/Vertical and 2D grid layouts

**Changed**
- `RP KSampler` display name changed to `RP KSampler (SDXL)`
- `RP Regional Detailer` display name changed to `RP Regional Detailer (SDXL)`
- `RPRegionalDetailer`: region assignment now uses bbox center coordinate vs divide_ratio boundaries (ZImage pattern)
- `RP Converter`: default model set to `gemma3:12b`; System Prompt restructured to PART A / PART B explicit split format
- `RP Converter`: User Prompt pre-splits input at COSPLAY/person keyword boundary in Python before passing to model
- `RP Converter`: `think: false` applied to all models; `qwen3` additionally receives `/no_think` directive
- `RP Converter`: `context: []` added to all API payloads — conversation history cleared on every request
- `RP Converter`: retry validation extended to detect duplicate ADDCOMM/ADDBASE (in addition to missing)

**Removed**
- `RP KSampler (Z-Image)` node removed (use `RPRegionalDetailerZImage` + standard KSampler instead)
- `RP KSampler (Qwen)` node removed
- `RP KSampler (FLUX.2)` node removed
- Deprecated stub files removed: `node_rp_conditioning.py`, `node_rp_filter_maker.py`, `node_rp_ratio_parser.py`

### v0.5.60 (2026-04-30)

**Bug Fixes**
- Fixed `RPKSampler` `use_base=False`: BASE block text (nolora_list[0]) is now correctly skipped; previously included as COL[0] causing region mis-mapping
- Fixed `RPKSampler` `use_base=False`: inject null BASE anchor (empty text, strength=0.0) to prevent COL-to-COL bleeding via area conditioning
- Fixed `RPKSampler` `use_base=False`: COL conditioning `strength` forced to 1.0 (base_ratio now ignored when use_base=False)
- Fixed `RPKSampler` `use_base=False`: exclusive area boundaries applied (no overlap) to prevent region bleed
- Fixed `RPKSampler` col_lora_map index: `lora_offset` added to LoRADivisionManager so LoRA maps correctly when use_base=False
- Fixed `RPKSampler` `use_base=True`: BASE text now encoded correctly (was incorrectly set to null in previous fix attempt)
- Fixed all samplers/detailers: `patches.clear()` replaced with `patches = {}` to prevent shared dict mutation across runs
- Fixed all samplers/detailers: `unpatch_model()` called before sampling to release stale LowVramPatch registrations
- Fixed `RPRegionalDetailerZImage`: LoRA models pre-built before inpaint loop (not inside loop) to prevent object_patch stacking
- Fixed `RPRegionalDetailerZImage`: always use `model.clone()` as fresh base to prevent cross-run patch contamination

**Changed**
- `core/prompt_parser.py`: `subprompts_raw` now contains col-only text; common text no longer pre-merged here
- `core/lora_manager.py`: added `lora_offset` parameter to `setup()`, `prebuild_cache()`, `get_model_for_division()`
- `RPKSampler`: common text pre-merged into each DIV encode text exactly once before CLIP encoding (`_null_base` strategy)


### v0.5.59 (2026-04-29)

**Added**
- `RP Txt2Img (OpenAI)` node: generate images via GPT Image API (gpt-image-2/1.5/1/1-mini)
- `RP Txt2Img (Gemini)` node: generate images via Google Gemini generateContent API
- `RP Txt2Img (Grok)` node: generate images via xAI Grok Image API
- API key settings registration in ComfyUI Settings (openai / gemini / grok)
- `regional_col_n_row` and `divide_mode` optional dot inputs on all Txt2Img nodes
- `_rp_txt2img_common.py`: shared conversion utilities for all Txt2Img nodes
- `image_size` parameter for Gemini node (`1K` / `2K` / `4K`)
- `quality` and `resolution` parameters for Grok node
- All Txt2Img nodes fully independent (no cross-node dependency)

**Bug Fixes**
- Fixed `_convert_rp_to_natural` IndexError: Case C (1D Horizontal) cells were 4-tuple but code accessed index 4
- Fixed Vertical grid position labels: rows/cols were swapped when `divide_mode=Vertical`
- Fixed Horizontal asymmetric grid (e.g. 2+1 rows): single-row region now labeled `lower-center` instead of `lower-left`
- Fixed `base_ratio` not applied to BASE conditioning strength (was hardcoded to 1.0)

**Changed**
- `parse_prompt()`: `subprompts_raw` now contains col-only text; common is no longer pre-merged here
- `RPKSampler`: common text is pre-merged into each DIV encode text exactly once before CLIP encoding

---

## License

AGPL-3.0. See [LICENSE](LICENSE).

## Credits

- [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter) — original algorithm (hako-mikan)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) — detailer pattern
- [ComfyUI-ZImagePowerNodes](https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes) — Z-Image sampler
