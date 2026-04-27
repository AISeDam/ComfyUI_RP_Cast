# ComfyUI_RP_Cast

Generate images with **different prompts per region** — left/right, top/bottom, or grid layouts.
Supports SDXL, Z-Image, and Qwen models.

**Version: 0.5.40** | [GitHub](https://github.com/AISeDam/ComfyUI_RP_Cast)

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

### Step 2 — Pick the sampler that matches your model

| Your model | Use this sampler | Use this detailer |
|------------|-----------------|------------------|
| SDXL / SD 1.x | `RPKSampler` | `RPRegionalDetailer` |
| Z-Image | `RPKSamplerZImage` | `RPRegionalDetailerZImage` |
| Qwen | `RPKSamplerQwen` | `RPRegionalDetailerQwen` |

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

## Z-Image — RPKSamplerZImage

No `EmptyLatentImage` needed — set `width` and `height` directly on the node.

> **Why one merged prompt?**
> Z-Image uses a different attention architecture from SDXL.
> SDXL's region-blending works by processing each region's prompt separately
> per sampling step and blending the resulting latents (denoised callback).
> Z-Image does not expose the per-step hooks needed for this,
> so region-based latent blending cannot be applied.
> Instead, all region prompts are merged into a single prompt
> with natural language position labels
> (e.g. `(character A) on the left side and (character B) on the right side`)
> and sampled in **one pass**.

| Widget | What to set |
|--------|-------------|
| `width` / `height` | Output image size (set directly on node) |
| `steps` | Default: `8`. Z-Image Turbo works well with fewer steps. |
| `cfg` | Default: `1.0`. Z-Image based models typically use `cfg=1.0`. |
| `shift` | Z-Image noise schedule. Default: `3.0`. Recommended `3~6` for Z-Image Turbo. Higher = more noise in later steps. |

**Basic connection:**

```
ZImageCheckpointLoader ────────────→ RPKSamplerZImage
RPPromptParser → RPRatioParser ────→ RPKSamplerZImage
```

---

## Qwen — RPKSamplerQwen

Same approach as Z-Image, but uses a narrative scene description instead of position labels.

> **Why one merged prompt?**
> Same reason as Qwen uses the same architecture — per-step region hooks are not available.
> Qwen builds a scene narrative
> (`(character A) on the left side and (character B) on the right side,
> interacting naturally in the same scene, seamless composition`)
> designed for **2-character scenes** where natural interaction matters.

| Widget | What to set |
|--------|-------------|
| `width` / `height` | Output image size |
| `steps` | Default: `20`. Recommended `15~20` steps for Qwen. |
| `cfg` | Default: `1.0`. Recommended for Qwen based models. |
| `shift` | Qwen noise schedule. Default: `3.0`. Start at `3~6`. |

**Basic connection:**

```
QwenCheckpointLoader ──────────────→ RPKSamplerQwen
RPPromptParser → RPRatioParser ────→ RPKSamplerQwen
```


---

## Detailer nodes

Run after the sampler to refine each detected person separately using inpainting.
Requires a YOLO model (e.g. `bbox/person_yolov8m-seg.pt`). The model list is auto-detected from your `models/ultralytics/` folder.

| Node | For which model |
|------|----------------|
| `RPRegionalDetailer` | SDXL / SD1.x |
| `RPRegionalDetailerZImage` | Z-Image |
| `RPRegionalDetailerQwen` | Qwen |

### How each Detailer works

**RPRegionalDetailer** (SDXL / SD1.x)

1. Runs YOLO detection **independently per region** — each region mask is cropped
   and YOLO is applied separately to select the largest person within that region.
2. For each detected bbox: applies mask dilation + blur → upscales crop (LANCZOS)
   → VAE encode → inpaints with the region's own prompt (COMMON + BASE + DIV combined)
   → VAE decode → pastes back.
3. Each region loads its own LoRA independently before inpainting — true per-region
   LoRA isolation is fully achieved at this stage.

**RPRegionalDetailerZImage** (Z-Image)

1. Runs YOLO on the **full image once**, then assigns each detected person to a region
   by comparing the bbox center coordinate to the divide_ratio boundaries.
2. Optionally runs **WD14 ONNX gender classification** (boy/girl score) per detected
   bbox to select the best-matching region prompt automatically.
3. Inpaints each bbox with a Z-Image latent (16-channel, auto-generated from crop size).
   **Prompt encoding uses the same COMMON + BASE + DIV combination as RPRegionalDetailer**
   — not the merged scene-narrative format used by RPKSamplerZImage.
   Per-region LoRA isolation is identical to SDXL.

**RPRegionalDetailerQwen** (Qwen)

Delegates entirely to RPRegionalDetailerZImage with one structural difference:
Qwen's VAE may return a 5D tensor `[B, T, H, W, C]`, which is normalized to 4D
`[B, H, W, C]` before being passed to RPRegionalDetailerZImage.
Region prompts are passed through **as-is without any scene-narrative merging** —
the same COMMON + BASE + DIV encoding used by RPRegionalDetailerZImage applies here too.


---

## License

AGPL-3.0. See [LICENSE](LICENSE).

## Credits

- [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter) — original algorithm (hako-mikan)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) — detailer pattern
- [ComfyUI-ZImagePowerNodes](https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes) — Z-Image sampler
