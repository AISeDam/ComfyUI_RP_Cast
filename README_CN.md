# ComfyUI_RP_Cast

在 ComfyUI 中对**不同区域应用不同提示词**的节点集合。
支持左右、上下、网格等多种分割方式。
可用于 SDXL、Z-Image、Qwen 模型。

**版本：0.5.60** | [GitHub](https://github.com/AISeDam/ComfyUI_RP_Cast)

---

## 关于本程序

本项目参考了 hako-mikan 的
[sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter)，
并将部分功能**移植和改编**至 ComfyUI 环境。

可以在左侧生成角色 A、右侧生成角色 B，各自使用不同的提示词和 LoRA。
无需手动绘制蒙版，仅通过编写提示词即可划分区域。

提示词分割概念（ADDCOMM / ADDBASE / ADDCOL / ADDROW）、
divide_ratio 语法、区域 latent 混合算法均源自原作。
以下内容为适配 ComfyUI 环境及扩展模型支持而**新增或改编**：

- ComfyUI 节点架构（模块化 Python 节点，JS 前端扩展）
- Z-Image 支持（RPKSamplerZImage、RPRegionalDetailerZImage）
- Qwen 场景合成支持（RPKSamplerQwen、RPRegionalDetailerQwen）
- 区域 LoRA 分割管理
- 基于 YOLO 的 Regional Detailer 节点

> 原始提示词语法与算法详情请参考：
> **→ https://github.com/hako-mikan/sd-webui-regional-prompter**

---

## 安装

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/AISeDam/ComfyUI_RP_Cast
```

重启 ComfyUI 后，8 个节点将自动出现。

---

## 应该使用哪些节点？

### 第一步 — 始终从这两个节点开始

| 节点 | 作用 |
|------|------|
| `RPPromptParser` | 编写各区域提示词并自动分割 |
| `RPRatioParser` | 接收 `divide_ratio`、`divide_mode`、`threshold` 作为输入，输出区域数据（`regional_col_n_row`）和 threshold。连接在 RPPromptParser 之后 |

### 第二步 — 选择适合您模型的采样器和细化器

| 使用模型 | 采样器 | 细化器 |
|---------|-------|-------|
| SDXL / SD 1.x | `RP KSampler (SDXL)` | `RP Regional Detailer (SDXL)` |
| Z-Image / Qwen | *（使用ComfyUI标准KSampler）* | `RP Regional Detailer (Z-Image)` |
| Qwen | *（使用ComfyUI标准KSampler）* | `RP Regional Detailer (Qwen)` |

### 可选 — RP Converter

| 节点 | 功能 |
|------|------|
| `RP Converter` | 通过本地Ollama LLM（推荐`gemma3:12b`）将自然语言场景描述转换为RP结构提示词。按COSPLAY/人物关键词自动分割输入，支持风格关键词追加和每区域LoRA自动标签功能。 |

---

## SDXL / SD1.x — RPKSampler

连接 `EmptyLatentImage` 节点以指定画布大小。

| 控件 | 说明 |
|------|------|
| `use_base` | 与 RPPromptParser 的 `use_base` 设置相同 |
| `use_common` | 将 COMMON 提示词应用于所有区域 |
| `base_ratio` | BASE 与区域提示词的混合比例。`0.2` = BASE 20%。按区域指定：`0.2,0.3` |
| `lora_weight_adj` | 全局 LoRA 权重倍率。`100` = 原始，`50` = 减半，`200` = 两倍，`0` = 禁用 |

> **无法实现区域独立 LoRA 应用的技术原因**
>
> ComfyUI 的模型执行管道不提供在采样过程中安全地按区域切换 LoRA 权重的钩子。
> 内部 RPKSampler 注册了 `set_model_unet_function_wrapper` 回调，
> 拦截每次 UNet 调用并识别当前步骤处理的是哪个区域。
> 然而，在该时间点实际替换已应用 LoRA 的模型权重会导致
> conditioning 不一致和采样不稳定，
> 因此所有区域共用单一模型运行。
>
> 结果是，所有区域的 LoRA 会在采样前**平均合并**为一个模型统一应用。
> 可使用 `lora_weight_adj` 对合并后的整体权重进行批量调整。
>
> 区域角色特征的强化在 **Detailer 阶段**得到补充。
> 由于 YOLO 检测到的每个 bbox 会使用对应区域的提示词和 LoRA 单独进行修复（inpaint），
> 角色特有的特征在此阶段得到最有效的体现。

**基本连接:**

```
CheckpointLoader ──────────────────→ RPKSampler
EmptyLatentImage → RPKSampler (latent_image)
RPPromptParser → RPRatioParser ────→ RPKSampler
```

---

## Z-Image — RPKSamplerZImage

无需 `EmptyLatentImage` — 直接在节点上设置 `width`、`height`。

> **为什么使用合并提示词？**
> Z-Image 使用与 SDXL 不同的 attention 架构。
> SDXL 的区域采样在每个采样步骤中分别处理各区域的提示词，
> 并通过 denoised callback 方式按区域混合 latent。
> Z-Image 不支持这种逐步骤的钩子，因此无法应用区域 latent 混合。
> 取而代之，将所有区域提示词与自然语言位置标签合并为一个提示词
> （例：`(角色 A) on the left side and (角色 B) on the right side`）
> 进行**单次采样**。

| 控件 | 说明 |
|------|------|
| `width` / `height` | 输出图像尺寸（直接在节点上设置） |
| `steps` | 默认值：`8`。Z-Image Turbo 用较少步数即可运行。 |
| `cfg` | 默认值：`1.0`。Z-Image 基础模型使用 `cfg=1.0`。 |
| `shift` | Z-Image 噪声调度。默认值：`3.0`。推荐值：`3~6`。值越大，噪声越多地移向后续步骤。 |

**基本连接:**

```
ZImageCheckpointLoader ────────────→ RPKSamplerZImage
RPPromptParser → RPRatioParser ────→ RPKSamplerZImage
```

---

## Qwen — RPKSamplerQwen

与 Z-Image 方式相同，但使用叙事场景描述代替位置标签。

> **为什么使用合并提示词？**
> Qwen 使用相同的架构，因此无法使用逐步骤的区域钩子。
> Qwen 以场景叙述方式构建提示词：
> (`(角色 A) on the left side and (角色 B) on the right side,
> interacting naturally in the same scene, seamless composition`)
> 最适合**两个角色自然互动的场景**。

| 控件 | 说明 |
|------|------|
| `width` / `height` | 输出图像尺寸 |
| `steps` | 默认值：`20`。Qwen 推荐值：`15~20`。 |
| `cfg` | 默认值：`1.0`。Qwen 基础模型的推荐值。 |
| `shift` | Qwen 噪声调度。默认值：`3.0`。推荐值：`3~6`。 |

**基本连接:**

```
QwenCheckpointLoader ──────────────→ RPKSamplerQwen
RPPromptParser → RPRatioParser ────→ RPKSamplerQwen
```

---

## 细化器节点

在采样器运行后，对检测到的人物按区域进行单独修复（inpaint）。
需要 YOLO 模型（例：`bbox/person_yolov8m-seg.pt`）。`models/ultralytics/` 文件夹中的模型会自动检测。

| 节点 | 目标模型 |
|------|---------|
| `RP Regional Detailer (SDXL)` | SDXL / SD1.x |
| `RP Regional Detailer (Z-Image)` | Z-Image / Qwen |
| `RP Regional Detailer (Qwen)` | Qwen（委托给Z-Image细化器） |

**后备修复绘制（所有细化器通用）**

同一区域检测到多个人物时，面积最大的人物获得COL区域提示词。
其余落选人物使用**基础提示词**（`COMMON + BASE text`）进行修复绘制。

### 各 Detailer 的工作机制

**RP Regional Detailer (SDXL)**

1. **对全图运行一次YOLO**，通过bbox中心坐标与divide_ratio边界比较分配区域（ZImage方式）。
2. 同一区域多人时，面积最大者优先分配，其余进入基础提示词后备修复绘制队列。
3. 对每个bbox：掩码膨胀(dilation) + 模糊 → crop放大(LANCZOS)
   → VAE编码 → 使用区域提示词(COMMON + BASE + DIV组合)修复绘制
   → VAE解码 → 掩码混合合成回原图。
4. 修复绘制前独立加载该区域LoRA，**区域级LoRA独立应用在此阶段完全实现**。

*(原名: RPRegionalDetailer)*

1. **按区域蒙版分别运行 YOLO** — 裁剪每个区域后单独运行 YOLO，
   选取该区域内面积最大的人物 bbox。
2. 对每个检测到的 bbox：蒙版膨胀（dilation）+ 模糊 → crop 放大（LANCZOS）
   → VAE 编码 → 使用该区域提示词（COMMON + BASE + DIV 组合）修复
   → VAE 解码 → 合成到原始图像。
3. 每个区域修复前独立加载该区域的 LoRA，
   **区域独立 LoRA 应用在此阶段完全实现**。

**RPRegionalDetailerZImage** (Z-Image)

1. 对**全图执行一次 YOLO**，然后通过将每个检测到的人物 bbox 中心坐标
   与 divide_ratio 边界进行比较，自动分类到对应区域。
2. 可选地对每个 bbox 运行 **WD14 ONNX 性别分类**（boy/girl 分数），
   自动选择最合适的区域提示词。
3. 使用 Z-Image latent（16 通道，从 crop 尺寸自动生成）对每个 bbox 进行修复。
   **提示词编码使用与 RPRegionalDetailer 相同的 COMMON + BASE + DIV 组合方式**，
   而非 RPKSamplerZImage 的位置标签合并方式。
   区域独立 LoRA 应用与 SDXL 相同。

**RPRegionalDetailerQwen** (Qwen)

将所有处理委托给 RPRegionalDetailerZImage。
结构上的区别是，Qwen 的 VAE 可能返回 5D 张量 `[B, T, H, W, C]`，
将其规范化为 4D `[B, H, W, C]` 后传递给 RPRegionalDetailerZImage。
区域提示词**不进行场景叙述合并，直接原样传递**，
应用与 RPRegionalDetailerZImage 相同的 COMMON + BASE + DIV 编码。
---

## Txt2Img API 节点

使用外部图像生成 API 根据 Regional Prompter 提示词生成图像。
无需 GPU，提示词将被转换为自然语言后发送至 API。

> `RPPromptParser → RPRatioParser` 的连接方式与现有采样器相同。
> 将 `regional_col_n_row`（RPRatioParser 输出）和 `divide_mode`（RPPromptParser 输出）
> 连接到节点，即可自动计算网格布局。

### RP Txt2Img (OpenAI)

使用 [OpenAI Image Generation API](https://platform.openai.com/docs/api-reference/images)。

| 控件 | 说明 |
|------|------|
| `model` | `gpt-image-2` / `gpt-image-1.5` / `gpt-image-1` / `gpt-image-1-mini` |
| `size` | `1024×1024` / `1536×1024` / `1024×1536` / `auto` |
| `quality` | `auto` / `high` / `medium` / `low` |
| `debug` | 在控制台输出转换后的提示词 |

**设置键**: `ComfyUI-RP-Cast.Configuration.openai_api_key`

---

### RP Txt2Img (Gemini)

使用 [Google Gemini Image Generation API](https://ai.google.dev/gemini-api/docs/image-generation)。

| 控件 | 说明 |
|------|------|
| `model` | `gemini-3.1-flash-image-preview` / `gemini-3-pro-image-preview` / `gemini-2.5-flash-image` |
| `aspect_ratio` | `1:1` / `3:4` / `4:3` / `9:16` / `16:9` 等共10种 |
| `image_size` | `1K` / `2K` / `4K` |
| `debug` | 输出转换后的提示词及 API 响应信息 |

**设置键**: `ComfyUI-RP-Cast.Configuration.gemini_api_key`
API Key 申请: https://aistudio.google.com/apikey

---

### RP Txt2Img (Grok)

使用 [xAI Grok Image Generation API](https://docs.x.ai/developers/rest-api-reference/inference/images)。

| 控件 | 说明 |
|------|------|
| `model` | `grok-imagine-image` / `grok-imagine-image-pro` |
| `aspect_ratio` | `1:1` / `3:4` / `4:3` / `9:16` / `16:9` 等共14种 |
| `quality` | `low` / `medium` / `high` |
| `resolution` | `1k` / `2k` |
| `debug` | 在控制台输出转换后的提示词 |

**设置键**: `ComfyUI-RP-Cast.Configuration.grok_api_key`

---

## 更新历史

### v0.6.00 (2026-05-03)

**新增**
- `RP Converter` 节点：通过 Ollama LLM 将自然语言场景描述转换为 RP 结构提示词
  - `style_prompt`：Ollama 转换后在 ADDCOMM 与 ADDBASE 之间追加关键词
  - `lora_directory` + `lora_auto_apply`：每个 COL 区域自动添加随机 LoRA（含触发词，AGP 风格）
  - ComfyUI 启动时自动获取 Ollama 模型列表，未运行时在 execute 时重试
  - 推荐`gemma3:12b`；同时支持`llama3.2:3b`和`qwen3`（`qwen3`自动添加`/no_think`指令）
  - 转换完成后从 VRAM 卸载模型（`keep_alive=0`）
  - Ollama 输出无效时自动回退到 WD14 标签匹配（未安装时自动下载）
  - RP 结构无效（ADDCOMM/ADDBASE 缺失或重复）时重试 1 次
- `RPPromptParser`: 新增 `NL_prompts` STRING 输出（用于直接 CLIP 编码连接）
- `RPKSampler (SDXL)`：在 `lora_weight_adj` 上方新增 `steps_add_per_div`、`cfg_add_per_div` 控件
  - steps = steps + (n_div × steps_add_per_div), cfg = cfg + (n_div × cfg_add_per_div)
  - 默认值`0` — 未使用时无操作
- `RPRegionalDetailer` + `RPRegionalDetailerZImage`：为未分配人物添加 fallback 修复绘制
  - 未分配到任何 COL 区域的人物使用基础提示词（common + base_text）修复绘制
  - 采用与主 COL 循环相同的 crop→upscale→VAE 编码→采样→混合流水线
- `RPRegionalDetailer`：切换为 ZImage 风格的全图单次 YOLO + bbox 中心坐标区域分类

**变更**
- `RP KSampler` 显示名称 → `RP KSampler (SDXL)`
- `RP Regional Detailer` 显示名称 → `RP Regional Detailer (SDXL)`
- `RPRegionalDetailer`：区域分配改为 bbox 中心坐标与 divide_ratio 边界比较（ZImage 模式）
- `RP Converter`：默认模型改为`gemma3:12b`；System Prompt 重构为 PART A / PART B 明确分割格式
- `RP Converter`：Python 按 COSPLAY/人物关键词预分割输入后传递给模型
- `RP Converter`：所有模型应用`think: false`；`qwen3`额外自动添加`/no_think`指令
- `RP Converter`：所有 API payload 添加`context: []` — 每次请求清除对话历史
- `RP Converter`：retry 验证新增 ADDCOMM/ADDBASE 重复检测（除缺失外，重复也触发 retry）

**删除**
- 删除 `RP KSampler (Z-Image)` 节点
- 删除 `RP KSampler (Qwen)` 节点
- 删除 `RP KSampler (FLUX.2)` 节点
- 删除未使用 stub 文件：`node_rp_conditioning.py`、`node_rp_filter_maker.py`、`node_rp_ratio_parser.py`

### v0.5.60 (2026-04-30)

**Bug 修复**
- 修复 `RPKSampler` `use_base=False`: BASE 块文本(nolora_list[0])未正确跳过，导致区域映射错误
- 修复 `RPKSampler` `use_base=False`: 注入 null BASE anchor（空文本, strength=0.0）防止 COL 间 bleeding
- 修复 `RPKSampler` `use_base=False`: COL conditioning strength 强制为 1.0（use_base=False 时忽略 base_ratio）
- 修复 `RPKSampler` `use_base=False`: 应用 exclusive area 边界防止区域 bleed
- 修复 `RPKSampler` col_lora_map 索引: 添加 `lora_offset` 使 use_base=False 时 LoRA 正确映射
- 修复 `RPKSampler` `use_base=True`: BASE 文本现在正常编码（上次修复中错误设为 null 的问题）
- 全 sampler/detailer: `patches.clear()` 替换为 `patches = {}` 防止共享 dict 变异
- 全 sampler/detailer: sampling 前显式调用 `unpatch_model()` 释放残留 LowVramPatch
- `RPRegionalDetailerZImage`: LoRA 模型改为在 inpaint 循环前预构建（防止循环内 object_patch 堆叠）
- `RPRegionalDetailerZImage`: 始终使用 `model.clone()` 作为 fresh base 防止跨 run patch 污染

**变更**
- `core/prompt_parser.py`: `subprompts_raw` 现在只包含 col 文本（不含 common）
- `core/lora_manager.py`: `setup()`、`prebuild_cache()`、`get_model_for_division()` 添加 `lora_offset` 参数
- `RPKSampler`: CLIP 编码前将 common 文本 pre-merge 到各 DIV，仅执行一次（`_null_base` 策略）


### v0.5.59 (2026-04-29)

**新增**
- `RP Txt2Img (OpenAI)` 节点: 通过 GPT Image API 生成图像 (gpt-image-2/1.5/1/1-mini)
- `RP Txt2Img (Gemini)` 节点: 通过 Google Gemini generateContent API 生成图像
- `RP Txt2Img (Grok)` 节点: 通过 xAI Grok Image API 生成图像
- 在 ComfyUI Settings 中添加 API 密钥设置项 (openai / gemini / grok)
- 所有 Txt2Img 节点添加 `regional_col_n_row`、`divide_mode` optional dot 输入
- `_rp_txt2img_common.py`: Txt2Img 节点共用工具模块
- Gemini 节点添加 `image_size` 参数 (`1K` / `2K` / `4K`)
- Grok 节点添加 `quality`、`resolution` 参数
- 所有 Txt2Img 节点完全独立运行（节点间无相互依赖）

**Bug 修复**
- 修复 `_convert_rp_to_natural` IndexError: Case C (1D Horizontal) cells 为 4-tuple 但代码访问了 index 4
- 修复 Vertical 网格位置标签: `divide_mode=Vertical` 时 rows/cols 互换的问题
- 修复 Horizontal 非对称网格 (如 2+1 行): 单行区域现在标记为 `lower-center` 而非 `lower-left`
- 修复 `base_ratio` 未应用于 BASE conditioning strength 的 Bug（原来硬编码为 1.0）

**变更**
- `parse_prompt()`: `subprompts_raw` 现在只包含 col 文本，common 不再在此处合并
- `RPKSampler`: 在 CLIP 编码前将 common 文本 pre-merge 到各 DIV 文本中，仅执行一次

---


---

## 许可证

AGPL-3.0. 请参阅 [LICENSE](LICENSE)。

## 参考

- [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter) — 原始算法 (hako-mikan)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) — Detailer 模式
- [ComfyUI-ZImagePowerNodes](https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes) — Z-Image 采样器
