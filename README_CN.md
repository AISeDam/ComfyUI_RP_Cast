# ComfyUI_RP_Cast

在 ComfyUI 中对**不同区域应用不同提示词**的节点集合。
支持左右、上下、网格等多种分割方式。
可用于 SDXL、Z-Image、Qwen 模型。

**版本：0.5.40** | [GitHub](https://github.com/AISeDam/ComfyUI_RP_Cast)

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

### 第二步 — 根据模型选择采样器

| 使用模型 | 采样器 | 细化器 |
|---------|-------|-------|
| SDXL / SD 1.x | `RPKSampler` | `RPRegionalDetailer` |
| Z-Image | `RPKSamplerZImage` | `RPRegionalDetailerZImage` |
| Qwen | `RPKSamplerQwen` | `RPRegionalDetailerQwen` |

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
| `RPRegionalDetailer` | SDXL / SD1.x |
| `RPRegionalDetailerZImage` | Z-Image |
| `RPRegionalDetailerQwen` | Qwen |

### 各 Detailer 的工作机制

**RPRegionalDetailer** (SDXL / SD1.x)

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

## 许可证

AGPL-3.0. 请参阅 [LICENSE](LICENSE)。

## 参考

- [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter) — 原始算法 (hako-mikan)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) — Detailer 模式
- [ComfyUI-ZImagePowerNodes](https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes) — Z-Image 采样器
