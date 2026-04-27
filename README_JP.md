# ComfyUI_RP_Cast

ComfyUIで**領域ごとに異なるプロンプト**を適用するノード集です。
左右・上下・グリッド分割などに対応しています。
SDXL、Z-Image、Qwen モデルで使用できます。

**バージョン: 0.5.40** | [GitHub](https://github.com/AISeDam/ComfyUI_RP_Cast)

---

## このプログラムについて

このプロジェクトは hako-mikan による
[sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter) を参照し、
一部の機能を ComfyUI 環境向けに**移植・改変**したプログラムです。

左側にキャラクター A、右側にキャラクター B を、それぞれ異なるプロンプトと LoRA で生成できます。
マスクや手動選択なしに、プロンプトを記述するだけで領域を分割できます。

プロンプト分割の概念（ADDCOMM / ADDBASE / ADDCOL / ADDROW）、
divide_ratio 構文、領域ごとの latent ブレンディングアルゴリズムは原作から派生しています。
以下の項目は ComfyUI 環境および拡張モデルサポートのために**追加・改変**されました：

- ComfyUI ノードアーキテクチャ（モジュール化 Python ノード、JS フロントエンド拡張）
- Z-Image サポート（RPKSamplerZImage、RPRegionalDetailerZImage）
- Qwen シーン構成サポート（RPKSamplerQwen、RPRegionalDetailerQwen）
- 領域ごとの LoRA 分割管理
- YOLO ベースの Regional Detailer ノード

> 元のプロンプト構文とアルゴリズムの詳細は以下を参照してください：
> **→ https://github.com/hako-mikan/sd-webui-regional-prompter**

---

## インストール

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/AISeDam/ComfyUI_RP_Cast
```

ComfyUI を再起動すると 8 つのノードが自動的に表示されます。

---

## どのノードを使えばいいですか？

### ステップ 1 — 常にこの 2 つのノードから始める

| ノード | 役割 |
|--------|------|
| `RPPromptParser` | 領域ごとのプロンプトを記述し、自動的に分割します |
| `RPRatioParser` | `divide_ratio`、`divide_mode`、`threshold` を入力として受け取り、領域データ（`regional_col_n_row`）と threshold を出力します。RPPromptParser の後に接続します |

### ステップ 2 — モデルに合ったサンプラーを選択

| 使用モデル | サンプラー | ディテイラー |
|-----------|-----------|------------|
| SDXL / SD 1.x | `RPKSampler` | `RPRegionalDetailer` |
| Z-Image | `RPKSamplerZImage` | `RPRegionalDetailerZImage` |
| Qwen | `RPKSamplerQwen` | `RPRegionalDetailerQwen` |

---

## SDXL / SD1.x — RPKSampler

`EmptyLatentImage` ノードを接続してキャンバスサイズを指定します。

| ウィジェット | 説明 |
|------------|------|
| `use_base` | RPPromptParser の `use_base` と同じ値に設定 |
| `use_common` | COMMON プロンプトをすべての領域に適用 |
| `base_ratio` | BASE と領域プロンプトの混合比率。`0.2` = BASE 20%。領域ごとの指定：`0.2,0.3` |
| `lora_weight_adj` | 全 LoRA 重みの倍率。`100` = 元のまま、`50` = 半分、`200` = 2倍、`0` = 無効 |

> **領域ごとの LoRA 独立適用が実装できない技術的理由**
>
> ComfyUI のモデル実行パイプラインは、サンプリング中に領域ごとに LoRA の重みを
> 安全に切り替えるフックを提供していません。
> 内部的に RPKSampler は `set_model_unet_function_wrapper` コールバックを登録して
> 各 UNet 呼び出しをインターセプトし、そのステップでどの領域が処理されているかを識別します。
> しかし実際に LoRA が適用されたモデルの重みをその時点で切り替えると
> conditioning の不一致とサンプリングの不安定が発生するため、
> すべての領域に単一の共有モデルを使用する方式で動作します。
>
> 結果として、すべての領域の LoRA は**平均マージ**されてサンプリング前に
> 1 つのモデルとして統合適用されます。
> `lora_weight_adj` でこのマージされた全体の重みを一括調整できます。
>
> 領域ごとのキャラクター特性の強化は **Detailer ステージ**で補完されます。
> YOLO で検出された各 bbox を該当領域のプロンプトと LoRA で個別にインペイントするため、
> キャラクター固有の特徴がこの段階で最も効果的に反映されます。

**基本的な接続:**

```
CheckpointLoader ──────────────────→ RPKSampler
EmptyLatentImage → RPKSampler (latent_image)
RPPromptParser → RPRatioParser ────→ RPKSampler
```

---

## Z-Image — RPKSamplerZImage

`EmptyLatentImage` 不要 — ノード上で `width`、`height` を直接設定します。

> **なぜ統合プロンプトを使用するのですか？**
> Z-Image は SDXL とは異なる attention アーキテクチャを使用しています。
> SDXL の領域ごとのサンプリングは、各サンプリングステップで領域ごとのプロンプトを
> 個別に処理し、その結果の latent を領域ごとにブレンドする方式（denoised callback）で動作します。
> Z-Image はこのステップごとのフックをサポートしていないため、
> 領域ごとの latent ブレンディングを適用できません。
> 代わりに、すべての領域プロンプトを自然言語の位置ラベルとともに 1 つのプロンプトにマージして
> （例：`(キャラクター A) on the left side and (キャラクター B) on the right side`）
> **1 回のサンプリング**を行います。

| ウィジェット | 説明 |
|------------|------|
| `width` / `height` | 出力画像サイズ（ノード上で直接設定） |
| `steps` | デフォルト：`8`。Z-Image Turbo は少ないステップでも動作します。 |
| `cfg` | デフォルト：`1.0`。Z-Image ベースモデルは `cfg=1.0` を使用します。 |
| `shift` | Z-Image ノイズスケジュール。デフォルト：`3.0`。推奨値：`3~6`。大きいほどノイズを後半ステップへ移動。 |

**基本的な接続:**

```
ZImageCheckpointLoader ────────────→ RPKSamplerZImage
RPPromptParser → RPRatioParser ────→ RPKSamplerZImage
```

---

## Qwen — RPKSamplerQwen

Z-Image と同じアプローチですが、位置ラベルの代わりにシーン叙述型プロンプトを使用します。

> **なぜ統合プロンプトを使用するのですか？**
> Qwen も同じアーキテクチャを使用しているため、ステップごとの領域フックは使用できません。
> 代わりに Qwen はシーン叙述方式でプロンプトを構成します。
> (`(キャラクター A) on the left side and (キャラクター B) on the right side,
> interacting naturally in the same scene, seamless composition`)
> **2 キャラクターが自然にインタラクションするシーン**に最も適しています。

| ウィジェット | 説明 |
|------------|------|
| `width` / `height` | 出力画像サイズ |
| `steps` | デフォルト：`20`。Qwen 推奨値：`15~20`。 |
| `cfg` | デフォルト：`1.0`。Qwen ベースモデルの推奨値。 |
| `shift` | Qwen ノイズスケジュール。デフォルト：`3.0`。推奨値：`3~6`。 |

**基本的な接続:**

```
QwenCheckpointLoader ──────────────→ RPKSamplerQwen
RPPromptParser → RPRatioParser ────→ RPKSamplerQwen
```

---

## ディテイラーノード

サンプラー実行後、検出された人物を領域ごとに個別インペイントして補正します。
YOLO モデルが必要です（例：`bbox/person_yolov8m-seg.pt`）。`models/ultralytics/` フォルダのモデルが自動検出されます。

| ノード | 対象モデル |
|--------|----------|
| `RPRegionalDetailer` | SDXL / SD1.x |
| `RPRegionalDetailerZImage` | Z-Image |
| `RPRegionalDetailerQwen` | Qwen |

### 各 Detailer の動作メカニズム

**RPRegionalDetailer** (SDXL / SD1.x)

1. **領域マスクごとに YOLO を個別実行** — 各領域エリアを切り取って YOLO を個別に適用し、
   該当領域内で最大の人物の bbox を選択します。
2. 検出された各 bbox に対して：マスク膨張（dilation）+ ブラー → crop アップスケール（LANCZOS）
   → VAE エンコード → 該当領域のプロンプト（COMMON + BASE + DIV の組み合わせ）でインペイント
   → VAE デコード → 元の画像に合成します。
3. 各領域インペイント直前に該当領域の LoRA を独立してロードするため、
   **領域ごとの LoRA 独立適用がこの段階で完全に実現**されます。

**RPRegionalDetailerZImage** (Z-Image)

1. **全体画像で YOLO を 1 回実行**した後、各検出された人物の bbox 中心座標を
   divide_ratio の境界と比較して該当領域を自動的に分類します。
2. オプションで **WD14 ONNX 性別分類**（boy/girl スコア）を bbox ごとに実行して
   最も適した領域プロンプトを自動的に選択します。
3. 各 bbox を Z-Image latent（16 チャンネル、crop サイズから自動生成）でインペイントします。
   **プロンプトエンコーディングは RPRegionalDetailer と同じ COMMON + BASE + DIV 組み合わせ方式**を使用します。
   RPKSamplerZImage の位置ラベルマージ方式ではありません。
   領域ごとの LoRA 独立適用は SDXL と同様です。

**RPRegionalDetailerQwen** (Qwen)

RPRegionalDetailerZImage にすべての処理を委任します。
構造的な違いは、Qwen の VAE が 5D テンソル `[B, T, H, W, C]` を返す場合があり、
これを 4D `[B, H, W, C]` に正規化してから RPRegionalDetailerZImage に渡す点です。
領域プロンプトは**シーン叙述方式でマージせずそのまま渡され**、
RPRegionalDetailerZImage と同じ COMMON + BASE + DIV エンコーディングが適用されます。

---

## ライセンス

AGPL-3.0. [LICENSE](LICENSE) を参照してください。

## クレジット

- [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter) — 元のアルゴリズム (hako-mikan)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) — Detailer パターン
- [ComfyUI-ZImagePowerNodes](https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes) — Z-Image サンプラー
