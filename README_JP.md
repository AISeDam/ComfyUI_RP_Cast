# ComfyUI_RP_Cast

ComfyUIで**領域ごとに異なるプロンプト**を適用するノード集です。
左右・上下・グリッド分割などに対応しています。
SDXL、Z-Image、Qwen モデルで使用できます。

**バージョン: 0.5.60** | [GitHub](https://github.com/AISeDam/ComfyUI_RP_Cast)

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

### ステップ 2 — モデルに合ったサンプラーとディテイラーを選択

| 使用モデル | サンプラー | ディテイラー |
|-----------|-----------|------------|
| SDXL / SD 1.x | `RP KSampler (SDXL)` | `RP Regional Detailer (SDXL)` |
| Z-Image / Qwen | *(ComfyUI標準KSamplerを使用)* | `RP Regional Detailer (Z-Image)` |
| Qwen | *(ComfyUI標準KSamplerを使用)* | `RP Regional Detailer (Qwen)` |

### オプション — RP Converter

| ノード | 役割 |
|--------|------|
| `RP Converter` | 自然言語のシーン説明をOllama LLM（`gemma3:12b`推奨）でRP構造プロンプトに変換します。COSPLAY/人物キーワードで入力を自動分割し、スタイルキーワード追加と領域ごとのLoRA自動タグ機能を含みます。 |

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
| `RP Regional Detailer (SDXL)` | SDXL / SD1.x |
| `RP Regional Detailer (Z-Image)` | Z-Image / Qwen |
| `RP Regional Detailer (Qwen)` | Qwen（Z-Imageディテイラーに委譲） |

**フォールバックインペインティング（全ディテイラー共通）**

同じ領域で複数の人物が検出された場合、最大面積の人物がCOL領域プロンプトを担当します。
残りの落選人物は**ベースプロンプト**（`COMMON + BASE text`）でインペインティングされます。

### 各 Detailer の動作メカニズム

**RP Regional Detailer (SDXL)**

1. **全体画像に対してYOLOを1回実行**し、各人物のbboxセンター座標を
   divide_ratio境界と比較して領域を分類します（ZImageスタイル）。
2. 同じ領域に複数の人物がいる場合は最大面積の人物が優先配置され、
   残りはベースプロンプトのフォールバックインペインティング対象になります。
3. 各bboxに対して: マスク膨張(dilation) + ブラー → cropアップスケール(LANCZOS)
   → VAEエンコード → 領域プロンプト(COMMON + BASE + DIV組み合わせ)でインペインティング
   → VAEデコード → マスクブレンドで合成します。
4. 各領域インペインティング直前に該当領域のLoRAを個別ロードするため、
   **領域ごとのLoRA独立適用がこの段階で完全に実現**されます。

*(旧: RPRegionalDetailer)*

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

## Txt2Img API ノード

外部画像生成 API を使用して Regional Prompter プロンプトから画像を生成します。
GPU 不要で使用可能。プロンプトは自然言語に変換されて API に送信されます。

> `RPPromptParser → RPRatioParser` の接続方法は既存のサンプラーと同じです。
> `regional_col_n_row`（RPRatioParser 出力）と `divide_mode`（RPPromptParser 出力）を
> ノードに接続するとグリッドレイアウトが自動計算されます。

### RP Txt2Img (OpenAI)

[OpenAI Image Generation API](https://platform.openai.com/docs/api-reference/images) を使用します。

| ウィジェット | 説明 |
|-------------|------|
| `model` | `gpt-image-2` / `gpt-image-1.5` / `gpt-image-1` / `gpt-image-1-mini` |
| `size` | `1024×1024` / `1536×1024` / `1024×1536` / `auto` |
| `quality` | `auto` / `high` / `medium` / `low` |
| `debug` | 変換されたプロンプトをコンソールに出力 |

**設定キー**: `ComfyUI-RP-Cast.Configuration.openai_api_key`

---

### RP Txt2Img (Gemini)

[Google Gemini Image Generation API](https://ai.google.dev/gemini-api/docs/image-generation) を使用します。

| ウィジェット | 説明 |
|-------------|------|
| `model` | `gemini-3.1-flash-image-preview` / `gemini-3-pro-image-preview` / `gemini-2.5-flash-image` |
| `aspect_ratio` | `1:1` / `3:4` / `4:3` / `9:16` / `16:9` など10種 |
| `image_size` | `1K` / `2K` / `4K` |
| `debug` | 変換プロンプトおよび API レスポンス情報を出力 |

**設定キー**: `ComfyUI-RP-Cast.Configuration.gemini_api_key`
API Key 取得: https://aistudio.google.com/apikey

---

### RP Txt2Img (Grok)

[xAI Grok Image Generation API](https://docs.x.ai/developers/rest-api-reference/inference/images) を使用します。

| ウィジェット | 説明 |
|-------------|------|
| `model` | `grok-imagine-image` / `grok-imagine-image-pro` |
| `aspect_ratio` | `1:1` / `3:4` / `4:3` / `9:16` / `16:9` など14種 |
| `quality` | `low` / `medium` / `high` |
| `resolution` | `1k` / `2k` |
| `debug` | 変換されたプロンプトをコンソールに出力 |

**設定キー**: `ComfyUI-RP-Cast.Configuration.grok_api_key`

---

## アップデート履歴

### v0.6.00 (2026-05-03)

**追加**
- `RP Converter` ノード: 自然言語のシーン説明をOllama LLMでRP構造プロンプトに変換
  - `style_prompt`: Ollama変換後にADDCOMM～ADDBASE間にキーワードを追加
  - `lora_directory` + `lora_auto_apply`: COLセクションごとにランダムLoRAを自動追加（トリガーワード付き、AGPスタイル）
  - ComfyUI起動時にOllamaモデルリストを自動取得、未起動時はexecute時にリトライ
  - `gemma3:12b`推奨; `llama3.2:3b`、`qwen3`もサポート（`qwen3`は`/no_think`指示語を自動適用）
  - 変換完了後にVRAMからモデルをアンロード（`keep_alive=0`）
  - Ollama出力が無効な場合はWD14タグマッチングにフォールバック（未インストール時は自動ダウンロード）
  - 無効なRP構造（ADDCOMM/ADDBASE欠落または重複）時に1回リトライ
  - `RPPromptParser`に`NL_prompts` STRING出力を追加（CLIPエンコードへの直接接続用）
- `RPKSampler (SDXL)`: `steps_add_per_div`、`cfg_add_per_div`ウィジェットを追加（`lora_weight_adj`の上）
  - steps = steps + (n_div × steps_add_per_div), cfg = cfg + (n_div × cfg_add_per_div)
- `RPRegionalDetailer` + `RPRegionalDetailerZImage`: 落選人物のフォールバックインペインティング
  - COLに未割当の人物をベースプロンプト（common + base_text）でインペインティング
- `RPRegionalDetailer`: ZImageスタイルの全体画像1回YOLO + bboxセンター座標による領域分類に切り替え

**変更**
- `RP KSampler`表示名 → `RP KSampler (SDXL)`
- `RP Regional Detailer`表示名 → `RP Regional Detailer (SDXL)`
- `RPRegionalDetailer`: 領域割当をbboxセンター座標 vs divide_ratio境界比較に変更（ZImageパターン）

**削除**
- `RP KSampler (Z-Image)`ノード削除
- `RP KSampler (Qwen)`ノード削除
- `RP KSampler (FLUX.2)`ノード削除
- 未使用stubファイル削除: `node_rp_conditioning.py`、`node_rp_filter_maker.py`、`node_rp_ratio_parser.py`

- `RP Converter`: デフォルトモデルを`gemma3:12b`に変更（llama3.2:3b比で指示遵守性能が優れている）
- `RP Converter`: System PromptをGemma3専用に再構成 — PART A / PART B明示的分割形式＋EXAMPLE付き
- `RP Converter`: User PromptでCOSPLAY/人物キーワード基準でPythonが先に入力を分割して渡すことで、セクション間タグ混用問題を解消
- `RP Converter`: ストリーミング/非ストリーミング全APIペイロードに`think: false`を適用
- `RP Converter`: `qwen3`モデル使用時はUser Prompt冒頭に`/no_think`指示語を自動追加（二重保証）
- `RP Converter`: 全APIペイロードに`context: []`追加 — 毎リクエストごとに会話履歴を完全消去し、以前のプロンプト漏洩を防止
- `RP Converter`: retry検証に`ADDCOMM`/`ADDBASE`重複検出を追加（欠落に加えて重複もretryトリガー）
- `RP Converter`: retryヒントメッセージに「ADDCOMM must appear EXACTLY ONCE / ADDBASE must appear EXACTLY ONCE」を明示
- `RPKSampler (SDXL)`: `steps_add_per_div`/`cfg_add_per_div`デフォルト値を`0`に設定（未使用時は動作なし）
- `RPRegionalDetailer (SDXL)`/`RPRegionalDetailerZImage`: fallbackインペインティングをメインCOLループと同じcrop→upscale→VAEエンコード→sample→blendパイプラインに改善

### v0.5.60 (2026-04-30)

**バグ修正**
- `RPKSampler` `use_base=False` 修正: BASE ブロックテキスト(nolora_list[0])が正しくスキップされず、リージョンマッピングがずれていた問題
- `RPKSampler` `use_base=False` 修正: null BASE アンカー（空テキスト, strength=0.0）を注入し COL 間 bleeding を防止
- `RPKSampler` `use_base=False` 修正: COL conditioning strength を 1.0 に強制（use_base=False 時は base_ratio を無視）
- `RPKSampler` `use_base=False` 修正: exclusive area 境界を適用してリージョン bleed を防止
- `RPKSampler` col_lora_map インデックス修正: `lora_offset` 追加で use_base=False 時に LoRA が正しくマッピングされるよう修正
- `RPKSampler` `use_base=True` 修正: BASE テキストが正常にエンコードされるよう修正（前回の修正で null になっていた問題）
- 全 sampler/detailer: `patches.clear()` を `patches = {}` に置換し shared dict 変異を防止
- 全 sampler/detailer: sampling 前に `unpatch_model()` を明示的に呼び出し stale LowVramPatch を解放
- `RPRegionalDetailerZImage`: LoRA モデルを inpaint ループ前に事前ビルドするよう変更
- `RPRegionalDetailerZImage`: 常に `model.clone()` で fresh base を使用しクロスラン patch 汚染を防止

**変更**
- `core/prompt_parser.py`: `subprompts_raw` が col のみを含むよう変更（common は含まない）
- `core/lora_manager.py`: `setup()`、`prebuild_cache()`、`get_model_for_division()` に `lora_offset` パラメータを追加
- `RPKSampler`: common テキストを CLIP エンコード前に各 DIV に 1 回だけ pre-merge（`_null_base` 戦略）


### v0.5.59 (2026-04-29)

**追加**
- `RP Txt2Img (OpenAI)` ノード: GPT Image API で画像生成 (gpt-image-2/1.5/1/1-mini)
- `RP Txt2Img (Gemini)` ノード: Google Gemini generateContent API で画像生成
- `RP Txt2Img (Grok)` ノード: xAI Grok Image API で画像生成
- ComfyUI Settings に API キー設定項目を追加 (openai / gemini / grok)
- 全 Txt2Img ノードに `regional_col_n_row`、`divide_mode` の optional dot 入力を追加
- `_rp_txt2img_common.py`: Txt2Img ノード共通ユーティリティモジュール
- Gemini ノードに `image_size` パラメータ追加 (`1K` / `2K` / `4K`)
- Grok ノードに `quality`、`resolution` パラメータ追加
- 全 Txt2Img ノードが完全独立動作（ノード間の相互依存なし）

**バグ修正**
- `_convert_rp_to_natural` IndexError 修正: Case C (1D Horizontal) の cells が 4-tuple なのに index 4 にアクセスしていた問題
- Vertical グリッド位置ラベル修正: `divide_mode=Vertical` の際に rows/cols が入れ替わっていた問題
- Horizontal 非対称グリッド修正 (例: 2+1 行): 単独行領域が `lower-left` ではなく `lower-center` と表示されるよう修正
- `base_ratio` が BASE conditioning strength に適用されないバグ修正 (1.0 固定になっていた)

**変更**
- `parse_prompt()`: `subprompts_raw` が col のみを含むよう変更（common はここではマージしない）
- `RPKSampler`: CLIP エンコード前に common テキストを各 DIV テキストに 1 回だけ pre-merge 処理

---


---

## ライセンス

AGPL-3.0. [LICENSE](LICENSE) を参照してください。

## クレジット

- [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter) — 元のアルゴリズム (hako-mikan)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) — Detailer パターン
- [ComfyUI-ZImagePowerNodes](https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes) — Z-Image サンプラー
