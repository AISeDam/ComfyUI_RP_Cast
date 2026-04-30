# ComfyUI_RP_Cast

이미지의 **구역별로 다른 프롬프트**를 적용하는 ComfyUI 노드 모음입니다.
왼쪽/오른쪽, 위/아래, 격자 분할 등을 지원합니다.
SDXL, Z-Image, Qwen 모델에서 사용할 수 있습니다.

**버전: 0.5.60** | [GitHub](https://github.com/AISeDam/ComfyUI_RP_Cast)

---

## 이 프로그램에 대해

이 프로젝트는 hako-mikan의
[sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter)를
참조하여, 일부 기능을 ComfyUI 환경에 맞게 **포팅 및 변형**한 프로그램입니다.

왼쪽에는 A 캐릭터, 오른쪽에는 B 캐릭터를 각각 다른 프롬프트와 LoRA로 생성할 수 있습니다.
마스크나 수동 선택 없이 프롬프트 작성만으로 구역을 나눌 수 있습니다.

프롬프트 분할 개념 (ADDCOMM / ADDBASE / ADDCOL / ADDROW),
divide_ratio 문법, 구역별 latent 블렌딩 알고리즘은 원본에서 파생되었습니다.
다음 항목은 ComfyUI 환경 및 확장 모델 지원을 위해 **추가·변형**되었습니다:

- ComfyUI 노드 아키텍처 (모듈화 Python 노드, JS 프론트엔드 확장)
- Z-Image 지원 (RPKSamplerZImage, RPRegionalDetailerZImage)
- Qwen 장면 구성 지원 (RPKSamplerQwen, RPRegionalDetailerQwen)
- 구역별 LoRA 분할 관리
- YOLO 기반 Regional Detailer 노드

> 원본 프롬프트 문법과 알고리즘 상세는 아래를 참고하세요:
> **→ https://github.com/hako-mikan/sd-webui-regional-prompter**

---

## 설치

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/AISeDam/ComfyUI_RP_Cast
```

ComfyUI를 재시작하면 8개 노드가 자동으로 나타납니다.

---

## 어떤 노드를 쓰면 되나요?

### 1단계 — 항상 이 두 노드부터 시작

| 노드 | 역할 |
|------|------|
| `RPPromptParser` | 구역별 프롬프트를 작성하고 자동으로 분리합니다 |
| `RPRatioParser` | `divide_ratio`, `divide_mode`, `threshold`를 입력받아 구역 데이터(`regional_col_n_row`)와 threshold를 출력합니다. RPPromptParser 다음에 연결합니다 |

### 2단계 — 모델에 맞는 샘플러 선택

| 사용 모델 | 샘플러 | 디테일러 |
|-----------|--------|----------|
| SDXL / SD 1.x | `RPKSampler` | `RPRegionalDetailer` |
| Z-Image | `RPKSamplerZImage` | `RPRegionalDetailerZImage` |
| Qwen | `RPKSamplerQwen` | `RPRegionalDetailerQwen` |

---

## SDXL / SD1.x — RPKSampler

`EmptyLatentImage` 노드를 연결해서 캔버스 크기를 지정합니다.

| 위젯 | 설명 |
|------|------|
| `use_base` | RPPromptParser의 `use_base`와 동일하게 설정 |
| `use_common` | COMMON 프롬프트를 모든 구역에 적용 |
| `base_ratio` | BASE와 구역 프롬프트의 혼합 비율. `0.2` = BASE 20%. 구역별 지정: `0.2,0.3` |
| `lora_weight_adj` | 전체 LoRA 가중치 배율. `100` = 원본, `50` = 절반, `200` = 2배, `0` = 비활성 |

> **구역별 LoRA 독립 적용이 불가능한 기술적 이유**
>
> ComfyUI의 모델 실행 파이프라인은 샘플링 도중 구역별로 LoRA 가중치를 안전하게
> 교체하는 훅을 제공하지 않습니다.
> 내부적으로 RPKSampler는 `set_model_unet_function_wrapper` 콜백을 등록하여
> 각 UNet 호출을 가로채고 해당 스텝에서 어느 구역(division)이 처리되는지 식별합니다.
> 그러나 실제로 LoRA가 적용된 모델 가중치를 그 시점에 교체하면
> conditioning 불일치와 샘플링 불안정이 발생하므로,
> 모든 구역에 단일 공유 모델을 사용하는 방식으로 동작합니다.
>
> 결과적으로 모든 구역의 LoRA는 **평균 병합**되어 샘플링 전에 하나의 모델로 통합 적용됩니다.
> `lora_weight_adj`로 이 병합된 전체 가중치를 일괄 조정할 수 있습니다.
>
> 구역별 캐릭터 특징 강화는 **Detailer 단계**에서 보완됩니다.
> YOLO로 감지된 각 bbox를 해당 구역의 프롬프트와 LoRA로 개별 인페인팅하므로,
> 캐릭터 고유의 특징이 이 단계에서 가장 효과적으로 반영됩니다.

**기본 연결:**

```
CheckpointLoader ──────────────────→ RPKSampler
EmptyLatentImage → RPKSampler (latent_image)
RPPromptParser → RPRatioParser ────→ RPKSampler
```

---

## Z-Image — RPKSamplerZImage

`EmptyLatentImage` 불필요 — 노드에서 `width`, `height`를 직접 설정합니다.

> **왜 통합 프롬프트를 사용하나요?**
> Z-Image는 SDXL과 다른 attention 아키텍처를 사용합니다.
> SDXL의 구역별 샘플링은 각 샘플링 스텝마다 Region별 프롬프트를 별도로 처리하고
> 그 결과 latent를 구역별로 블렌딩하는 방식(denoised callback)으로 동작합니다.
> Z-Image는 이 스텝별 훅을 지원하지 않아 구역별 latent 블렌딩을 적용할 수 없습니다.
> 대신 모든 구역 프롬프트를 자연어 위치 레이블과 함께 하나의 프롬프트로 병합하여
> (예: `(캐릭터 A) on the left side and (캐릭터 B) on the right side`)
> **1회 샘플링**합니다.

| 위젯 | 설명 |
|------|------|
| `width` / `height` | 출력 이미지 크기 (노드에서 직접 설정) |
| `steps` | 기본값: `8`. Z-Image Turbo는 적은 스텝으로도 동작합니다. |
| `cfg` | 기본값: `1.0`. Z-Image 기반 모델은 `cfg=1.0`을 사용합니다. |
| `shift` | Z-Image 노이즈 스케줄. 기본값: `3.0`. 권장값: `3~6`. 값이 클수록 노이즈를 후반 스텝으로 이동 |

**기본 연결:**

```
ZImageCheckpointLoader ────────────→ RPKSamplerZImage
RPPromptParser → RPRatioParser ────→ RPKSamplerZImage
```

---

## Qwen — RPKSamplerQwen

Z-Image와 같은 방식이지만, 위치 레이블 대신 장면 서술형 프롬프트를 사용합니다.

> **왜 통합 프롬프트를 사용하나요?**
> Qwen도 동일한 아키텍처를 사용하므로 스텝별 구역 훅을 사용할 수 없습니다.
> 대신 Qwen은 장면 서술 방식으로 프롬프트를 구성합니다.
> (`(캐릭터 A) on the left side and (캐릭터 B) on the right side,
> interacting naturally in the same scene, seamless composition`)
> **두 캐릭터가 자연스럽게 상호작용하는 장면**에 가장 적합합니다.

| 위젯 | 설명 |
|------|------|
| `width` / `height` | 출력 이미지 크기 |
| `steps` | 기본값: `20`. Qwen 권장값: `15~20`. |
| `cfg` | 기본값: `1.0`. Qwen 기반 모델 권장값. |
| `shift` | Qwen 노이즈 스케줄. 기본값: `3.0`. 권장값: `3~6` |

**기본 연결:**

```
QwenCheckpointLoader ──────────────→ RPKSamplerQwen
RPPromptParser → RPRatioParser ────→ RPKSamplerQwen
```


---

## 디테일러 노드

샘플러 실행 후, 감지된 인물을 구역별로 개별 인페인팅하여 보정합니다.
YOLO 모델이 필요합니다 (예: `bbox/person_yolov8m-seg.pt`). `models/ultralytics/` 폴더의 모델이 자동으로 감지됩니다.

| 노드 | 대상 모델 |
|------|-----------|
| `RPRegionalDetailer` | SDXL / SD1.x |
| `RPRegionalDetailerZImage` | Z-Image |
| `RPRegionalDetailerQwen` | Qwen |

### 각 Detailer의 동작 메커니즘

**RPRegionalDetailer** (SDXL / SD1.x)

1. **구역 마스크별로 YOLO를 개별 실행** — 각 구역 영역을 잘라내어 YOLO를 따로 적용하고,
   해당 구역 내에서 가장 큰 인물의 bbox를 선택합니다.
2. 감지된 각 bbox에 대해: 마스크 팽창(dilation) + 블러 → crop 업스케일(LANCZOS)
   → VAE 인코딩 → 해당 구역의 프롬프트(COMMON + BASE + DIV 조합)로 인페인팅
   → VAE 디코딩 → 원본 이미지에 합성합니다.
3. 각 구역 인페인팅 직전에 해당 구역의 LoRA를 독립적으로 로드하므로,
   **구역별 LoRA 독립 적용이 이 단계에서 완전히 실현**됩니다.

**RPRegionalDetailerZImage** (Z-Image)

1. **전체 이미지에서 YOLO를 1회 실행**한 후, 각 감지된 인물의 bbox 중심 좌표를
   divide_ratio 경계와 비교하여 해당 구역을 자동으로 분류합니다.
2. 선택적으로 **WD14 ONNX 성별 분류** (boy/girl 스코어)를 bbox별로 실행하여
   가장 적합한 구역 프롬프트를 자동으로 선택합니다.
3. 각 bbox를 Z-Image latent (16채널, crop 크기에서 자동 생성)로 인페인팅합니다.
   **프롬프트 인코딩은 RPRegionalDetailer와 동일한 COMMON + BASE + DIV 조합 방식**을 사용합니다.
   RPKSamplerZImage의 위치 레이블 병합 방식이 아닙니다.
   구역별 LoRA 독립 적용은 SDXL과 동일합니다.

**RPRegionalDetailerQwen** (Qwen)

RPRegionalDetailerZImage에 전체 처리를 위임합니다.
구조적 차이점은 Qwen의 VAE가 5D 텐서 `[B, T, H, W, C]`를 반환할 수 있어,
이를 4D `[B, H, W, C]`로 정규화한 후 RPRegionalDetailerZImage에 전달한다는 것입니다.
구역 프롬프트는 **장면 서술 방식으로 병합하지 않고 그대로 전달**되며,
RPRegionalDetailerZImage와 동일한 COMMON + BASE + DIV 인코딩이 적용됩니다.

---

## Txt2Img API 노드

외부 이미지 생성 API를 사용해 Regional Prompter 프롬프트로 이미지를 생성합니다.
GPU 없이도 사용 가능하며, 프롬프트는 자연어로 변환되어 API에 전송됩니다.

> `RPPromptParser → RPRatioParser` 연결 방식은 기존 샘플러와 동일합니다.
> `regional_col_n_row`(RPRatioParser 출력)와 `divide_mode`(RPPromptParser 출력)를
> 노드에 연결하면 그리드 레이아웃이 자동 계산됩니다.

### RP Txt2Img (OpenAI)

[OpenAI Image Generation API](https://platform.openai.com/docs/api-reference/images)를 사용합니다.

| 위젯 | 설명 |
|------|------|
| `model` | `gpt-image-2` / `gpt-image-1.5` / `gpt-image-1` / `gpt-image-1-mini` |
| `size` | `1024×1024` / `1536×1024` / `1024×1536` / `auto` |
| `quality` | `auto` / `high` / `medium` / `low` |
| `debug` | 변환된 프롬프트를 콘솔에 출력 |

**설정 키**: `ComfyUI-RP-Cast.Configuration.openai_api_key`

---

### RP Txt2Img (Gemini)

[Google Gemini Image Generation API](https://ai.google.dev/gemini-api/docs/image-generation)를 사용합니다.

| 위젯 | 설명 |
|------|------|
| `model` | `gemini-3.1-flash-image-preview` / `gemini-3-pro-image-preview` / `gemini-2.5-flash-image` |
| `aspect_ratio` | `1:1` / `3:4` / `4:3` / `9:16` / `16:9` 등 10종 |
| `image_size` | `1K` / `2K` / `4K` |
| `debug` | 변환된 프롬프트 및 API 응답 정보 출력 |

**설정 키**: `ComfyUI-RP-Cast.Configuration.gemini_api_key`
API Key 발급: https://aistudio.google.com/apikey

---

### RP Txt2Img (Grok)

[xAI Grok Image Generation API](https://docs.x.ai/developers/rest-api-reference/inference/images)를 사용합니다.

| 위젯 | 설명 |
|------|------|
| `model` | `grok-imagine-image` / `grok-imagine-image-pro` |
| `aspect_ratio` | `1:1` / `3:4` / `4:3` / `9:16` / `16:9` 등 14종 |
| `quality` | `low` / `medium` / `high` |
| `resolution` | `1k` / `2k` |
| `debug` | 변환된 프롬프트를 콘솔에 출력 |

**설정 키**: `ComfyUI-RP-Cast.Configuration.grok_api_key`

---

## 업데이트 이력

### v0.5.60 (2026-04-30)

**버그 수정**
- `RPKSampler` `use_base=False` 수정: BASE 블록 텍스트(nolora_list[0])가 올바르게 skip되지 않아 region 매핑이 틀어지던 문제
- `RPKSampler` `use_base=False` 수정: null BASE anchor(빈 텍스트, strength=0.0) 주입으로 COL 간 bleeding 방지
- `RPKSampler` `use_base=False` 수정: COL conditioning strength 1.0 강제 (use_base=False이면 base_ratio 무시)
- `RPKSampler` `use_base=False` 수정: exclusive area 경계 적용으로 region bleed 방지
- `RPKSampler` col_lora_map index 수정: `lora_offset` 추가로 use_base=False 시 LoRA가 올바르게 매핑됨
- `RPKSampler` `use_base=True` 수정: BASE 텍스트가 정상 인코딩되도록 수정 (이전 수정 시도에서 null로 처리되던 문제)
- 전체 sampler/detailer: `patches.clear()` → `patches = {}`로 교체하여 shared dict 변이 방지
- 전체 sampler/detailer: sampling 전 `unpatch_model()` 명시 호출로 stale LowVramPatch 해제
- `RPRegionalDetailerZImage`: LoRA 모델을 inpaint 루프 전에 미리 빌드하도록 변경 (루프 내 object_patch 중첩 방지)
- `RPRegionalDetailerZImage`: 항상 `model.clone()`으로 fresh base 사용하여 cross-run patch 오염 방지

**변경**
- `core/prompt_parser.py`: `subprompts_raw`가 col만 포함하도록 변경 (common 미포함)
- `core/lora_manager.py`: `setup()`, `prebuild_cache()`, `get_model_for_division()`에 `lora_offset` 파라미터 추가
- `RPKSampler`: common 텍스트를 CLIP 인코딩 전에 각 DIV에 1회만 pre-merge (`_null_base` 전략)


### v0.5.59 (2026-04-29)

**추가**
- `RP Txt2Img (OpenAI)` 노드: GPT Image API로 이미지 생성 (gpt-image-2/1.5/1/1-mini)
- `RP Txt2Img (Gemini)` 노드: Google Gemini generateContent API로 이미지 생성
- `RP Txt2Img (Grok)` 노드: xAI Grok Image API로 이미지 생성
- ComfyUI Settings에 API 키 설정 항목 추가 (openai / gemini / grok)
- 모든 Txt2Img 노드에 `regional_col_n_row`, `divide_mode` optional dot 입력 추가
- `_rp_txt2img_common.py`: Txt2Img 노드 공통 유틸리티 모듈
- Gemini 노드에 `image_size` 파라미터 추가 (`1K` / `2K` / `4K`)
- Grok 노드에 `quality`, `resolution` 파라미터 추가
- 모든 Txt2Img 노드 완전 독립 동작 (노드 간 상호 의존 없음)

**버그 수정**
- `_convert_rp_to_natural` IndexError 수정: Case C (1D Horizontal) cells가 4-tuple인데 index 4에 접근하던 문제
- Vertical 그리드 위치 레이블 수정: `divide_mode=Vertical`일 때 rows/cols가 뒤바뀌던 문제
- Horizontal 비대칭 그리드 수정 (예: 2+1 행): 단독 행 구역이 `lower-left` 대신 `lower-center`로 표시되도록 수정
- `base_ratio`가 BASE conditioning strength에 적용되지 않던 버그 수정 (1.0으로 고정되어 있었음)

**변경**
- `parse_prompt()`: `subprompts_raw`가 col만 포함하도록 변경 (common은 여기서 합산하지 않음)
- `RPKSampler`: common 텍스트를 CLIP 인코딩 전에 각 DIV 텍스트에 1회만 pre-merge 처리

---


---

## 라이선스

AGPL-3.0. [LICENSE](LICENSE) 참고.

## 참고

- [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter) — 원본 알고리즘 (hako-mikan)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) — 디테일러 패턴
- [ComfyUI-ZImagePowerNodes](https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes) — Z-Image 샘플러
