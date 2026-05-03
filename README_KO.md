# ComfyUI_RP_Cast

이미지의 **구역별로 다른 프롬프트**를 적용하는 ComfyUI 노드 모음입니다.
왼쪽/오른쪽, 위/아래, 격자 분할 등을 지원합니다.
SDXL, Z-Image, Qwen 모델에서 사용할 수 있습니다.

**버전: 0.6.00** | [GitHub](https://github.com/AISeDam/ComfyUI_RP_Cast)

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

### 2단계 — 모델에 맞는 샘플러와 디테일러 선택

| 사용 모델 | 샘플러 | 디테일러 |
|-----------|--------|----------|
| SDXL / SD 1.x | `RP KSampler (SDXL)` | `RP Regional Detailer (SDXL)` |
| Z-Image / Qwen | *(ComfyUI 기본 KSampler 사용)* | `RP Regional Detailer (Z-Image)` |
| Qwen | *(ComfyUI 기본 KSampler 사용)* | `RP Regional Detailer (Qwen)` |

### 선택 사항 — RP Converter

| 노드 | 역할 |
|------|------|
| `RP Converter` | 자연어 씬 설명을 로컬 Ollama LLM(`gemma3:12b` 권장)으로 RP 구조 프롬프트로 변환합니다. COSPLAY/인물 키워드 기준으로 입력을 자동 분리하고, 스타일 키워드 추가 및 구역별 LoRA 자동 태그 기능을 포함합니다. |

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

## 디테일러 노드

샘플러 실행 후, 감지된 인물을 구역별로 개별 인페인팅하여 보정합니다.
YOLO 모델이 필요합니다 (예: `bbox/person_yolov8m-seg.pt`). `models/ultralytics/` 폴더의 모델이 자동으로 감지됩니다.

| 노드 | 대상 모델 |
|------|-----------|
| `RP Regional Detailer (SDXL)` | SDXL / SD1.x |
| `RP Regional Detailer (Z-Image)` | Z-Image / Qwen |
| `RP Regional Detailer (Qwen)` | Qwen (Z-Image Detailer에 위임) |

**Fallback 인페인팅 (모든 Detailer 공통)**

동일 구역에 여러 인물이 감지된 경우, 가장 큰 인물이 해당 COL 구역 프롬프트를 담당합니다.
나머지 탈락 인물은 **베이스 프롬프트** (`COMMON + BASE text`)로 인페인팅되어 보정이 적용됩니다.

### 각 Detailer의 동작 메커니즘

**RP Regional Detailer (SDXL)**

1. **전체 이미지에서 YOLO를 1회 실행**한 후, 각 인물의 bbox 중심 좌표를
   divide_ratio 경계와 비교하여 구역을 분류합니다 (ZImage 방식).
   Horizontal, Vertical, 2D 그리드 레이아웃을 모두 지원합니다.
2. 동일 구역에 여러 인물이 겹치면 가장 큰 인물이 우선 배정되고,
   나머지는 베이스 프롬프트 fallback 인페인팅 대상이 됩니다.
3. 각 bbox에 대해: 마스크 팽창(dilation) + 블러 → crop 업스케일(LANCZOS)
   → VAE 인코딩 → 해당 구역의 프롬프트(COMMON + BASE + DIV 조합)로 인페인팅
   → VAE 디코딩 → 마스크 블렌딩으로 원본에 합성합니다.
4. 각 구역 인페인팅 직전에 해당 구역의 LoRA를 독립적으로 로드하므로,
   **구역별 LoRA 독립 적용이 이 단계에서 완전히 실현**됩니다.

**RP Regional Detailer (Z-Image)**

1. **전체 이미지에서 YOLO를 1회 실행**, bbox 중심 좌표로 구역을 분류합니다.
2. 선택적으로 **WD14 ONNX 성별 분류** (boy/girl 스코어)를 실행하여
   가장 적합한 구역 프롬프트를 자동 선택합니다.
3. Z-Image latent (16채널)로 인페인팅합니다.
   프롬프트 인코딩은 **COMMON + BASE + DIV 조합 방식**을 사용합니다.
4. 탈락 인물은 베이스 프롬프트로 fallback 인페인팅됩니다.

**RP Regional Detailer (Qwen)**

`RP Regional Detailer (Z-Image)`에 전체 처리를 위임합니다.
Qwen의 VAE가 5D 텐서 `[B, T, H, W, C]`를 반환할 경우 4D `[B, H, W, C]`로 정규화한 후 전달합니다.
프롬프트는 **COMMON + BASE + DIV 조합 방식**으로 인코딩됩니다.

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

### v0.6.00 (2026-05-03)

**추가**
- `RP Converter` 노드: 자연어 씬 설명을 Ollama LLM을 통해 RP 구조 프롬프트로 변환
  - `style_prompt`: Ollama 변환 후 ADDCOMM~ADDBASE 사이에 키워드 추가
  - `lora_directory` + `lora_auto_apply`: COL 섹션별 랜덤 LoRA 자동 추가 (트리거 워드 포함, AGP 방식)
  - ComfyUI 시작 시 Ollama 모델 리스트 자동 조회, 미실행 시 execute 시 재시도
  - `gemma3:12b` 권장; `llama3.2:3b`, `qwen3` 지원 (`qwen3`는 `/no_think` 지시어 자동 적용)
  - 변환 완료 후 모델 VRAM 해제 (`keep_alive=0`)
  - Ollama 출력 이상 시 WD14 태그 매칭 fallback (미설치 시 자동 다운로드)
  - 규격외 RP 구조(ADDCOMM/ADDBASE 누락 또는 중복) 시 1회 retry
- `RPPromptParser`: `NL_prompts` STRING output 추가 (CLIP encode 직접 연결용)
- `RPKSampler (SDXL)`: `steps_add_per_div`, `cfg_add_per_div` 위젯 추가 (`lora_weight_adj` 위)
  - steps = steps + (n_div × steps_add_per_div), cfg = cfg + (n_div × cfg_add_per_div)
  - 기본값 `0` — 미사용 시 동작 없음
- `RPRegionalDetailer` + `RPRegionalDetailerZImage`: 탈락 인물 fallback 인페인팅
  - COL에 미배정된 인물을 base 프롬프트(common + base_text)로 인페인팅
  - 메인 COL 루프와 동일한 crop→upscale→VAE encode→sample→blend 파이프라인 적용
- `RPRegionalDetailer`: ZImage 방식 전체 이미지 1회 YOLO + bbox 중심 좌표 기반 구역 분류로 전환

**변경**
- `RP KSampler` 표시명 → `RP KSampler (SDXL)`
- `RP Regional Detailer` 표시명 → `RP Regional Detailer (SDXL)`
- `RPRegionalDetailer`: 구역 할당 방식을 bbox 중심 좌표 vs divide_ratio 경계 비교로 변경 (ZImage 방식)
- `RP Converter`: 기본 모델 `gemma3:12b`로 변경; System Prompt를 PART A / PART B 명시적 분리 형식으로 재구성
- `RP Converter`: COSPLAY/인물 키워드 기준으로 Python이 입력을 사전 분리 후 전달
- `RP Converter`: 모든 모델에 `think: false` 적용; `qwen3`는 추가로 `/no_think` 지시어 자동 추가
- `RP Converter`: 모든 API 페이로드에 `context: []` 추가 — 매 요청마다 대화 이력 소거
- `RP Converter`: retry 검증에 ADDCOMM/ADDBASE 중복 감지 추가 (누락 외 중복도 retry 트리거)

**제거**
- `RP KSampler (Z-Image)` 노드 제거
- `RP KSampler (Qwen)` 노드 제거
- `RP KSampler (FLUX.2)` 노드 제거
- 미사용 stub 파일 제거: `node_rp_conditioning.py`, `node_rp_filter_maker.py`, `node_rp_ratio_parser.py`

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
