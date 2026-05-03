import os, re, csv, json

_HERE = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════
# Ollama API helper
# ══════════════════════════════════════════════════════════════════════════
# ── Ollama base URL: OLLAMA_HOST env → 127.0.0.1 (avoid Windows IPv6) ──
import os as _os
_OLLAMA_BASE = _os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
_OLLAMA_MODELS: list   = []
_OLLAMA_AVAILABLE: bool = False

def _fetch_ollama_models() -> bool:
    """Query Ollama /api/tags. Returns True on success."""
    global _OLLAMA_MODELS, _OLLAMA_AVAILABLE
    import urllib.request, urllib.error

    # Try configured host first, then fallback candidates
    candidates = list(dict.fromkeys([
        _OLLAMA_BASE,
        "http://127.0.0.1:11434",
        "http://localhost:11434",
    ]))

    for base in candidates:
        try:
            req = urllib.request.urlopen(f"{base}/api/tags", timeout=5)
            data = json.loads(req.read().decode())
            _OLLAMA_MODELS    = [m["name"] for m in data.get("models", [])]
            _OLLAMA_AVAILABLE = True
            print(f"[RPConverter] Ollama OK ({base}) "
                  f"— models: {_OLLAMA_MODELS}")
            return True
        except Exception:
            continue

    _OLLAMA_AVAILABLE = False
    _OLLAMA_MODELS    = []
    print("[RPConverter] WARNING: Ollama not reachable. "
          "Start Ollama (ollama serve) and restart ComfyUI.")
    return False

# Try at module load; also retried at execute() time
_fetch_ollama_models()


def _extract_first_rp_block(text: str) -> str:
    """Extract first complete RP block.
    Removes trailing empty ADDCOL/ADDBASE/ADDCOMM with no content.
    """
    out_lines = []
    in_block = False
    addcol_seen = False

    for line in text.strip().split("\n"):
        s = line.strip()
        if s in ("---", "=== END EXAMPLES ===", "=== END EXAMPLE ===") and out_lines:
            break
        if s.startswith("=== FORMAT") or (s.startswith("Scene:") and out_lines):
            break
        if not s:
            if addcol_seen and out_lines and out_lines[-1] not in ("ADDCOMM","ADDBASE","ADDCOL"):
                break
            continue
        in_block = True
        if s == "ADDCOL":
            addcol_seen = True
        out_lines.append(s)

    # Remove trailing empty keywords (e.g. bare ADDCOL with no character content)
    while out_lines and out_lines[-1] in ("ADDCOL", "ADDCOMM", "ADDBASE"):
        out_lines.pop()

    return "\n".join(out_lines)


def _ollama_unload(model: str) -> None:
    """Unload model from memory after use (keep_alive=0)."""
    import urllib.request
    candidates = list(dict.fromkeys([
        _OLLAMA_BASE,
        "http://127.0.0.1:11434",
        "http://localhost:11434",
    ]))
    for base in candidates:
        try:
            payload = json.dumps({
                "model": model,
                "keep_alive": 0
            }).encode()
            req = urllib.request.Request(
                f"{base}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            print(f"[RPConverter] model unloaded: {model}")
            return
        except Exception:
            continue


def _ollama_generate(model: str, prompt: str, system: str = "") -> str:
    """Call Ollama /api/generate.
    - Non-thinking models (llama3.2 etc): stream=False, direct response
    - Thinking models (qwen3 etc): stream=True, capture after </think>
    """
    import urllib.request, urllib.error, re as _re

    candidates = list(dict.fromkeys([
        _OLLAMA_BASE,
        "http://127.0.0.1:11434",
        "http://localhost:11434",
    ]))

    last_err = None
    for base in candidates:
        try:
            # First try non-streaming (fast, works for llama3.2 etc)
            payload = json.dumps({
                "model":  model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "think":  False,
                "context": [],  # clear conversation history every request
                "options": {"temperature": 0.0, "num_predict": 400},
            }).encode()

            req = urllib.request.Request(
                f"{base}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read().decode())
            result = data.get("response", "").strip()

            # If response is empty or looks like thinking text (no ADDBASE/ADDCOMM)
            # → fallback to streaming to capture after </think>
            has_rp_structure = "ADDCOMM" in result or "ADDBASE" in result
            is_thinking_output = (
                len(result) > 200 and
                not has_rp_structure and
                ("Steps:" in result or "We are" in result or "However" in result)
            )

            if not result or is_thinking_output:
                print(f"[RPConverter] non-stream gave thinking output → streaming fallback")
                result = _stream_after_think(base, model, prompt, system)

            if result:
                # Normalize abbreviated keywords
                result = _re.sub(r"\bADCOMM\b",  "ADDCOMM", result)
                result = _re.sub(r"\bADBASE\b",   "ADDBASE", result)
                result = _re.sub(r"\bADCOL\b",    "ADDCOL",  result)
                # Extract only first RP block (prevent duplicates)
                result = _extract_first_rp_block(result)
                return result

        except urllib.error.URLError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            break

    raise RuntimeError(
        f"[RPConverter] Ollama connection failed: {last_err}\n"
        f"\u2192 Start Ollama: ollama serve\n"
        f"\u2192 Then restart ComfyUI."
    )


def _stream_after_think(base: str, model: str, prompt: str, system: str) -> str:
    """Streaming fallback: capture tokens after </think> tag (for qwen3 etc)."""
    import urllib.request, json

    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "think":  False,
        "context": [],  # clear conversation history every request
        "options": {"temperature": 0.0},
    }).encode()

    req = urllib.request.Request(
        f"{base}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=600)

    full       = ""
    think_done = False
    result_buf = ""

    for line in resp:
        chunk = json.loads(line.decode("utf-8"))
        token = chunk.get("response", "")
        full += token

        if not think_done:
            if "</think>" in full:
                think_done = True
                result_buf = full.split("</think>", 1)[1]
        else:
            result_buf += token
            if result_buf.count("ADDCOL") >= 1:
                lines_so_far = [l.strip() for l in result_buf.split("\n") if l.strip()]
                if lines_so_far and lines_so_far[-1] not in ("ADDCOMM","ADDBASE","ADDCOL"):
                    if len(lines_so_far) >= 6:
                        break

        if chunk.get("done", False):
            break

    return (result_buf if think_done else "").strip()


_WD14_TAG_SET = None   # dict: name → category (loaded once on first use)


def _load_wd14_tags():
    global _WD14_TAG_SET
    if _WD14_TAG_SET is not None:
        return

    csv_name  = "selected_tags.csv"
    data_dir  = os.path.join(_HERE, "data", "wd14")
    os.makedirs(data_dir, exist_ok=True)

    csv_path = None
    search_dirs = [data_dir]
    try:
        import folder_paths as _fp
        search_dirs += [
            os.path.join(_fp.base_path, "custom_nodes",
                         "ComfyUI-WD14-Tagger", "models"),
            os.path.join(_fp.models_dir, "wd14"),
            os.path.join(_fp.models_dir, "tagger"),
        ]
    except Exception:
        pass

    for d in search_dirs:
        cp = os.path.join(d, csv_name)
        if os.path.isfile(cp):
            csv_path = cp
            print(f"[RPConverter] WD14 tags found: {cp}")
            break

    if csv_path is None:
        print("[RPConverter] WD14 tags not found → downloading...")
        try:
            from huggingface_hub import hf_hub_download
            csv_path = hf_hub_download(
                repo_id="SmilingWolf/wd-v1-4-swinv2-tagger-v2",
                filename=csv_name,
                local_dir=data_dir,
            )
            print(f"[RPConverter] WD14 tags downloaded: {csv_path}")
        except ImportError:
            raise RuntimeError(
                "huggingface_hub required: pip install huggingface_hub"
            )

    tag_set = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip().replace("_", " ")
            cat  = int(row.get("category", 0))
            tag_set[name] = cat

    _WD14_TAG_SET = tag_set
    print(f"[RPConverter] WD14 tags loaded: {len(tag_set)}")


def _match_wd14_tags(text: str) -> list[str]:
    """Match WD14 tag names in free-form text (longest match first)."""
    _load_wd14_tags()
    text_lower = " " + re.sub(r"[^\w\s]", " ", text.lower()) + " "
    matched = []
    used_spans: set = set()

    for tag in sorted(_WD14_TAG_SET.keys(), key=lambda x: -len(x)):
        if len(tag) < 3:
            continue
        pat = re.compile(r"\b" + re.escape(tag) + r"\b", re.IGNORECASE)
        m = pat.search(text_lower)
        if m:
            s, e = m.start(), m.end()
            if not any(a <= s < b or a < e <= b for a, b in used_spans):
                used_spans.add((s, e))
                matched.append(tag)

    return matched


# ══════════════════════════════════════════════════════════════════════════
# Ollama system prompt for RP conversion
# ══════════════════════════════════════════════════════════════════════════
_SYSTEM_PROMPT = """You are an RP (Regional Prompter) tag converter.
Convert scene descriptions into RP-structured tag prompts.

OUTPUT FORMAT (use exactly, no extra explanation):
<scene tags>
ADDCOMM
<style/quality tags>
ADDBASE
1girl/1boy, <character 1 tags>
ADDCOL
1girl/1boy, <character 2 tags>

===STRICT RULES===
1. ONLY use words/phrases from the input. NEVER invent tags.
2. ADDCOMM appears EXACTLY ONCE. ADDBASE appears EXACTLY ONCE.
3. Every character section MUST start with: 1girl / 1boy / 1woman / 1man / 1person
4. Tags = comma-separated SHORT phrases (1-5 words). NO full sentences.
5. Scene/background/location/environment tags → before ADDCOMM ONLY
6. Style/quality/camera/lighting-style tags → after ADDCOMM, before ADDBASE ONLY
7. Character appearance tags → character sections only

===INPUT SPLIT (CRITICAL)===
Look for the COSPLAY keyword or ",1girl," / ",1boy," / ",1woman," / ",1man," separator in the input.
  PART A = all text BEFORE "COSPLAY" (or before the second ",1girl," / ",1boy," separator)
  PART B = all text AFTER "COSPLAY," (including the cosplay character name and tags)
  → Extract scene/location tags from PART A → put BEFORE ADDCOMM
  → Extract style/quality tags from PART A → put AFTER ADDCOMM
  → Extract character appearance tags from PART A → put AFTER ADDBASE (starting with 1girl/1boy)
  → Put ALL of PART B tags → AFTER ADDCOL (starting with 1girl, cosplay, ...)
  NEVER put PART B tags into ADDBASE.
  NEVER put PART A character tags into ADDCOL.

===EXAMPLE===
Input: Rainy city street at night, neon signs, wet pavement. A woman in red coat walks alone.,1girl, COSPLAY,rei_ayanami, neon_genesis, blue short hair, red eyes, white plugsuit, pale skin, quiet expression
PART A scene: rainy city street, neon signs, wet pavement, night
PART A char: 1woman, red coat, walking alone
PART B: cosplay, rei_ayanami, neon_genesis, blue short hair, red eyes, white plugsuit, pale skin, quiet expression
Output:
rainy city street, neon signs, wet pavement, night
ADDCOMM
ADDBASE
1woman, red coat, walking alone
ADDCOL
1girl, cosplay, rei_ayanami, neon_genesis, blue short hair, red eyes, white plugsuit, pale skin, quiet expression"""


# ══════════════════════════════════════════════════════════════════════════
# RPConverter Node
# ══════════════════════════════════════════════════════════════════════════
# ── AGP-style LoRA helpers ──────────────────────────────────────────────────

def _read_lora_trigger(filepath: str) -> str:
    """Extract trigger word from safetensors metadata.
    Priority: modelspec.trigger_phrase → activation text → ss_output_name → filename
    """
    import struct, json as _json, os as _os
    if not filepath.endswith(".safetensors"):
        return _os.path.splitext(_os.path.basename(filepath))[0]
    try:
        with open(filepath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            if header_size == 0 or header_size > 50 * 1024 * 1024:
                raise ValueError(f"invalid header size: {header_size}")
            meta = _json.loads(f.read(header_size).decode("utf-8")).get("__metadata__", {})
        for key in ("modelspec.trigger_phrase", "activation text", "ss_output_name"):
            t = meta.get(key, "").strip()
            if t:
                return t
        return _os.path.splitext(_os.path.basename(filepath))[0]
    except Exception as e:
        print(f"[RPConverter] LoRA metadata read failed ({_os.path.basename(filepath)}): {e}")
        return _os.path.splitext(_os.path.basename(filepath))[0]


def _resolve_lora_folder(folder_path: str) -> str:
    """Convert relative path to absolute based on ComfyUI root.
    node path: ComfyUI/custom_nodes/ComfyUI_RP_Cast/nodes/
    comfy_root: ComfyUI/  (3 levels up from node file)
    """
    import os as _os

    def _comfy_root():
        # nodes/ → ComfyUI_RP_Cast/ → custom_nodes/ → ComfyUI/
        node_dir = _os.path.dirname(_os.path.abspath(__file__))
        return _os.path.dirname(_os.path.dirname(_os.path.dirname(node_dir)))

    if not folder_path or not folder_path.strip():
        # Default: use folder_paths API first, then fallback
        try:
            import folder_paths as _fp
            lora_dirs = _fp.get_folder_paths("loras")
            if lora_dirs:
                return lora_dirs[0]
        except Exception:
            pass
        return _os.path.join(_comfy_root(), "models", "loras")

    if _os.path.isabs(folder_path):
        return folder_path

    # Relative path: base from ComfyUI root
    return _os.path.join(_comfy_root(), folder_path)


def _list_lora_files(folder_path: str) -> list:
    """List LoRA files in folder (.safetensors/.pt/.ckpt/.bin)."""
    import os as _os, glob as _glob
    resolved = _resolve_lora_folder(folder_path)
    if not resolved or not _os.path.isdir(resolved):
        print(f"[RPConverter] LoRA folder not found: {resolved!r}")
        return []
    files = []
    for ext in ("*.safetensors", "*.pt", "*.ckpt", "*.bin"):
        files.extend(_glob.glob(_os.path.join(resolved, "**", ext), recursive=True))
    return files


def _make_lora_tag(filepath: str) -> str:
    """filepath → <lora:relative_name:1.0> tag"""
    import os as _os
    try:
        import folder_paths as _fp
        lora_dirs = _fp.get_folder_paths("loras")
        lora_base = lora_dirs[0] if lora_dirs else ""
    except Exception:
        node_dir   = _os.path.dirname(_os.path.abspath(__file__))
        comfy_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(node_dir)))
        lora_base  = _os.path.join(comfy_root, "models", "loras")
    try:
        rel = _os.path.relpath(filepath, lora_base)
        rel_noext = _os.path.splitext(rel)[0].replace("\\", "/")
        return f"<lora:{rel_noext}:1.0>"
    except ValueError:
        name = _os.path.splitext(_os.path.basename(filepath))[0]
        return f"<lora:{name}:1.0>"



class RPConverter:
    CATEGORY = "Regional Prompter"
    cnr_id   = "ComfyUI_RP_Cast"

    @classmethod
    def INPUT_TYPES(cls):
        # Model list: ["(none - WD14 only)"] + Ollama models
        model_list = ["(none - WD14 only)"] + _OLLAMA_MODELS
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "placeholder":
                        "Describe the scene with characters in natural language.\n"
                        "e.g. A girl with blonde hair wearing a red dress and "
                        "a boy in a blue jacket are in a cafe.",
                }),
                "style_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "e.g. anime style, soft watercolor, cinematic, ...",
                    "tooltip":
                        "Style keywords appended between ADDCOMM and ADDBASE. "
                        "Applied after Ollama conversion.",
                }),
                "lora_directory": ("STRING", {
                    "default": "models\\loras",
                    "placeholder": "models/loras",
                    "tooltip":
                        "LoRA folder path (relative to ComfyUI root or absolute). "
                        "Leave empty to use ComfyUI default loras folder. "
                        "Used when lora_auto_apply is ON.",
                }),
                "lora_auto_apply": ("BOOLEAN", {
                    "default": False,
                    "label_on": "LoRA Auto ON",
                    "label_off": "LoRA Auto OFF",
                    "tooltip":
                        "ON: Scan lora_directory and match LoRA files whose name "
                        "keywords appear in the scene. "
                        "Appends matched <lora:name:1.0> tags and trigger words "
                        "to the converted RP prompt.",
                }),
                "ollama_host": ("STRING", {
                    "default": "http://127.0.0.1:11434",
                    "tooltip":
                        "Ollama server URL. Default: http://127.0.0.1:11434. "
                        "Change if Ollama runs on a different host or port.",
                }),
                "ollama_model": (model_list, {
                    "default": model_list[0],
                    "tooltip":
                        "Select Ollama model. "
                        "If list is empty: set correct ollama_host and run once to refresh. "
                        "Select '(none - WD14 only)' for keyword-only mode.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("RP_prompt",)
    FUNCTION      = "execute"

    def execute(self, prompt, style_prompt="", lora_directory="", lora_auto_apply=False, ollama_host="http://127.0.0.1:11434", ollama_model="gemma3:12b", debug=False):
        _dbg = print if debug else lambda *a, **kw: None

        use_ollama = (
            ollama_model and
            ollama_model != "(none - WD14 only)" and
            ollama_model.strip()
        )

        # ── Ollama availability: sync host from widget + retry ───────────
        # Override base URL from widget (user may have changed ollama_host)
        global _OLLAMA_BASE
        if ollama_host and ollama_host.strip():
            _OLLAMA_BASE = ollama_host.strip().rstrip("/")

        if use_ollama and not _OLLAMA_AVAILABLE:
            print(f"[RPConverter] Retrying Ollama at {_OLLAMA_BASE} ...")
            _fetch_ollama_models()

        # ── Ollama path ───────────────────────────────────────────────────
        if use_ollama:
            if not _OLLAMA_AVAILABLE:
                raise RuntimeError(
                    "[RPConverter] Ollama is not running.\n"
                    "→ Start Ollama:     ollama serve\n"
                    "→ Restart ComfyUI to reload the model list.\n"
                    f"→ Or set OLLAMA_HOST env var (current: {_OLLAMA_BASE})"
                )

            # Wrap user prompt with explicit rewrite instruction
            _raw = prompt.strip()

            # Pre-split: detect COSPLAY separator for explicit hint to model
            import re as _re
            _cosplay_match = _re.search(
                r",\s*(1(?:girl|boy|woman|man|person))\s*,\s*COSPLAY", _raw, _re.IGNORECASE)
            _comma_match = _re.search(
                r"(?<=[a-zA-Z0-9.!?])\s*,,\s*(1(?:girl|boy|woman|man|person))", _raw, _re.IGNORECASE)

            if _cosplay_match or _comma_match:
                _m = _cosplay_match or _comma_match
                _part_a = _raw[:_m.start()].strip().strip(',').strip()
                _part_b = _raw[_m.start()+1:].strip()
                user_prompt = (
                    f"Convert to RP format.\n"
                    f"\n"
                    f"PART A (Character 1): {_part_a}\n"
                    f"PART B (Character 2 - COSPLAY): {_part_b}\n"
                    f"\n"
                    f"Rules:\n"
                    f"- PART A scene/location tags → before ADDCOMM\n"
                    f"- PART A style/quality tags → after ADDCOMM\n"
                    f"- PART A character tags → ADDBASE (start with 1girl/1boy/1woman/1man)\n"
                    f"- PART B tags → ADDCOL ONLY (start with 1girl, cosplay, ...)\n"
                    f"- NEVER mix PART A and PART B tags\n"
                    f"\n"
                    f"Output:\n"
                )
            else:
                user_prompt = (
                    f"Convert to RP format. Use ONLY words from the input.\n"
                    f"\n"
                    f"Input: {_raw}\n"
                    f"\n"
                    f"Output:\n"
                )

            print(f"[RPConverter] calling Ollama model: {ollama_model}")
            _MAX_RETRY = 1
            rp_prompt = ""
            # For qwen3 models: prepend /no_think directive for double assurance
            _user_prompt_final = user_prompt
            if "qwen3" in ollama_model.lower():
                _user_prompt_final = "/no_think\n\n" + user_prompt

            for _attempt in range(1, _MAX_RETRY + 1):
                _result = _ollama_generate(
                    model=ollama_model,
                    prompt=_user_prompt_final,
                    system=_SYSTEM_PROMPT,
                )
                _has_addcomm = "ADDCOMM" in _result
                _has_addbase = "ADDBASE" in _result
                _dup_addcomm = _result.count("ADDCOMM") > 1
                _dup_addbase = _result.count("ADDBASE") > 1
                _is_valid = _has_addcomm and _has_addbase and not _dup_addcomm and not _dup_addbase

                if _is_valid:
                    rp_prompt = _result
                    print(f"[RPConverter] attempt {_attempt}: valid RP structure")
                    break
                else:
                    _issues = []
                    if not _has_addcomm: _issues.append("ADDCOMM missing")
                    if not _has_addbase: _issues.append("ADDBASE missing")
                    if _dup_addcomm:     _issues.append(f"ADDCOMM duplicated ({_result.count('ADDCOMM')}x)")
                    if _dup_addbase:     _issues.append(f"ADDBASE duplicated ({_result.count('ADDBASE')}x)")
                    print(f"[RPConverter] attempt {_attempt}: invalid output "
                          f"({_issues}) → retry")
                    if _attempt < _MAX_RETRY:
                        _hint = (
                            f"INVALID OUTPUT: {_issues}\n"
                            f"ADDCOMM must appear EXACTLY ONCE. ADDBASE must appear EXACTLY ONCE.\n"
                            f"Do NOT repeat these keywords. Fix and retry.\n"
                            f"Previous output was:\n{_result}\n\n"
                        )
                        user_prompt = _hint + user_prompt
            if not rp_prompt:
                print(f"[RPConverter] all retries failed → using WD14 fallback")
                rp_prompt = self._wd14_convert(prompt, _dbg)
            # Unload model from VRAM after conversion
            _ollama_unload(ollama_model)
            _dbg(f"[RPConverter] Ollama raw output:\n{rp_prompt}")

            # Validate: must contain at least ADDBASE
            if "ADDBASE" not in rp_prompt:
                print(f"[RPConverter] WARNING: Ollama output missing ADDBASE — "
                      f"falling back to WD14 mode")
                rp_prompt = self._wd14_convert(prompt, _dbg)

        # ── WD14-only path ────────────────────────────────────────────────
        else:
            _dbg("[RPConverter] WD14-only mode (no Ollama model selected)")
            rp_prompt = self._wd14_convert(prompt, _dbg)

        # ── style_prompt: append between ADDCOMM and ADDBASE ────────────
        if style_prompt and style_prompt.strip():
            _sp = style_prompt.strip()
            _sp_lines = rp_prompt.split("\n")
            _sp_out   = []
            i = 0
            while i < len(_sp_lines):
                _sp_out.append(_sp_lines[i])
                if _sp_lines[i].strip() == "ADDCOMM":
                    # Collect ADDCOMM content until ADDBASE
                    i += 1
                    while i < len(_sp_lines) and _sp_lines[i].strip() != "ADDBASE":
                        _sp_out.append(_sp_lines[i])
                        i += 1
                    # Append style_prompt before ADDBASE
                    _sp_out.append(_sp)
                    continue  # ADDBASE line will be added in next iteration
                i += 1
            rp_prompt = "\n".join(_sp_out)
            _dbg(f"  [style_prompt] appended: {_sp[:80]}")

        # ── LoRA auto-apply: random 1 LoRA per COL section ──────────────
        if lora_auto_apply:
            import random as _rand
            _lora_files = _list_lora_files(lora_directory)
            if not _lora_files:
                print(f"[RPConverter] LoRA auto-apply: no LoRA files found in {lora_directory!r}")
            else:
                # Split rp_prompt into lines and find ADDBASE / ADDCOL positions
                _lines = rp_prompt.split("\n")
                _out   = []
                _col_n = 0  # 0 = ADDBASE section, 1+ = ADDCOL sections
                _in_char = False  # True after ADDBASE keyword
                _used   = []  # track used files to avoid duplicates

                def _pick_lora(used):
                    """Pick random LoRA not yet used in this prompt."""
                    available = [f for f in _lora_files if f not in used]
                    if not available:
                        available = _lora_files  # reset if all used
                    chosen = _rand.choice(available)
                    used.append(chosen)
                    tag     = _make_lora_tag(chosen)
                    trigger = _read_lora_trigger(chosen)
                    return tag, trigger

                i = 0
                while i < len(_lines):
                    line = _lines[i]
                    stripped = line.strip()

                    if stripped == "ADDBASE":
                        _in_char = True
                        _col_n   = 0
                        _out.append(line)
                        i += 1
                        # Collect this character's content lines
                        _char_lines = []
                        while i < len(_lines) and _lines[i].strip() not in ("ADDCOL", "ADDCOMM", "ADDBASE", "ADDROW"):
                            _char_lines.append(_lines[i])
                            i += 1
                        # Append LoRA to last content line
                        _ltag, _ltrig = _pick_lora(_used)
                        if _char_lines:
                            _last = _char_lines[-1].rstrip()
                            _lora_suffix = f", {_ltrig}, {_ltag}" if _ltrig else f", {_ltag}"
                            _char_lines[-1] = _last + _lora_suffix
                        else:
                            _lora_suffix = f"{_ltrig}, {_ltag}" if _ltrig else _ltag
                            _char_lines.append(_lora_suffix)
                        _out.extend(_char_lines)
                        print(f"[RPConverter] DIV[0] LoRA: {_ltag}  trigger={_ltrig!r}")
                        continue

                    elif stripped == "ADDCOL":
                        _col_n += 1
                        _out.append(line)
                        i += 1
                        # Collect this character's content lines
                        _char_lines = []
                        while i < len(_lines) and _lines[i].strip() not in ("ADDCOL", "ADDCOMM", "ADDBASE", "ADDROW"):
                            _char_lines.append(_lines[i])
                            i += 1
                        # Append LoRA to last content line
                        _ltag, _ltrig = _pick_lora(_used)
                        if _char_lines:
                            _last = _char_lines[-1].rstrip()
                            _lora_suffix = f", {_ltrig}, {_ltag}" if _ltrig else f", {_ltag}"
                            _char_lines[-1] = _last + _lora_suffix
                        else:
                            _lora_suffix = f"{_ltrig}, {_ltag}" if _ltrig else _ltag
                            _char_lines.append(_lora_suffix)
                        _out.extend(_char_lines)
                        print(f"[RPConverter] DIV[{_col_n}] LoRA: {_ltag}  trigger={_ltrig!r}")
                        continue

                    else:
                        _out.append(line)
                        i += 1

                rp_prompt = "\n".join(_out)

        print(f"[RPConverter] output:\n{rp_prompt}")
        return (rp_prompt,)

    # ── WD14 keyword matching conversion ──────────────────────────────────
    @staticmethod
    def _wd14_convert(prompt: str, _dbg) -> str:
        """Fallback: WD14 tag matching + rule-based classification."""
        matched = _match_wd14_tags(prompt)
        _dbg(f"[RPConverter] WD14 matched ({len(matched)}): {matched[:15]}")

        # Classify
        BG = re.compile(
            r"^(?:cafe|restaurant|bar|office|bedroom|classroom|park|"
            r"street|city|forest|beach|ocean|sky|elevator|room|"
            r"indoor|outdoor|background|scenery|building|interior|"
            r"ceiling|floor|wall|window|mirror|stairs|hallway|corridor|"
            r"night|day|sunset|sunrise|morning|evening|"
            r"(?:simple|white|black|gradient|abstract)\s+background|bokeh|"
            r"depth of field|lighting|shadow|reflection|"
            r"(?:soft|warm|cool|dim|bright)\s+(?:light|lighting))(?:\s.*)?$",
            re.I)
        STYLE = re.compile(
            r"^(?:masterpiece|best quality|ultra.?detailed|highres|absurdres|"
            r"photorealistic|realistic|hyperrealistic|cinematic|film grain|"
            r"4k|8k|hd|anime|illustration|digital art|oil painting|watercolor|"
            r"soft lighting|dramatic lighting|studio lighting|"
            r"bokeh|sharp focus|portrait|close.?up|full body|upper body|"
            r"solo|looking at viewer|smile|depth of field)(?:\s.*)?$",
            re.I)
        PERSON = re.compile(
            r"^(?:\d*\s*(?:girl|boy|woman|man|female|male|lady)s?)$", re.I)
        CLOTH = re.compile(
            r"^(?:dress|skirt|shirt|blouse|jacket|coat|suit|uniform|"
            r"sweater|hoodie|cardigan|vest|pants|jeans|shorts|leggings|"
            r"stockings|thighhighs|swimsuit|bikini|apron|kimono|yukata|"
            r"shoes|boots|heels|sneakers|sandals|loafers|slippers|"
            r"hat|cap|ribbon|hairband|glasses|sunglasses|mask|scarf|"
            r"gloves|belt|tie|collar|choker|bag|purse|backpack|"
            r"earring|earrings|necklace|bracelet|ring|jewelry|"
            r"(?:red|blue|green|black|white|yellow|pink|purple|orange|"
            r"brown|grey|gray|dark|light)\s+"
            r"(?:dress|skirt|shirt|jacket|coat|pants|jeans|shoes|boots|"
            r"sneakers|bag|hat))(?:\s.*)?$",
            re.I)

        bg_tags, style_out, char_blocks = [], [], []
        current = None

        for tag in matched:
            if PERSON.match(tag):
                m = re.match(r"^(\d*)\s*(girl|boy|woman|man|female|male|lady)s?$",
                             tag, re.I)
                count = m.group(1) or "1"
                kind  = m.group(2).lower()
                if current is not None:
                    char_blocks.append(current)
                current = {"trigger": f"{count}{kind}", "tags": []}
            elif BG.match(tag):
                bg_tags.append(tag)
            elif STYLE.match(tag):
                style_out.append(tag)
            else:
                if current is not None:
                    current["tags"].append(tag)
                else:
                    bg_tags.append(tag)

        if current is not None:
            char_blocks.append(current)

        # Merge style
        all_style = style_out[:]
        if not all_style:
            all_style = ["masterpiece", "best quality"]

        # Assemble
        lines = [", ".join(bg_tags) if bg_tags else "(scene)",
                 "ADDCOMM",
                 ", ".join(all_style),
                 "ADDBASE"]

        if not char_blocks:
            lines.append(prompt.strip())
        else:
            for i, cb in enumerate(char_blocks):
                col = ", ".join([cb["trigger"]] + cb["tags"])
                lines.append(col)
                if i < len(char_blocks) - 1:
                    lines.append("ADDCOL")

        return "\n".join(lines)
