import os
import re
import json
import ast
import math
import time
import psutil
import wandb
from typing import Optional, Tuple
from tqdm import tqdm
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from jsonschema import validate, ValidationError
import torch, re, textwrap


from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"


# wandb 초기화 (파일 상단에 추가)
wandb.init(
    project="postprocess_comparison2", 
    name=f"ner_merge_{int(time.time())}",
    config={
        "version": "ner_merge",
        "model": model_id,
        "description": "NER version with merge optimization",
        "features": ["ner_processing", "merge_optimization", "llm_calls"]
    },
    tags=["ner_merge", "optimized", "comparison"]
)

def get_gpu_memory():
    """GPU 메모리 사용량 반환 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def log_function_performance(func_name, start_time, end_time, start_memory, end_memory):
    """함수 실행 시간과 메모리 사용량을 wandb와 로그 파일에 기록"""
    duration = end_time - start_time
    memory_used = end_memory - start_memory
    
    # wandb 로깅
    wandb.log({
        f"{func_name}_duration": duration,
        f"{func_name}_memory_used": memory_used,
        f"{func_name}_end_memory": end_memory
    })
    
    # 로그 파일에 기록
    log_entry = {
        "function": func_name,
        "duration_seconds": duration,
        "memory_used_mb": memory_used,
        "end_memory_mb": end_memory,
        "timestamp": time.time()
    }
    
    with open("/home/kylee/kylee/LongVALE/logs/performance_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["video_id", "event_id", "tags", "objects", "actors", "event",  "LOD"],
    "properties": {
        "video_id": {"type": "string"},
        "event_id": {
        "type": "string",
        "pattern": r"^E_.+$"
        },
        "tags": {
        "type": "array",
        "items": {"type": "string"},
        },
        "objects": {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["object_id", "name", "attributes"],
            "properties": {
            "object_id": {"type": "string", "pattern": r"^O\d{3,}$"},
            "name": {"type": "string"},
            "attributes": {
                "type": "object",
                "additionalProperties": {"type": "string"}
            }
            }
        }
        },
        "actors": {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["actor_id","name", "ref_object", "role", "entity"],
            "properties": {
            "actor_id": {"type": "string", "pattern": r"^A\d{3,}$"},
            "name": {"type": "string"},
            "ref_object": {"type": "string"},
            "role": {"type": "string"},
            "entity": {"type": "string"}
            }
        }
        },
        "event": {
        "type": "object",
        "additionalProperties": False,
        "required": ["event_id", "name",  "time", "actors", "objects"],
        "properties": {
            "event_id": {"type": "string"},
            "name": {"type": "string"},
            "time": {
            "type": "object",
            "additionalProperties": False,
            "required": ["start", "end"],
            "properties": {
                "start": {"type": "string"},
                "end": {"type": "string"}
            }
            },
            "actors": {"type": "array", "items": {"type": "string"}},
            "objects": {"type": "array", "items": {"type": "string"}}
        }
        },
        "LOD": {
        "type": "object",
        "additionalProperties": False,
        "required": ["abstract_topic", "scene_topic", "summary", "implications"],
        "properties": {
            "abstract_topic": {"type": "array", "items": {"type": "string"}},
            "scene_topic": {"type": "string"},
            "summary": {"type": "string"},
            "implications": {"type": "string"}
        }
        }
    }
    }

#clean
META_PREFIX_RE = re.compile(
    r"""^\s*
        (?:
          (?:[Tt]he\s+)?(?:speaker|narrator|host|video|segment|scene|clip)\s+
          (?:says|says\s+that|mentions|talks\s+about|discusses|describes|announces|states|explains)\s*:?\s*
        )
    """, re.X)

def clean_meta(text: str) -> str:
    text = text.strip()
    text = META_PREFIX_RE.sub("", text)          # 메타 프리픽스 제거
    text = re.sub(r"^(```|Output\s*:|Answer\s*:)\s*", "", text, flags=re.I).strip()
    text = text.strip().strip('“”"\'')           # 양끝 따옴표 제거(내부 인용은 유지)
    text = text.splitlines()[0].strip()          # 한 줄만
    return text or "None"

def llm(system_prompt,user_prompt):
    start_time = time.time()
    start_memory = get_gpu_memory()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    response = response.strip()
    end_time = time.time()
    end_memory = get_gpu_memory()
    log_function_performance("llm", start_time, end_time, start_memory, end_memory)
    return response

def split_modality_caption_with_llm(caption_text: str) -> str:
    start_time = time.time()
    start_memory = get_gpu_memory()
    
    """LLM을 이용해 Visual/Audio로 분리"""
    # Few-shot 프롬프트
    system_prompt = """
    You output ONLY valid JSON with exactly two keys:
    {"visual": string, "audio": string}

    DEFINITIONS
    - visual: strictly what is seen.
    - audio: ANY audible content explicitly mentioned in the text, including
    speech/dialogue/announcer/narration, crowd reactions (cheer, boo, applause),
    music/singing, environmental/FX (engine, footsteps, wind, beep, whistle, horn), alarms.

    EXTRACTION RULES
    - Find ALL sound mentions; do not drop any. Preserve the order in the input.
    - Convert each sound mention to a concise phrase (noun/verb). Keep quotes for speech if present.
    - Join multiple audio mentions with ", " (comma+space).
    - Do NOT invent unheard details. If none, set "audio" to "".
    - Keep the same language as the input.
    - JSON only. No extra text, no code fences, no trailing commas.
    """

    # 예시를 빼고 돌려도 되지만, 안정성 위해 초소형 예시 1개(≤120 tokens) 권장
    user_prompt = f"""
        Example:
        Input: "A woman plays the piano while singing 'I love you'. Applause erupts and a bell rings."
        Output:
        {{"visual":"A woman plays the piano.","audio":"Singing 'I love you', applause, bell rings."}}
        ---
        Now process the segment below.

        Input: {caption_text}
        Output JSON:
        """
    text = llm(system_prompt, user_prompt)
    # 1차 파싱
    import json, re
    def try_parse_json(s: str):
        # 응답 앞에 잡음이 섞였을 때 첫 '{'부터 추출
        i = s.find("{")
        if i != -1:
            s = s[i:]
        return json.loads(s)

    try:
        obj = try_parse_json(text)
    except Exception:
        # 자동 리페어 프롬프트(간단)
        repair_prompt = (
            "The previous output was not valid JSON with keys {\"visual\",\"audio\"}.\n"
            "Fix it to valid JSON now. Do not add extra keys.\n"
            f"Previous:\n{text}\n"
            "Output JSON only:"
        )
        text2 = llm(system_prompt, repair_prompt)
        obj = try_parse_json(text2)

    # 최종 후처리: 키 보정 & 타입 보장
    visual = obj.get("visual", "")
    audio  = obj.get("audio", "")
    if not isinstance(visual, str): visual = str(visual)
    if not isinstance(audio, str):  audio  = str(audio)

    # 규칙 준수 보정: 스피치 흔적이 audio에 들어오면 제거(간단한 패턴 예시)
    speech_markers = [r"\bsaid\b", r"\bsays\b", r"\bquote\b", r"\".+?\""]
    if any(re.search(p, audio, flags=re.IGNORECASE) for p in speech_markers):
        audio = ""  # 필요시 더 정교한 필터 로직 적용

    end_time = time.time()
    end_memory = get_gpu_memory()
    log_function_performance("split_modality_caption_with_llm", start_time, end_time, start_memory, end_memory)
    return {"visual": visual.strip(), "audio": audio.strip()}


# 시간 구간 추출
MIN_SEGMENT_REPEAT = 2
MIN_NGRAM_REPEAT = 3
NGRAM_OVERLAP_THRESHOLD = 0.8
KL_SIMILARITY_THRESHOLD = 0.01
JSD_SIMILARITY_THRESHOLD = 0.03
KL_EPS = 1e-8

def split_segments(caption: str):
    # pattern = r'From (\d+) to (\d+), (.*?)(?=From \d+ to \d+|$)' # 정수 
    pattern = r'From (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?), (.*?)(?=From \d+(?:\.\d+)? to \d+(?:\.\d+)?|$)'
    matches = re.findall(pattern, caption, flags=re.S)
    return [(m[0], m[1], m[2].strip()) for m in matches]

def merge_segments_by_text(segments, min_repeat_count=MIN_SEGMENT_REPEAT):
    if not segments:
        return []

    merged = []
    idx = 0
    total = len(segments)

    while idx < total:
        start, end, text = segments[idx]
        repeat_count = 1
        last_end = end
        next_idx = idx + 1

        while next_idx < total and segments[next_idx][2] == text:
            repeat_count += 1
            last_end = segments[next_idx][1]
            next_idx += 1

        if repeat_count >= min_repeat_count:
            merged.append((start, last_end, text))
        else:
            merged.extend(segments[idx:next_idx])

        idx = next_idx

    return merged

def _ngram_overlap_ratio(text_a: str, text_b: str, n: int) -> float:
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    if len(tokens_a) < n or len(tokens_b) < n:
        return 0.0

    ngrams_a = {tuple(tokens_a[i:i+n]) for i in range(len(tokens_a) - n + 1)}
    ngrams_b = {tuple(tokens_b[i:i+n]) for i in range(len(tokens_b) - n + 1)}
    if not ngrams_a or not ngrams_b:
        return 0.0

    overlap = ngrams_a & ngrams_b
    denom = min(len(ngrams_a), len(ngrams_b))
    return len(overlap) / denom if denom else 0.0

def repeated_ngram_ratio(segments, min_ngram_repeat=MIN_NGRAM_REPEAT, overlap_threshold=NGRAM_OVERLAP_THRESHOLD):
    start_time = time.time()
    start_memory = get_gpu_memory()
    
    if not segments:
        return []

    merged = []
    idx = 0
    total = len(segments)

    while idx < total:
        start, end, text = segments[idx]
        first_text = text
        prev_text = text
        last_end = end
        next_idx = idx + 1

        while next_idx < total:
            next_text = segments[next_idx][2]
            ratio = _ngram_overlap_ratio(prev_text, next_text, min_ngram_repeat)
            if ratio < overlap_threshold:
                break
            last_end = segments[next_idx][1]
            prev_text = next_text
            next_idx += 1

        merged_text = first_text.strip()
        merged.append((start, last_end, merged_text))
        idx = next_idx

    end_time = time.time()
    end_memory = get_gpu_memory()
    log_function_performance("repeated_ngram_ratio", start_time, end_time, start_memory, end_memory)
    return merged

def _token_distribution(text: str) -> dict[str, float]:
    tokens = text.split()
    if not tokens:
        return {}

    counter = Counter(tokens)
    total = sum(counter.values())
    if total == 0:
        return {}

    return {token: count / total for token, count in counter.items()}

def _kl_divergence(p: dict[str, float], q: dict[str, float], eps: float = KL_EPS) -> float:
    if not p:
        return 0.0

    tokens = set(p) | set(q)
    divergence = 0.0
    for token in tokens:
        p_val = p.get(token, eps)
        q_val = q.get(token, eps)
        divergence += p_val * math.log((p_val + eps) / (q_val + eps))
    return divergence

def merge_segments_by_kl(segments, similarity_threshold=KL_SIMILARITY_THRESHOLD):
    start_time = time.time()
    start_memory = get_gpu_memory()
    if not segments:
        return []

    merged = []
    idx = 0
    total = len(segments)

    while idx < total:
        start, end, text = segments[idx]
        first_text = text
        prev_dist = _token_distribution(text)
        last_end = end
        next_idx = idx + 1

        while next_idx < total:
            next_text = segments[next_idx][2]
            next_dist = _token_distribution(next_text)
            divergence = _kl_divergence(prev_dist, next_dist)
            if divergence >= similarity_threshold:
                break
            last_end = segments[next_idx][1]
            prev_dist = next_dist
            next_idx += 1

        merged.append((start, last_end, first_text.strip()))
        idx = next_idx

    end_time = time.time()
    end_memory = get_gpu_memory()
    log_function_performance("merge_segments_by_kl", start_time, end_time, start_memory, end_memory)
    return merged

def _js_divergence(p: dict[str, float], q: dict[str, float], eps: float = KL_EPS) -> float:
    # p,q가 모두 빈 분포면 0
    if not p and not q:
        return 0.0

    tokens = set(p) | set(q)
    # 공통 토큰 집합 기준으로 확률 벡터 구성
    P = {t: p.get(t, eps) for t in tokens}
    Q = {t: q.get(t, eps) for t in tokens}
    M = {t: 0.5 * (P[t] + Q[t]) for t in tokens}

    def _kl(A, B):
        s = 0.0
        for t in tokens:
            s += A[t] * math.log((A[t] + eps) / (B[t] + eps))
        return s

    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)

def merge_segments_by_jsd(segments, similarity_threshold=JSD_SIMILARITY_THRESHOLD):
    if not segments:
        return []

    merged = []
    idx = 0
    total = len(segments)

    while idx < total:
        start, end, text = segments[idx]
        first_text = text
        prev_dist = _token_distribution(text)
        last_end = end
        next_idx = idx + 1

        while next_idx < total:
            next_text = segments[next_idx][2]
            next_dist = _token_distribution(next_text)
            divergence = _js_divergence(prev_dist, next_dist)
            if divergence >= similarity_threshold:
                break
            last_end = segments[next_idx][1]
            prev_dist = next_dist
            next_idx += 1

        merged.append((start, last_end, first_text.strip()))
        idx = next_idx

    return merged

# JSON 보정 유틸
def _has_colon_outside_quotes(text: str) -> bool:
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string and ch == ':':
            return True
    return False

def _fix_attribute_content(content: str) -> str:
    lines = content.splitlines(keepends=True)
    fixed_lines = []
    attr_counter = 1

    for line in lines:
        if not line.strip():
            fixed_lines.append(line)
            continue

        body = line.rstrip("\r\n")
        newline = line[len(body):]
        stripped = body.strip()
        if not stripped:
            fixed_lines.append(line)
            continue

        has_trailing_comma = stripped.endswith(',')
        base = stripped[:-1].strip() if has_trailing_comma else stripped

        if base and not _has_colon_outside_quotes(base):
            indent = body[:len(body) - len(body.lstrip())]
            try:
                value_obj = json.loads(base)
            except Exception:
                value_obj = base.strip('"')
            value_json = json.dumps(value_obj, ensure_ascii=False)
            new_body = f'{indent}"attr_{attr_counter:03d}": {value_json}'
            if has_trailing_comma:
                new_body += ','
            fixed_lines.append(new_body + newline)
            attr_counter += 1
        else:
            fixed_lines.append(line)

    return "".join(fixed_lines)

def _find_matching_brace(text: str, open_brace_index: int) -> int:
    depth = 0
    in_string = False
    escape = False

    for idx in range(open_brace_index, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return idx
    return -1

def _fix_attributes_in_json(text: str) -> str:
    pattern = re.compile(r'"attributes"\s*:\s*\{', re.IGNORECASE)
    idx = 0
    result = text

    while True:
        match = pattern.search(result, idx)
        if not match:
            break

        content_start = match.end()
        closing_index = _find_matching_brace(result, match.end() - 1)
        if closing_index == -1:
            idx = content_start
            continue

        content = result[content_start:closing_index]
        fixed_content = _fix_attribute_content(content)
        result = result[:content_start] + fixed_content + result[closing_index:]
        idx = content_start + len(fixed_content)

    return result

def _normalize_object_attributes(obj: dict) -> None:
    objects = obj.get("objects")
    if not isinstance(objects, list):
        obj["objects"] = []
        return

    for item in objects:
        if not isinstance(item, dict):
            continue

        attrs = item.get("attributes")
        if isinstance(attrs, dict):
            normalized = {}
            for key, value in attrs.items():
                key_str = str(key)
                if isinstance(value, str):
                    normalized[key_str] = value
                elif isinstance(value, (int, float, bool)):
                    normalized[key_str] = str(value)
                elif value is None:
                    continue
                else:
                    normalized[key_str] = json.dumps(value, ensure_ascii=False)
            item["attributes"] = normalized
        elif isinstance(attrs, list):
            normalized = {}
            for idx, value in enumerate(attrs, start=1):
                normalized[f"attr_{idx:03d}"] = str(value)
            item["attributes"] = normalized
        elif isinstance(attrs, str):
            item["attributes"] = {"description": attrs}
        else:
            item["attributes"] = {}

# answer 추출
def extract_answer_from_line(line: str) -> str:
    start_time = time.time()
    start_memory = get_gpu_memory()
    
    """
    한 줄 문자열에서 "answer": "..." 부분만 뽑아냄
    """
    key = '"answer":'
    start = line.find(key)
    if start == -1:
        return ""

    # answer 뒤 첫 따옴표
    start = line.find('"', start + len(key))
    if start == -1:
        return ""

    # answer 끝 따옴표
    end = line.find('"', start + 1)
    if end == -1:
        return ""

    end_time = time.time()
    end_memory = get_gpu_memory()
    log_function_performance("extract_answer_from_line", start_time, end_time, start_memory, end_memory)
    return line[start + 1:end]

# video_id 추출
def extract_videoinform_from_line(line: str) -> Optional[Tuple[str, str]]:
    """
    한 줄 문자열에서 "video_id"와 "duration" 값을 추출해 (video_id, duration) 튜플로 반환한다.
    두 필드 중 하나라도 없으면 None을 반환한다.
    """
    def _find_value(key: str) -> Optional[str]:
        idx = line.find(key)
        if idx == -1:
            return None

        start = line.find('"', idx + len(key))
        if start == -1:
            return None

        end = line.find('"', start + 1)
        if end == -1:
            return None

        return line[start + 1:end]

    video_id = _find_value('"video_id":')
    duration = _find_value('"duration":')
    if video_id is None or duration is None:
        return None

    return video_id, duration

# === LLM 호출 프롬프트 ===
def extract_info_with_llm(video_id, seg_idx, start, end, text, not_json_dir):
    start_time = time.time()
    start_memory = get_gpu_memory()
    system_prompt = """
    You are a structured information extraction engine that outputs ONLY valid JSON for an MPEG-7–style schema.
    Follow ALL rules strictly:

    GENERAL
    - Output JSON ONLY. No code fences, no explanations, no trailing text.
    - Use double quotes for all keys/strings. No comments. No trailing commas.
    - If a field is unknown, use "" (empty string) or [] (empty array). Do NOT invent facts.
    - Keep key order exactly as the schema lists. Do not add extra keys.

    ID & REFERENTIAL INTEGRITY
    - "event_id" must be "E_<video_id>_<start>_<end>".
    - Sanitize <video_id> by replacing non-alphanumeric chars with "_" (keep case).
    - "objects[].object_id" must be unique like "O001", "O002", … (3 digits).
    - "objects[].attributes" MUST be a flat JSON object of string values only.
    - "actors[].actor_id" must be unique like "A001", "A002", … (3 digits).
    - "actors[].ref_object" must reference an existing objects[].object_id.
    - "event.actors" and "event.objects" are arrays of IDs that must exist above.

    TYPES & ENUMS
    - "time.start" and "time.end" are strings echoing the given inputs (no reformat).
    - "LOD.abstract_topic" is an array of strings; "scene_topic", "summary", "implications" are strings.

    SCHEMA (required keys in this exact order)
    {
        "video_id": string,
        "event_id": string,
        "tags": string[],
        "objects": [
        {
        "object_id": string,
        "name": string,
        "attributes": { string: string }
        }
        ],
        "actors": [
            {
            "actor_id": string,
            "name": string,
            "ref_object": string,
            "role": string,
            "entity": string
            }
        ],
        "event": {
            "event_id": string,
            "name": string,
            "time": { "start": string, "end": string },
            "actors": string[],
            "objects": string[]
        },
        "LOD": {
            "abstract_topic": string[],
            "scene_topic": string,
            "summary": string,
            "implications": string
        }
    }
    
    FIELD GUIDELINES (minimal)
    - "tags": 3–6 topic keywords in snake_case (nouns only; no punctuation).
    e.g., ["economy", "central_bank", "interest_rate"]
    - "LOD.abstract_topic": very coarse category label (keep input language).
    e.g., "economy news", "non-violent scene"
    - "LOD.scene_topic": one-sentence event summary (include actor/action; keep input language).
    e.g., "The Bank of Korea governor announces that the base rate is kept on hold."
    - "LOD.implications": short phrase on significance/impact.
    e.g., "Major economics,finance news event"
    """
    user_prompt = f"""

    Now process the following input:

    Input segment:
    Video {video_id}, time {start}-{end}, description: {text}

    Output JSON:
    """


    response = llm(system_prompt, user_prompt)
    SCHEMA_TEXT = json.dumps(SCHEMA, ensure_ascii=False, separators=(",", ":"))  # JSON 문자열로(토큰 절약)

    try:
        response = _fix_attributes_in_json(response)
        obj = json.loads(response)
        _normalize_object_attributes(obj)
        validate(obj, SCHEMA)
        end_time = time.time()
        end_memory = get_gpu_memory()
        log_function_performance("extract_info_with_llm", start_time, end_time, start_memory, end_memory)
        return obj
    except (json.JSONDecodeError, ValidationError):
        # 리페어는 system을 그대로 두고, user에 'ORIGINAL+SCHEMA'를 넣습니다.
        repair_user = (
            "The previous output was invalid JSON or did not match the schema. "
            "Fix it to match the JSON Schema EXACTLY. Return JSON ONLY.\n\n"
            f"SCHEMA:\n{SCHEMA_TEXT}\n\n"
            f"ORIGINAL:\n{response}\n"
        )
        text2 = llm(system_prompt, repair_user)
        try:
            text2 = _fix_attributes_in_json(text2)
            obj2 = json.loads(text2)
            _normalize_object_attributes(obj2)
            validate(obj2, SCHEMA)
            end_time = time.time()
            end_memory = get_gpu_memory()
            log_function_performance("extract_info_with_llm", start_time, end_time, start_memory, end_memory)
            return obj2
        except Exception as e:
            # append 저장
            error_dict = {'obj':text2, 'error': getattr(e, 'message', None) or str(e)}
            with open(not_json_dir, "a", encoding="utf-8") as out_f:
                out_f.write(str(error_dict) + "\n")
            raise


def extract_speech_from_caption_with_llm(caption_text: str, speech_summary: str, start: str, end: str, duration:str) -> str:
    start_time = time.time()
    start_memory = get_gpu_memory()
    """
    LLaMA를 이용해 특정 시간대 caption + 전체 speech summary를 기반으로
    그 시간대의 speech 내용을 추출
    """

    system_prompt = f"""
    You are a structured information extraction engine that extracts spoken speech at a given video time segment.
    RULES:
    - Output ONLY the speech text as one plain line. No labels, no quotes, no JSON, no code fences, no preambles (e.g., "The speaker ..."), no explanations.
    - Use only content present in the given speech summary; do NOT invent.
    - Select the portion most relevant to the segment {start}-{end} and its caption. Total Range is (00.00 - {duration}).
    - If no relevant speech exists, output: None
    - Keep the input language and keep it concise.
    - Preserve inner quotes if present. Output must NOT start with meta phrases.
    """

    user_prompt = f"""
    Now process this input:
    
    Time: {start}-{end}
    Caption: "{caption_text}"
    Speech summary: "{speech_summary}"

    Output (speech only, extracted from the summary):
    """
    output = llm(system_prompt, user_prompt)

    output = clean_meta(output)
    
    if not output or output.lower() in {"none", "(none)"}:
        return "None"
    
    end_time = time.time()
    end_memory = get_gpu_memory()
    log_function_performance("extract_speech_from_caption_with_llm", start_time, end_time, start_memory, end_memory)
    return output


def chunk_text(text, max_chars=1500):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def summarize_text(text, max_len=80):
    start_time = time.time()
    start_memory = get_gpu_memory()
    chunks = chunk_text(text)
    summaries = []
    system_prompt = textwrap.dedent("""
    You summarize speech transcripts into a single plain-text string.
    RULES:
    - Output only the summary text. No labels, no quotes, no bullets, no JSON, no code fences.
    - Keep the input language (do not translate).
    - Be faithful to the transcript; do not add facts that are not present.
    - Make it concise but complete.
    """)

    
    for chunk in chunks:
        user_prompt = f"Transcript:\n{chunk}\n\nSummarize in one concise paragraph (1–2 sentences):"
        chunk_summary = llm(system_prompt, user_prompt)
        summaries.append(chunk_summary)
    end_time = time.time()
    end_memory = get_gpu_memory()
    log_function_performance("summarize_text", start_time, end_time, start_memory, end_memory)
    return " ".join(summaries)


def translate_speech(video_id, speech_json_dir):
    start_time = time.time()
    start_memory = get_gpu_memory()
    # speech json 경로
    speech_json_path = os.path.join(speech_json_dir, f"{video_id}.json")
    if not os.path.isfile(speech_json_path):
        return

    # transcription 읽기
    with open(speech_json_path, "r", encoding="utf-8") as sf:
        speech_data = json.load(sf)
    transcription = speech_data.get("transcription", "")

    # 요약
    summary = summarize_text(transcription)
    end_time = time.time()
    end_memory = get_gpu_memory()
    log_function_performance("translate_speech", start_time, end_time, start_memory, end_memory)
    return summary

import json, re

def parse_split_caption_to_dict(split_caption, speech_timesplit=None):
    """
    split_caption: dict({'visual','audio'}) | JSON 문자열 | 'Visual: ...\nAudio: ...' 문자열
    반환: {'visual': str, 'audio': str}
    """
    visual = ""
    audio = ""

    # 1) 이미 dict인 경우
    if isinstance(split_caption, dict):
        visual = split_caption.get("visual") or ""
        audio  = split_caption.get("audio") or ""

    # 2) 문자열인 경우: 먼저 JSON 시도, 실패하면 라벨 파싱
    elif isinstance(split_caption, str):
        s = split_caption.strip()

        # 2-1) JSON 시도
        if s.startswith("{"):
            try:
                obj = json.loads(s)
                visual = obj.get("visual", "")
                audio  = obj.get("audio", "")
            except Exception:
                pass

        # 2-2) 라벨 포맷 파싱 (Visual: ..., Audio: ...)
        if not visual and not audio:
            m_vis = re.search(r'(?i)\bvisual\b\s*:\s*["“]?(.+?)["”]?(?:$|\n|;)', s)
            m_aud = re.search(r'(?i)\baudio\b\s*:\s*["“]?(.+?)["”]?(?:$|\n|;)', s)
            visual = (m_vis.group(1) if m_vis else "").strip()
            audio  = (m_aud.group(1) if m_aud else "").strip()

    # 3) 기타 타입(None 등)은 빈값 유지
    else:
        visual = visual or ""
        audio  = audio or ""

    # 4) 가벼운 후처리: 따옴표/공백/구분자 정리
    def clean(txt: str) -> str:
        txt = txt.strip().strip('"').strip("“”").strip()
        txt = re.sub(r"\s+", " ", txt)
        return txt

    visual = clean(visual)
    audio  = clean(audio)
    # 쉼표들을 세미콜론으로 정리(선호 시)
    audio = re.sub(r"\s*,\s*", "; ", audio).strip(" ;")

    return {"visual": visual, "audio": audio, 'speech':speech_timesplit}

def save_video_results(video_results, output_file):
    output_data = []
    for video_id, events in video_results.items():
        output_data.append({
            "video_id": video_id,
            "results": events
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"총 {len(output_data)}개 video 결과 저장 완료 → {output_file}")

def append_result(output_file, video_id, result_entry):
    rec = {"video_id": video_id, "result": result_entry}
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# 전체 
def process_txt_file(input_file, output_file, speech_json_dir, not_json_dir):
    start_time = time.time()
    start_memory = get_gpu_memory()
    video_results = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(tqdm(lines, desc="Processing lines")):
        line = line.strip()
        if not line:
            continue

        # answer만 추출
        caption_text = extract_answer_from_line(line)
        if not caption_text:
            continue
        
        # time split
        segments = split_segments(caption_text)
        # merge time split 
        segments = repeated_ngram_ratio(segments)
        segments = merge_segments_by_kl(segments)
        # unused
        # segments = merge_segments_by_text(segments) 
        video_info = extract_videoinform_from_line(line)
        if not video_info:
            continue
        video_id, duration = video_info
        
        # speech translation with summarized
        speech_translation = translate_speech(video_id, speech_json_dir)       
        
        for idx, (start, end, text) in enumerate(segments):
            result = extract_info_with_llm(video_id, idx, start, end, text, not_json_dir)

            # LLM 결과에 관계없이 실제 start/end로 덮어쓰기
            result.setdefault('event', {}).setdefault('time', {})
            result['event']['time']['start'] = str(start)
            result['event']['time']['end'] = str(end)

            # Visual, Audio modality split
            split_result = split_modality_caption_with_llm(text)
            
            # speech summary and extract time speech
            speech_timesplit = extract_speech_from_caption_with_llm(text, speech_translation, start, end, duration)
            # modality to dict
            split_result_dict = parse_split_caption_to_dict(split_result, speech_timesplit)
            
            result['LOD']['modalities'] = split_result_dict
            # 결과 저장
            result_entry = {
                "event_id": idx,
                "original_answer": text,
                "result" : result,
            }
            video_results[video_id].append(result_entry)

            # 추가: 나오자마자 JSONL로 저장
            # append_result(output_file, video_id, result_entry)

    save_video_results(video_results, output_file)
    
    end_time = time.time()
    end_memory = get_gpu_memory()
    log_function_performance("save_video_results", start_time, end_time, start_memory, end_memory)



# === 실행 예시 ===
if __name__ == "__main__":
    input_file = "/home/kylee/kylee/LongVALE/logs/inference_dvc_time.txt"      # 처리할 TXT 파일
    output_file = "/home/kylee/kylee/LongVALE/logs/result_1015_merge2.json"   # 결과 저장 json
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    process_txt_file(input_file, output_file, speech_json_dir="/home/kylee/kylee/LongVALE/data/speech_asr_1171", not_json_dir="/home/kylee/kylee/LongVALE/logs/wrong_sample_1015.txt")
