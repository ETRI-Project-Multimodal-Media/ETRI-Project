import os
import re
import json
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from jsonschema import validate, ValidationError

from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["video_id", "event_id", "tags", "objects", "actors", "event", "policy", "LOD"],
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
            "required": ["actor_id", "ref_object", "role", "entity"],
            "properties": {
            "actor_id": {"type": "string", "pattern": r"^A\d{3,}$"},
            "ref_object": {"type": "string"},
            "role": {"type": "string"},
            "entity": {"type": "string"}
            }
        }
        },
        "event": {
        "type": "object",
        "additionalProperties": False,
        "required": ["event_id", "name", "type", "time", "actors", "objects"],
        "properties": {
            "event_id": {"type": "string"},
            "name": {"type": "string"},
            "type": {"type": "string"},
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
        "policy": {
        "type": "object",
        "additionalProperties": False,
        "required": ["audience_filter", "priority"],
        "properties": {
            "audience_filter": {
            "type": "array",
            "items": {"type": "string", "enum": ["adult_mode", "child_mode"]},
            "uniqueItems": True,
            },
            "priority": {"type": "string", "enum": ["high", "mid", "low"]}
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

def llm(system_prompt,user_prompt ):
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
    return response

def split_modality_caption_with_llm(caption_text: str) -> str:
    """LLM을 이용해 Visual/Audio로 분리"""
    # Few-shot 프롬프트
    system_prompt = (
    "You are an extraction engine. Output ONLY valid JSON with exactly two keys:\n"
    '{"visual": string, "audio": string}\n'
    "- visual: strictly what is seen.\n"
    "- audio: ONLY non-speech sounds (music, noise, sfx). EXCLUDE speech/lyrics/dialogue.\n"
    '- If no audio cues exist, set "audio" to "".\n'
    "- Use the same language as the input.\n"
    "- Do not invent unseen/inaudible details.\n"
    "- JSON only. No extra text, no code fences, no trailing commas."
    )

    # 예시를 빼고 돌려도 되지만, 안정성 위해 초소형 예시 1개(≤120 tokens) 권장
    user_prompt = (
        'Example:\n'
        'Input: "A woman plays piano while singing. Applause is heard."\n'
        '{"visual":"A woman plays the piano.","audio":"Applause is heard."}\n'
        '---\n'
        f'Input: "{caption_text}"\n'
        "Output JSON:"
    )

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
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )

    text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

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
        messages2 = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": repair_prompt},
        ]
        input_ids2 = tokenizer.apply_chat_template(
            messages2, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs2 = model.generate(
            input_ids2,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        text2 = tokenizer.decode(outputs2[0][input_ids2.shape[-1]:], skip_special_tokens=True).strip()
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

    return {"visual": visual.strip(), "audio": audio.strip()}


# 시간 구간 추출
def split_segments(caption: str):
    pattern = r'From (\d+) to (\d+), (.*?)(?=From \d+ to \d+|$)'
    matches = re.findall(pattern, caption, flags=re.S)
    return [(m[0], m[1], m[2].strip()) for m in matches]

# answer 추출
def extract_answer_from_line(line: str) -> str:
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

    return line[start + 1:end]

# video_id 추출
def extract_videoid_from_line(line: str) -> str:
    """
    한 줄 문자열에서 "video_id": "..." 부분만 뽑아냄
    """
    key = '"video_id":'
    start = line.find(key)
    if start == -1:
        return ""

    # video_id 뒤 첫 따옴표
    start = line.find('"', start + len(key))
    if start == -1:
        return ""

    # video_id 끝 따옴표
    end = line.find('"', start + 1)
    if end == -1:
        return ""

    return line[start + 1:end]

# === LLM 호출 프롬프트 ===
def extract_info_with_llm(video_id, seg_idx, start, end, text):
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
    - "policy.audience_filter" is an array containing one of: ["adult_mode"] or ["child_mode"] (choose one or empty if unknown).
    - "policy.priority" is one of: "high" | "mid" | "low".
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
            "ref_object": string,
            "role": string,
            "entity": string
            }
        ],
        "event": {
            "event_id": string,
            "name": string,
            "type": string,
            "time": { "start": string, "end": string },
            "actors": string[],
            "objects": string[]
        },
        "policy": {
            "audience_filter": string[],
            "priority": "high" | "mid" | "low"
        },
        "LOD": {
            "abstract_topic": string[],
            "scene_topic": string,
            "summary": string,
            "implications": string
        }
    }
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
        obj = json.loads(response)
        validate(obj, SCHEMA)
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
        obj2 = json.loads(text2)
        validate(obj2, SCHEMA)
        return obj2


def extract_speech_from_caption_with_llm(caption_text: str, speech_summary: str, start: str, end: str) -> str:
    """
    LLaMA를 이용해 특정 시간대 caption + 전체 speech summary를 기반으로
    그 시간대의 speech 내용을 추출
    """

    prompt = f"""
    You are a helpful assistant that extracts spoken speech at a given video time segment.

    - You are given:
      1. The time range of the segment (start to end).
      2. The multimodal caption for a specific time segment of the video.
      3. The overall speech summary of the entire video.

    - Your tasks:
      1. From the overall speech summary, extract the part that is most relevant
      to the given caption and time range ({start}-{end}).
      Total Range is (00 - 90).
      2. Then summarize that extracted speech into a concise form.
      3. If there is no relevant speech, output "None".


    ---


    ### Example
    Time: 00–10
    Caption: "the man in the suit continues his speech, gesturing with his hands as he addresses the audience in the well-lit room."
    Speech summary: "Taking a picture of miles of it. There's old Tommy right there. Tommy Wilson. Okay, here we are. Mandatory picture of a picture being taken. Oregon and the Rose Bowl. We did that. Oh well.  All right, yeah, that's a good one who's the guy the what All righty  Tom Wilson we work together at KWU.  across the English Channel, the pill or hovercraft? Guess which one? The science prize that year.
    Thank you.  or even shorter shifts there at that time. But anyway, for about six months I would play the song, Isaac Brothers,
    Don't Let Go, Hear the Whistle, It's 10 O'Clock, and I'd play it every night. 
    And finally, one time I got a call, hotline, didn't I hear you play that song like last week? And I was, 
    I'd been playing it every night for about six months. So yeah, we would try to get away with whatever we could and have fun. 
    Now Dave, you served a stint after being a music director in Disc Jockey at KFRC, coming out of San Jose and SF State.
    Was your style or did you  you encountered his style as well. No, that wasn't mine. I'd love to listen, but I wouldn't take it to, you know,
    Paul took it to another."
    Output: " Taking a picture of miles of it. There's old Tommy right there. Tommy Wilson. Okay, here we are. Mandatory picture of a picture being taken. Oregon and the Rose Bowl. We did that. Oh well.  All right, yeah, that's a good one who's the guy the what All righty  Tom Wilson we work together at KWU.  across the English Channel, the pill or hovercraft? Guess which one? The science prize that year.
    Thank you.  or even shorter shifts there at that time. But anyway, for about six months I would play the song, Isaac Brothers"

    ### Example
    Time : 70-90
    Caption: "Visual: A car is driving fast. Audio: Engine noise is loud."
    Speech summary: "No speech content in this scene."
    Output: "None"

    ---

    ### Now process this input:
    
    Time: {start}-{end}
    Caption: "{caption_text}"
    Speech summary: "{speech_summary}"

    Output (speech only, extracted from the summary):
    """


    messages = [
        {"role": "system", "content": "You are an assistant that extracts speech lines from multimodal captions using video-level speech summary as additional context."},
        {"role": "user", "content": prompt},
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
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()


# 기존 BART 대신 Pegasus 사용
# summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def chunk_text(text, max_chars=1500):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def summarize_text(text, max_len=80):
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        inputs = summarizer.tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=1024
        ).to(summarizer.model.device)

        summary_ids = summarizer.model.generate(
            **inputs,
            max_length=max_len,
            min_length=min(20, max_len-1),
            do_sample=False
        )
        summaries.append(summarizer.tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    return " ".join(summaries)


def translate_speech(video_id, speech_json_dir):
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

    return summary


def parse_split_caption_to_dict(split_caption: str, speech_timesplit : str) -> dict:
    """
    'Visual: "..." Audio: "..." Speech: "..."' 형태의 문자열을 dict로 변환
    """
    result = {}
    # Visual:, Audio:, Speech: 뒤 따옴표 안 내용을 추출
    matches = re.findall(r'(Visual|Audio):\s*"?([^"]+)"?', split_caption)
    for key, value in matches:
        result[key] = value.strip()
    result["Speech"] = speech_timesplit
    return result



# 전체 
def process_txt_file(input_file, output_file, speech_json_dir):
    results = []

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
        video_id = extract_videoid_from_line(line)
        
        # speech translation with summarized
        speech_translation = translate_speech(video_id, speech_json_dir)       
         
        for idx, (start, end, text) in enumerate(segments):
            result = extract_info_with_llm(video_id, idx, start, end, text)

            # Visual, Audio modality split
            split_result = split_modality_caption_with_llm(text)
            
            # speech summary 
            speech_timesplit = extract_speech_from_caption_with_llm(text, speech_translation)
            # modality to dict
            split_result_dict = parse_split_caption_to_dict(split_result, speech_timesplit)
                        
            # 결과 저장
            result_entry = {
                "video_id" : video_id,
                "event_id": idx,
                "original_answer": text,
                "llm_result" : result,
                "split_caption": split_result_dict
            }
            results.append(result_entry)
        
            # append 저장
            with open(output_file, "a", encoding="utf-8") as out_f:
                out_f.write(str(result_entry) + "\n")

    return results


# === 실행 예시 ===
if __name__ == "__main__":
    input_file = "/home/kylee/kylee/LongVALE/logs/eval.txt"      # 처리할 TXT 파일
    output_file = "/home/kylee/kylee/LongVALE/logs/modality_split.txt"   # 결과 저장 파일
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    process_txt_file(input_file, output_file, speech_json_dir="/home/kylee/kylee/LongVALE/data/speech_asr_1171")
