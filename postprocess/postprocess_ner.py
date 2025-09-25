import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 
def split_modality_caption_with_llm(caption_text: str) -> str:
    """LLM을 이용해 Visual/Audio/Speech로 분리"""
    # Few-shot 프롬프트
    prompt = """
    You are a helpful assistant that splits a multimodal caption into three parts: 
    Visual, Audio, and Speech.

    Examples:
    Input: "A woman is playing the piano while singing 'I love you.' Applause can be heard."
    Output:
    Visual: "A woman is playing the piano."
    Audio: "Applause can be heard."

    Input: "A man is playing the guitar while singing 'I love you.' The acoustic guitar sound resonates in the background."
    Output:
    Visual: "A man is playing the guitar."
    Audio: "The acoustic guitar sound resonates in the background."

    ---

    Now split the following:
    
    Input: "{}"
    Output:
    Visual:""".format(caption_text)

    messages = [
        {"role": "system", "content": "You are an assistant that classifies captions into Visual, Audio, and Speech."},
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
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()


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

# answer 추출
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
    prompt = f"""
    You are an information extraction model following MPEG-7 style schema.
    Return ONLY valid JSON (no explanations).  

    Here is the required JSON structure:
    - video_id
    - event_id: use format E_<video_id>_<start>_<end>
    - tags: topic keywords
    - objects: object_id, name, attributes
    - actors: actor_id, ref_object, role, entity
    - event: event_id, name, type, time, actors, objects
    - policy: audience_filter (adult_mode/child_mode), priority (high/mid/low)
    - LOD: abstract_topic, scene_topic, summary, implications

    ---

    ### Example
    Input segment:
    Video vHTfzg4dBsY, time 00-99, description: "On a sunny day at the track and field stadium, a female athlete in a red and white uniform stands poised, holding a javelin, as the crowd cheers and the announcer describes the event."

    Output JSON:
    {{
    "video_id": "vHTfzg4dBsY",
    "event_id": "E_vHTfzg4dBsY_00_99",
    "tags": ["sports", "javelin", "athletics"],
    "objects": [
        {{
        "object_id": "O001",
        "name": "javelin",
        "attributes": {{"type": "equipment"}}
        }},
        {{
        "object_id": "O002",
        "name": "stadium",
        "attributes": {{"type": "location", "environment": "outdoor"}}
        }}
    ],
    "actors": [
        {{
        "actor_id": "A001",
        "ref_object": "O001",
        "role": "athlete",
        "entity": "female athlete in red and white uniform"
        }},
        {{
        "actor_id": "A002",
        "ref_object": "O002",
        "role": "audience",
        "entity": "crowd"
        }}
    ],
    "event": {{
        "event_id": "E_vHTfzg4dBsY_00_99",
        "name": "javelin throw preparation",
        "type": "sports_event",
        "time": {{"start": "00", "end": "99"}},
        "actors": ["A001","A002"],
        "objects": ["O001","O002"]
    }},
    "policy": {{
        "audience_filter": ["child_mode"],
        "priority": "high"
    }},
    "LOD": {{
        "abstract_topic": ["sports"],
        "scene_topic": "athlete preparing for javelin throw",
        "summary": "A female athlete prepares to throw the javelin as the crowd cheers.",
        "implications": "Highlights a competitive sports moment."
    }}
    }}

    ---

    Now process the following input:

    Input segment:
    Video {video_id}, time {start}-{end}, description: {text}

    Output JSON:
    """


    messages = [
        {"role": "system", "content": "You are an assistant that classifies captions into Visual, Audio, and Speech."},
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
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    response = response.strip()
    
    try:
        json_start = response.index("{")
        json_str = response[json_start:]
        parsed = json.loads(json_str)
    except Exception as e:
        print(f"[Warning] JSON parse error at segment {seg_idx}: {e}")
        parsed = {"raw_output": response}

    return parsed



# 전체 
def process_txt_file(input_file, output_file):
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(tqdm(lines, desc="Processing lines")):
        line = line.strip()
        if not line:
            continue

        # ✅ answer만 추출
        caption_text = extract_answer_from_line(line)
        if not caption_text:
            continue
        
        # time split
        segments = split_segments(caption_text)
        video_id = extract_videoid_from_line(line)
        
        for idx, (start, end, text) in enumerate(segments):
            result = extract_info_with_llm(video_id, idx, start, end, text)

            # modality split
            split_result = split_modality_caption_with_llm(text)
            # 결과 저장
            result_entry = {
                "video_id" : video_id,
                "event_id": idx,
                "original_answer": text,
                "llm_result" : result,
                "split_caption": split_result
            }
            results.append(result_entry)
        
            # append 저장
            with open(output_file, "a", encoding="utf-8") as out_f:
                out_f.write(str(result_entry) + "\n")

    return results


# === 실행 예시 ===
if __name__ == "__main__":
    input_file = "/home/kylee/LongVALE/logs/eval.txt"      # 처리할 TXT 파일
    output_file = "/home/kylee/LongVALE/logs/Structured_output_rev2.txt"   # 결과 저장 파일
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    process_txt_file(input_file, output_file)
