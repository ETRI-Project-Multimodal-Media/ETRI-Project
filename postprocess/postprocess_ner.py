import os
import re
import json
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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

# 
def split_modality_caption_with_llm(caption_text: str) -> str:
    """
    LLM을 이용해 Visual/Audio 로 분리
    If Audio is None Then Use Captioning  """
    # Few-shot 프롬프트
    prompt = """
    You are a helpful assistant that splits a multimodal caption into two parts:  
    Visual, Audio.
    
    - Your task:
      Extract Visual and Audio part in the given caption.
      If the caption has no Audio, output "None".

    Definitions:
    - "visual": what is seen.
    - "audio": non-speech sounds (music, beep, engine, crowd noise, etc.). If no audio, use "".

    Examples:
    Input: "A woman is playing the piano while singing 'I love you.' Applause can be heard."
    Output:
    Visual: "A woman is playing the piano."
    Audio: "A woman is singing 'I love you.'. Applause can be heard."
    
    Input: "A man is playing the guitar while singing 'I love you.' The acoustic guitar sound resonates in the background."
    Output:
    Visual: "A man is playing the guitar."
    Audio: "A man is singing 'I love you.'. The acoustic guitar sound resonates in the background."


    ---

    Now split the following:
    
    Input: "{}"
    Output:
    
    """.format(caption_text)

    messages = [
        {"role": "system", "content": "You are an assistant that classifies captions into Visual, Audio."},
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
    input_file = "/home/kylee/LongVALE/logs/eval.txt"      # 처리할 TXT 파일
    output_file = "/home/kylee/LongVALE/logs/modality_split.txt"   # 결과 저장 파일
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    process_txt_file(input_file, output_file, speech_json_dir="/home/kylee/LongVALE/data/speech_asr_1171")
