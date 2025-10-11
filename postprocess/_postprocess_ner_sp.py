import os
import json
import ast
from tqdm import tqdm
from transformers import pipeline
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

def parse_split_caption_to_dict(split_caption: str) -> dict:
    """
    'Visual: "..." Audio: "..." Speech: "..."' 형태의 문자열을 dict로 변환
    """
    result = {}
    matches = re.findall(r'(Visual|Audio|Speech):\s*"?([^"]+)"?', split_caption)
    for key, value in matches:
        result[key] = value.strip()
    return result

def convert_output_file(input_file, output_file):
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        try:
            entry = eval(line.strip())  # 원래 코드에서 dict를 str()로 저장했으므로 eval 필요
            if isinstance(entry, dict) and "split_caption" in entry:
                sc = entry["split_caption"]
                if isinstance(sc, str):
                    entry["split_caption"] = parse_split_caption_to_dict(sc)
            results.append(entry)
        except Exception as e:
            print("[Warning] parse error:", e)
            continue

    # 변환된 결과 다시 저장
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[Done] Converted file saved to {output_file}")



# 요약 모델 로드

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


def add_speech_to_split_caption(input_txt, speech_json_dir, output_txt):
    results = []

    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing lines"):
        try:
            entry = ast.literal_eval(line.strip())  # dict 복원
        except Exception as e:
            print("[Warning] parse error:", e)
            continue

        video_id = entry.get("video_id", None)
        if not video_id:
            continue

        # speech json 경로
        speech_json_path = os.path.join(speech_json_dir, f"{video_id}.json")
        if not os.path.isfile(speech_json_path):
            continue

        # transcription 읽기
        with open(speech_json_path, "r", encoding="utf-8") as sf:
            speech_data = json.load(sf)
        transcription = speech_data.get("transcription", "")

        # 요약
        summary = summarize_text(transcription)

        # split_caption 안에 Speech 넣기
        if "split_caption" not in entry or not isinstance(entry["split_caption"], dict):
            entry["split_caption"] = {}
        entry["split_caption"]["Speech"] = summary

        results.append(entry)

    # 결과 저장
    with open(output_txt, "w", encoding="utf-8") as out_f:
        for entry in results:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[Done] Updated file saved to {output_txt}")



# === 실행 예시 ===
if __name__ == "__main__":
    input_file = "/home/kylee/LongVALE/logs/Structured_output_rev2.txt"   # 기존 txt
    output_file = "/home/kylee/LongVALE/logs/Structured_output_rev_fixed.txt"  # 수정된 txt
    convert_output_file(input_file, output_file)
    
    input_txt = "/home/kylee/LongVALE/logs/Structured_output_rev_fixed.txt"
    speech_json_dir = "/home/kylee/LongVALE/data/speech_asr_1171"   # video_id.json 저장된 폴더
    output_txt = "/home/kylee/LongVALE/logs/Structured_output_with_speech.txt"
    add_speech_to_split_caption(input_txt, speech_json_dir, output_txt)
