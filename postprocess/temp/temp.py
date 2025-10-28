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




def add_audio_to_split_caption(input_txt, output_txt):
    results = []

    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing lines"):
        try:
            entry = ast.literal_eval(line.strip())  # dict 복원
        except Exception as e:
            print("[Warning] parse error:", e)
            continue

        entry["split_caption"]["Audio"] = None

        results.append(entry)

    # 결과 저장
    with open(output_txt, "w", encoding="utf-8") as out_f:
        for entry in results:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[Done] Updated file saved to {output_txt}")



# === 실행 예시 ===
if __name__ == "__main__":
    input_txt = "/home/kylee/LongVALE/logs/Structured_output_with_speech.txt"
    output_txt = "/home/kylee/LongVALE/logs/Structured_output_with_speech_fin.txt"
    add_audio_to_split_caption(input_txt, output_txt)
