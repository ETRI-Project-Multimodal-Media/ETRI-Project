import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 📂 JSON 파일이 들어 있는 폴더
input_folder = "/home/kylee/LongVALE/logs"
output_file = "/home/kylee/LongVALE/logs/postprocessing.jsonl"

# ✅ 사용할 LLM 모델 (로그인 필요 없는 공개 모델 추천)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def split_caption_with_llm(caption_text: str) -> str:
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
    Speech: "'I love you,' she sings."

    Input: "A man is playing the guitar while singing 'I love you.' The acoustic guitar sound resonates in the background."
    Output:
    Visual: "A man is playing the guitar."
    Audio: "The acoustic guitar sound resonates in the background."
    Speech: "'I love you,' he sings."

    ---

    Now split the following:

    Input: "{}"
    Output:
    """.format(caption_text)

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


from tqdm import tqdm

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

        # 👉 LLM 실행
        split_result = split_caption_with_llm(caption_text)

        # 결과 저장
        result_entry = {
            "line_id": idx,
            "original_answer": caption_text,
            "split_caption": split_result
        }
        results.append(result_entry)

        with open(output_file, "a", encoding="utf-8") as out_f:
            out_f.write(str(result_entry) + "\n")

    return results


if __name__ == "__main__":
    input_file = "/home/kylee/LongVALE/logs/eval.txt"      # 처리할 TXT 파일
    output_file = "/home/kylee/LongVALE/logs/LLAMA_postprocess.txt"   # 결과 저장 파일
    process_txt_file(input_file, output_file)
    print(f"✅ 결과가 {output_file}에 저장되었습니다.")