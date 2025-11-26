import os
import re
import json
import argparse
import copy
import textwrap
from collections import defaultdict

from tqdm import tqdm

from tree_merger import (
    merge_tree,
    merge_segments_by_kl,
    merge_segments_by_jsd,
    KL_SIMILARITY_THRESHOLD,
    JSD_SIMILARITY_THRESHOLD,
)
from info_extractor import extract_info_with_llm
from model_loader import llm
from modality_splitter import split_modality_caption_with_llm
from segment_utils import _collect_segments, _format_timestamp
from speech_extractor import extract_speech_from_caption_with_llm


# 시간 구간 추출
MIN_SEGMENT_REPEAT = 2
MIN_NGRAM_REPEAT = 3
NGRAM_OVERLAP_THRESHOLD = 0.8

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

    return merged

def chunk_text(text, max_chars=1500):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def summarize_text(text):
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

def _iter_video_payloads(raw_data):
    # Tree Output
    if isinstance(raw_data, dict):
        for video_id, payload in raw_data.items():
            yield str(video_id), payload
    elif isinstance(raw_data, list):
        # DVC output
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            video_id = item.get("video_id") or item.get("id")
            if not video_id:
                continue
            payload = item.get("tree") or item.get("data") or item.get("payload") or item
            yield str(video_id), payload
    else:
        raise ValueError("Unsupported input format: expected dict or list.")


# 전체 
def process_txt_file(input_file, output_dir, speech_json_dir, not_json_dir):
    video_results = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as f:
        # lines = f.readlines()
        raw_data = json.load(f)

    video_items = list(_iter_video_payloads(raw_data)) # (video_id, payload)

    # (one video) for timestamp, caption
    for video_id, video_tree in tqdm(video_items, desc="Processing videos"):
        if not isinstance(video_tree, dict):
            continue

        merge_tree(video_tree)

        segments = _collect_segments(video_tree)
        if not segments:
            continue

        duration = _format_timestamp(video_tree.get("end_time"))
        if not duration:
            duration = segments[-1]["end"]
        
        # merge time split 
        # segments = repeated_ngram_ratio(segments)
        # segments = merge_segments_by_kl(segments)
        # segments = merge_segments_by_jsd(segments)
        # unused
        # segments = merge_segments_by_text(segments) 
        
        # speech translation with summarized
        speech_translation = translate_speech(video_id, speech_json_dir)       

        for idx, segment in enumerate(segments):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            target_nodes = segment.get("nodes") or ([segment.get("node")] if segment.get("node") else [])
            if not target_nodes:
                continue
            # node 는 값이 아니고 참조변수
            result = extract_info_with_llm(video_id, idx, start, end, text, not_json_dir) # llama 1

            # LLM 결과에 관계없이 실제 start/end로 덮어쓰기
            result.setdefault('event', {}).setdefault('time', {})
            result['event']['time']['start'] = str(start)
            result['event']['time']['end'] = str(end)
            
            # LLM Summary를 그냥 caption으로 사용
            lod = result.setdefault('LOD', {})
            lod['summary'] = text

            # Visual, Audio modality split
            split_result = split_modality_caption_with_llm(text) # llama 1~2
            
            # speech summary and extract time speech
            speech_timesplit = extract_speech_from_caption_with_llm(text, speech_translation, start, end, duration) # llama 1
            # modality to dict
            split_result_dict = parse_split_caption_to_dict(split_result, speech_timesplit)
            
            result['LOD']['modalities'] = split_result_dict
            
            # node에 저장 
            for node in target_nodes:
                node["postprocess"] = {
                    "event_id": idx,
                    "original_answer": text,
                    "result": copy.deepcopy(result),
                }
        # save result
        output_path = os.path.join(output_dir, f"{video_id}.json")
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump({"video_id": video_id, "tree": video_tree}, out_f, ensure_ascii=False, indent=2)



# === 실행 예시 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run postprocess pipeline on one or more JSON inputs.")
    parser.add_argument(
        "--input",
        dest="input_files",
        nargs="+",
        default=[
            "../Example/Tree-Step3_part1.json",
            "../Example/Tree-Step3_part2.json",
            "../Example/Tree-Step3_part3.json",
            "../Example/Tree-Step3_part4.json",
        ],
        help="One or more JSON tree files to process.",
    )
    parser.add_argument(
        "--output-dir",
        default="../Example/postprocess",
        help="Directory to store processed outputs.",
    )
    parser.add_argument(
        "--speech-json-dir",
        default="../Example/speech_asr_1171",
        help="Directory containing speech ASR JSON files.",
    )
    parser.add_argument(
        "--not-json-dir",
        default="../Example/debug.txt",
        help="Path to log samples that fail JSON validation.",
    )
    parser.add_argument(
        "--log-dir",
        default="../Example/logs",
        help="Path to logs.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for input_file in args.input_files:
        process_txt_file(
            input_file,
            args.output_dir,
            speech_json_dir=args.speech_json_dir,
            not_json_dir=args.not_json_dir,
        )
