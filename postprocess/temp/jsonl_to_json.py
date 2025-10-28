import json
import os
from collections import defaultdict

def jsonl_to_grouped_json(jsonl_path, out_json_path=None):
    """
    JSONL(라인별 레코드) → [{video_id, results}] 리스트 JSON으로 변환.
    - 각 라인의 형태 예:
      {"video_id": "...", "result": {...}}
      {"video_id": "...", "results": [...]}
      {"video_id": "...", "postprocess": {...}}  # 호환 처리
    """
    if out_json_path is None:
        base, _ = os.path.splitext(jsonl_path)
        out_json_path = base + ".json"

    grouped = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            video_id = obj.get("video_id", "UNKNOWN")

            if "results" in obj and isinstance(obj["results"], list):
                grouped[video_id].extend(obj["results"])
            elif "result" in obj and obj["result"] is not None:
                grouped[video_id].append(obj["result"])
            elif "postprocess" in obj and obj["postprocess"] is not None:
                grouped[video_id].append(obj["postprocess"])

    output = [{"video_id": vid, "results": results} for vid, results in grouped.items()]

    with open(out_json_path, "w", encoding="utf-8") as out_f:
        json.dump(output, out_f, ensure_ascii=False, indent=2)

    print(f"저장 완료 → {out_json_path}")

if __name__ == "__main__":
    jsonl_to_grouped_json(
        "/home/kylee/kylee/LongVALE/logs/result_1015_merge.jsonl",
        "/home/kylee/kylee/LongVALE/logs/result_1015_merge.json"
    )