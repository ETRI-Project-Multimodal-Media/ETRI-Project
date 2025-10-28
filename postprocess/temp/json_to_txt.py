import os
import json
def normalize_video_results(video_results):
    """Return a list of {'video_id': ..., 'results': ...} blocks."""
    if isinstance(video_results, dict):
        return [
            {"video_id": video_id, "results": events}
            for video_id, events in video_results.items()
        ]

    if isinstance(video_results, list):
        if all(isinstance(block, dict) and "video_id" in block for block in video_results):
            # JSONL 형태(각 라인에 {"video_id":..., "result": {...}})를
            # 한 비디오당 results 리스트로 그룹화
            if all("result" in block or "postprocess" in block or "results" in block for block in video_results):
                grouped = {}
                for block in video_results:
                    video_id = block.get("video_id", "UNKNOWN")
                    if "results" in block and isinstance(block["results"], list):
                        grouped.setdefault(video_id, []).extend(block["results"])
                    else:
                        res = block.get("result")
                        if res is None:
                            res = block.get("postprocess")
                        if res is not None:
                            grouped.setdefault(video_id, []).append(res)
                return [{"video_id": vid, "results": evts} for vid, evts in grouped.items()]
            return video_results

        if all(
            isinstance(block, (list, tuple)) and len(block) == 2
            for block in video_results
        ):
            return [
                {"video_id": video_id, "results": events}
                for video_id, events in video_results
            ]

        normalized = []
        for idx, block in enumerate(video_results):
            if isinstance(block, dict):
                video_id = block.get("video_id", f"video_{idx:04d}")
                results = block.get("results", block.get("events", []))
                if not isinstance(results, list):
                    results = [results]
            else:
                video_id = f"video_{idx:04d}"
                results = block if isinstance(block, list) else [block]

            normalized.append({"video_id": video_id, "results": results})

        return normalized

    raise TypeError(
        f"Unsupported data type for video_results: {type(video_results).__name__}"
    )


def json_to_txt(json_path, txt_path=None, rewrite_json=False):
    # JSONL 지원: .jsonl이면 라인별 JSON을 읽어 리스트로 구성
    if json_path.lower().endswith(".jsonl"):
        raw_list = []
        with open(json_path, "r", encoding="utf-8") as jf:
            for line in jf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                raw_list.append(obj)
        raw_data = raw_list
    else:
        with open(json_path, "r", encoding="utf-8") as jf:
            raw_data = json.load(jf)

    output_data = normalize_video_results(raw_data)

    if txt_path is None:
        txt_path = os.path.splitext(json_path)[0] + ".txt"

    with open(txt_path, "w", encoding="utf-8") as txt_f:
        for block in output_data:
            video_id = block.get("video_id", "UNKNOWN")
            txt_f.write(f"[video_id] {video_id}\n")

            for event in block.get("results", []):
                if isinstance(event, dict):
                    event_id = event.get("event_id", "N/A")
                    original_answer = event.get("original_answer", "")
                    event_result = event.get("result")
                    if event_result is None:
                        event_result = event.get("postprocess", "")
                else:
                    event_id = "N/A"
                    original_answer = str(event)
                    event_result = event

                txt_f.write(f"  event_id: {event_id}\n")
                txt_f.write(f"  original_answer: {original_answer}\n")
                txt_f.write(
                    f"  result: {json.dumps(event_result, ensure_ascii=False)}\n"
                )
                txt_f.write("\n")

            txt_f.write("=" * 40 + "\n\n")

    if rewrite_json:
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(output_data, jf, ensure_ascii=False, indent=2)

    print(f"TXT 저장 완료 → {txt_path}")


if __name__ == "__main__":
    before_json_path = "/home/kylee/kylee/LongVALE/logs/result_1015_before_time.jsonl"
    after_json_path = "/home/kylee/kylee/LongVALE/logs/result_1015_merge.jsonl"

    json_to_txt(before_json_path)
    json_to_txt(after_json_path)
