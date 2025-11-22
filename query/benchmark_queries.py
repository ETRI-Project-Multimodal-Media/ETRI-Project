#!/usr/bin/env python
import csv
import json
import subprocess
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path("/home/kylee/LongVALE")
SCRIPT = REPO_ROOT / "temp_data" / "query" / "search_queries.py"

# 실험할 입력 JSON, 쿼리, 모드 정의
INPUTS = [
    REPO_ROOT / "temp_data" / "sO3wd7X-l7U.json",
    REPO_ROOT / "temp_data" / "w2nmzwknVco.json",
]

QUERIES = [
    "Throws javelin in the air",
    "javelin",
    "Comfortable seat",
    "Comfortable",
    "Monster"
]

MODES = ["heuristic", "text_embed"]  # 필요하면 "both"도 추가

def run_one(input_path: Path, query: str, mode: str) -> dict:
    out_dir = REPO_ROOT / "temp_data" / "query" / "bench_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{input_path.stem}_{mode}_{query.replace(' ', '_')}.json"

    cmd = [
        "python",
        str(SCRIPT),
        "--input", str(input_path),
        "--query", query,
        "--mode", mode,
        "--output", str(out_json),
    ]
    subprocess.run(cmd, check=True)

    with out_json.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPO_ROOT / "temp_data" / "query" / f"benchmark_{ts}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "input_file",
            "query",
            "mode",
            "elapsed_seconds",
            "query_embedding_bytes",
            "peak_memory_mib",
            "match_count",
            "text_embed_profile",
            "top1_score",
            "top1_start_time",
            "top1_end_time",
            "top1_matched_field",
            "top1_text_short",
        ])
        for inp in INPUTS:
            for q in QUERIES:
                for mode in MODES:
                    result = run_one(inp, q, mode)

                    raw_peak = result.get("peak_memory_mib")
                    text_embed_profile = result.get("text_embed_profile") or {}
                    query_embedding_bytes = result.get("query_embedding_bytes")

                    # 간단한 검색 결과 요약 (top-1)
                    matches = result.get("matches") or []
                    if matches:
                        top = matches[0]
                        top_score = top.get("score")
                        top_start = top.get("start_time")
                        top_end = top.get("end_time")
                        top_field = top.get("matched_field")
                        summary_text = (top.get("matched_text") or "").replace("\n", " ").strip()
                        # if len(summary_text) > 80:
                        #     summary_text = summary_text[:77] + "..."
                    else:
                        top_score = top_start = top_end = top_field = summary_text = ""

                    writer.writerow([
                        result.get("input_file"),
                        result.get("query"),
                        result.get("mode"),
                        result.get("elapsed_seconds"),
                        query_embedding_bytes,
                        raw_peak,
                        result.get("match_count"),
                        text_embed_profile,
                        top_score,
                        top_start,
                        top_end,
                        top_field,
                        summary_text,
                    ])
                    print("done:", inp.name, q, mode)

    print("CSV saved at:", csv_path)

if __name__ == "__main__":
    main()
