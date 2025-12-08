#!/usr/bin/env python
"""
Query search utility for LongVALE postprocess outputs.

Supports two scoring modes:
1. Heuristic (token overlap on tags or scene topics/summaries)
2. Sentence Embedding similarity
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from heuristic import rank_with_heuristic
from query_utils import measure_memory, select_top, tokenize
from text_embed import rank_with_text_embed


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flatten_segments(tree: Dict[str, Any], video_id: str, file_path: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []

    def _dfs(node: Dict[str, Any]) -> None:
        post = node.get("postprocess")
        if isinstance(post, dict):
            result = post.get("result") or {}
            lod = result.get("LOD") or {}
            segment = {
                "video_id": result.get("video_id", video_id),
                "event_id": result.get("event_id"),
                "start_time": _to_float(node.get("start_time")),
                "end_time": _to_float(node.get("end_time")),
                "tags": result.get("tags") or [],
                "scene_topic": lod.get("scene_topic") or node.get("summary"),
                "summary": lod.get("summary") or node.get("summary") or node.get("caption"),
                "caption": node.get("caption"),
                "file_path": file_path,
            }
            segments.append(segment)

        for child in node.get("children") or []:
            if isinstance(child, dict):
                _dfs(child)

    _dfs(tree)
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search LongVALE postprocess JSON with heuristic or Text Embed scores.")
    parser.add_argument("--input", type=Path, default=Path("../Example/postprocess/sO3wd7X-l7U.json"))
    parser.add_argument("--query", required=True, help="Query string to search for.")
    parser.add_argument("--mode", choices=("heuristic", "text_embed", "both"), default="heuristic")
    parser.add_argument("--top-k", type=int, default=20, dest="top_k")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum score required to keep a match.")
    parser.add_argument("--output", type=Path, default=Path("../Example/postprocess/query/clip_results.json"))
    parser.add_argument("--device", type=str, default=None, help="Device override for Senetence Embedding Model (e.g., cuda:0).")
    parser.add_argument(
        "--text-model",
        type=str,
        default="BAAI/bge-m3",
        help="sentence-transformers model name for text embedding mode (was CLIP).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_id = data.get("video_id") or Path(args.input).stem
    tree = data.get("tree")
    if not tree:
        raise ValueError("Input JSON must contain a 'tree' object.")

    segments = flatten_segments(tree, video_id, str(args.input))
    query_tokens = tokenize(args.query)

    matches: List[Dict[str, Any]] = []
    device_used: Optional[str] = None

    if args.mode in {"heuristic", "both"}:
        matches.extend(rank_with_heuristic(query_tokens, segments, args.threshold, args.top_k))

    query_embedding_bytes: Optional[int] = None

    if args.mode in {"text_embed", "both"}:
        embed_matches, resolved_device, text_embed_profile, query_embedding_bytes = rank_with_text_embed(
            args.query,
            segments,
            args.threshold,
            args.top_k,
            args.device,
            args.text_model,
        )
        matches.extend(embed_matches)
        device_used = resolved_device
    
    elapsed = time.time() - start_time
    current_mem, peak_mem = measure_memory()

    output_matches = select_top(matches, args.threshold, args.top_k) if args.mode == "both" else matches
    
    if args.mode in {"text_embed", "both"}:
        result = {
            "query": args.query,
            "input_file": str(args.input),
            "mode": args.mode,
            "top_k": args.top_k,
            "similarity_threshold": args.threshold,
            "device": device_used,
            "elapsed_seconds": elapsed,
            "current_memory_mib": current_mem,
            "peak_memory_mib": peak_mem,
            "match_count": len(output_matches),
            "matches": output_matches,
            "text_embed_profile": text_embed_profile,
            "query_embedding_bytes": query_embedding_bytes,
        }
    else:
        result = {
            "query": args.query,
            "input_file": str(args.input),
            "mode": args.mode,
            "top_k": args.top_k,
            "similarity_threshold": args.threshold,
            "device": device_used,
            "elapsed_seconds": elapsed,
            "current_memory_mib": current_mem,
            "peak_memory_mib": peak_mem,
            "match_count": len(output_matches),
            "matches": output_matches,
            "text_embed_profile": "",
            "query_embedding_bytes": None,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, ensure_ascii=False, indent=2)
    # visualize_query_pipeline(args.mode)




if __name__ == "__main__":
    main()
