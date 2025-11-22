#!/usr/bin/env python
"""
Query search utility for LongVALE postprocess outputs.

Supports two scoring modes:
1. Heuristic (token overlap on tags or scene topics/summaries)
2. CLIP text encoder similarity
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psutil

try:
    import torch
except ImportError:  # torch is only required for CLIP mode
    torch = None  # type: ignore

try:
    import open_clip
except ImportError:  # optional dependency
    open_clip = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


WORD_RE = re.compile(r"[0-9A-Za-z\uac00-\ud7a3']+")
EMBED_MODEL_CACHE: Dict[Tuple[str, str], "SentenceTransformer"] = {}


def tokenize(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return WORD_RE.findall(text.lower())


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


def select_top(matches: List[Dict[str, Any]], threshold: float, top_k: int) -> List[Dict[str, Any]]:
    matches.sort(key=lambda m: m["score"], reverse=True)
    above_threshold = [m for m in matches if m["score"] >= threshold]
    pool = above_threshold if above_threshold else matches
    return pool[:top_k]


def dedupe_best(matches: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[Any, Any, Any, str], Dict[str, Any]] = {}
    for item in matches:
        key = (item.get("video_id"), item.get("start_time"), item.get("end_time"), item.get("score_type"))
        if key not in best or item["score"] > best[key]["score"]:
            best[key] = item
    return list(best.values())


def rank_with_heuristic(query_tokens: List[str], segments: List[Dict[str, Any]], threshold: float, top_k: int) -> List[Dict[str, Any]]:
    if not query_tokens:
        return []

    single_word = len(query_tokens) == 1
    qset = set(query_tokens)
    raw_matches: List[Dict[str, Any]] = []

    for seg in segments:
        if single_word:
            tag_tokens: List[str] = []
            for tag in seg.get("tags", []):
                tag_tokens.extend(tokenize(tag))
            overlap = qset & set(tag_tokens)
            if not overlap:
                continue
            precision = len(overlap) / len(qset)
            score = precision
            match = {
                "video_id": seg["video_id"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "score": score,
                "matched_text": ", ".join(seg.get("tags", [])),
                "matched_field": "tags",
                "file_path": seg["file_path"],
                "score_type": "heuristic",
            }
            raw_matches.append(match)
            continue

        best_score = 0.0
        best_text = None
        best_field = None

        for field in ("scene_topic", "summary"):
            text = seg.get(field)
            tokens = tokenize(text)
            if not tokens:
                continue
            overlap = qset & set(tokens)
            if not overlap:
                continue
            precision = len(overlap) / len(qset)
            recall = len(overlap) / len(tokens)
            score = 0.5 * (precision + recall)
            if score > best_score:
                best_score = score
                best_text = text
                best_field = field

        if best_score == 0.0 or not best_text:
            continue

        raw_matches.append(
            {
                "video_id": seg["video_id"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "score": best_score,
                "matched_text": best_text,
                "matched_field": best_field,
                "file_path": seg["file_path"],
                "score_type": "heuristic",
            }
        )

    return select_top(dedupe_best(raw_matches), threshold, top_k)


def rank_with_clip(
    query: str,
    segments: List[Dict[str, Any]],
    threshold: float,
    top_k: int,
    device: Optional[str],
    model_name: str,
    batch_size: int = 64,
) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, Dict[str, float]], Optional[int]]:
    # 이제는 CLIP이 아니라 sentence-transformers 기반 텍스트 임베딩을 사용
    if SentenceTransformer is None:
        raise RuntimeError(
            "Text-embedding mode requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        )

    resolved_device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")

    text_embed_profile: Dict[str, Dict[str, float]] = {}

    def log_stage(label: str, t0: float) -> None:
        current_mem, peak_mem = measure_memory()
        text_embed_profile[label] = {
            "elapsed": time.perf_counter() - t0,
            "current_memory_mib": current_mem,
            "peak_memory_mib": peak_mem,
        }

    # 모델 로드 (캐시 활용)
    stage_start = time.perf_counter()
    cache_key = (model_name, resolved_device)
    text_model = EMBED_MODEL_CACHE.get(cache_key)
    cache_label = "model_ready"
    if text_model is None:
        text_model = SentenceTransformer(model_name, device=resolved_device)
        EMBED_MODEL_CACHE[cache_key] = text_model
    else:
        cache_label = "model_cached"
    log_stage(cache_label, stage_start)

    # 임베딩 함수: normalize_embeddings=True 로 코사인 유사도 직접 사용 가능
    def _encode(texts: List[str], tag: str) -> "np.ndarray":
        encode_t0 = time.perf_counter()
        embeddings = text_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        log_stage(tag, encode_t0)
        return embeddings

    # 쿼리 임베딩
    query_vec = _encode([query], "query_embedding")[0]  # shape: (d,)
    query_bytes: Optional[int] = int(query_vec.nbytes) if hasattr(query_vec, "nbytes") else None
    if query_bytes is not None:
        text_embed_profile.setdefault("query_embedding", {})["embedding_bytes"] = query_bytes

    # 후보 텍스트 구성
    candidate_payloads: List[Tuple[Dict[str, Any], str, str]] = []
    candidate_texts: List[str] = []
    for seg in segments:
        for field in ("scene_topic", "summary"):
            text = seg.get(field)
            if not text:
                continue
            candidate_payloads.append((seg, field, text))
            candidate_texts.append(text)

    if not candidate_texts:
        return ([], resolved_device, text_embed_profile, query_bytes)

    raw_matches: List[Dict[str, Any]] = []

    # 배치 단위로 임베딩 + 쿼리와 코사인 유사도
    for start in range(0, len(candidate_texts), batch_size):
        chunk = candidate_texts[start : start + batch_size]
        text_vecs = _encode(chunk, f"batch_{start // batch_size:03d}")  # (B, d)
        # query_vec 은 이미 정규화, text_vecs 도 정규화 → dot = cosine
        scores = (text_vecs @ query_vec).tolist()

        for idx, score in enumerate(scores):
            seg, field, text = candidate_payloads[start + idx]
            raw_matches.append(
                {
                    "video_id": seg["video_id"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "score": float(score),
                    "matched_text": text,
                    "matched_field": field,
                    "file_path": seg["file_path"],
                    "score_type": "text_embed",  # 구분을 위해 clip 대신 text_embed 로 써도 좋음
                }
            )

    ranked = select_top(dedupe_best(raw_matches), threshold, top_k)
    return (ranked, resolved_device, text_embed_profile, query_bytes)


def measure_memory() -> Tuple[float, float]:
    process = psutil.Process(os.getpid())
    current = process.memory_info().rss / (1024 * 1024)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is reported in KB on Linux
    if os.name == "posix":
        peak = peak / 1024
    else:
        peak = peak / (1024 * 1024)
    return (current, peak)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search LongVALE postprocess JSON with heuristic or CLIP scores.")
    parser.add_argument("--input", type=Path, default=Path("/home/kylee/LongVALE/temp_data/sO3wd7X-l7U.json"))
    parser.add_argument("--query", required=True, help="Query string to search for.")
    parser.add_argument("--mode", choices=("heuristic", "text_embed", "both"), default="heuristic")
    parser.add_argument("--top-k", type=int, default=20, dest="top_k")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum score required to keep a match.")
    parser.add_argument("--output", type=Path, default=Path("/home/kylee/LongVALE/temp_data/query/clip_results.json"))
    parser.add_argument("--device", type=str, default=None, help="Device override for CLIP (e.g., cuda:0).")
    parser.add_argument(
        "--clip-model",
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
        clip_matches, resolved_device, text_embed_profile, query_embedding_bytes = rank_with_clip(
            args.query,
            segments,
            args.threshold,
            args.top_k,
            args.device,
            args.clip_model,
        )
        matches.extend(clip_matches)
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
