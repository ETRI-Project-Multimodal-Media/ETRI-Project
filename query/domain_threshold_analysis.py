#!/usr/bin/env python
"""Domain-aware scoring for LongVALE level-1 captions."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from search_queries import flatten_segments, rank_with_clip

DOMAINS: Dict[str, Dict[str, List[str]]] = {
    "auto": {
        "keywords": [
            "car",
            "truck",
            "suv",
            "vehicle",
            "drive",
            "ford",
            "engine",
            "dealership",
            "sedan",
        ],
        "queries": [
            "luxury suv interior",
            "car dealership promotion",
        ],
    },
    "sports": {
        "keywords": [
            "game",
            "match",
            "team",
            "player",
            "goal",
            "stadium",
            "coach",
            "javelin",
            "athlete",
        ],
        "queries": [
        "sports"
        ],
    },
    "food": {
        "keywords": [
            "cook",
            "chef",
            "recipe",
            "kitchen",
            "dish",
            "restaurant",
            "flavor",
            "bake",
            "ingredient",
        ],
        "queries": [
                "food"
        ],
    },
    "news": {
        "keywords": [
            "news",
            "report",
            "breaking",
            "headline",
            "interview",
            "press",
            "conference",
            "statement",
            "anchor",
        ],
        "queries": [
            "news"
        ],
    },
    "kids": {
        "keywords": [
            "kid",
            "child",
            "children",
            "school",
            "teacher",
            "lesson",
            "cartoon",
            "toy",
            "classroom",
        ],
        "queries": [
            "kids"
        ],
    },
}

WORD_RE = re.compile(r"[0-9A-Za-z\uac00-\ud7a3']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "the",
    "their",
    "to",
    "with",
}


@dataclass
class VideoKeywords:
    video_id: str
    keywords: str


def tokenize_caption(text: str) -> List[str]:
    if not text:
        return []
    return WORD_RE.findall(text.lower())
def summarize_to_keywords(captions: Iterable[str], max_keywords: int) -> str:
    token_counts: Counter[str] = Counter()
    for caption in captions:
        for token in tokenize_caption(caption):
            if token in STOPWORDS:
                continue
            token_counts[token] += 1
    if not token_counts:
        return ""
    ordered_tokens = [token for token, _ in token_counts.most_common(max_keywords)]
    return " ".join(ordered_tokens)


def encode_texts(model: SentenceTransformer, texts: Sequence[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def load_level1_keywords(tree_path: Path, max_keywords: int) -> List[VideoKeywords]:
    with tree_path.open("r", encoding="utf-8") as f:
        tree_data = json.load(f)

    entries: List[VideoKeywords] = []
    for video_id, root in tree_data.items():
        children = root.get("children") or []
        level1_captions = []
        for child in children:
            if not isinstance(child, dict):
                continue
            if child.get("level") != 1:
                continue
            caption = child.get("caption")
            if caption:
                level1_captions.append(caption)
        keywords = summarize_to_keywords(level1_captions, max_keywords)
        if keywords:
            entries.append(VideoKeywords(video_id=video_id, keywords=keywords))
    return entries


def assign_domains(
    videos: Sequence[VideoKeywords],
    video_embeddings: np.ndarray,
    domain_query_vectors: Dict[str, np.ndarray],
) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    for idx, video in enumerate(videos):
        vector = video_embeddings[idx]
        best_domain = None
        best_score = -1.0
        for domain, query_vecs in domain_query_vectors.items():
            if query_vecs.size == 0:
                continue
            scores = vector @ query_vecs.T
            score = float(scores.max())
            if score > best_score:
                best_score = score
                best_domain = domain
        if best_domain:
            assignments[video.video_id] = best_domain
    return assignments


def load_segments_for_video(video_id: str, video_dir: Path) -> List[Dict[str, Any]]:
    path = video_dir / f"{video_id}.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tree = data.get("tree")
    if tree is None:
        return []
    return flatten_segments(tree, video_id, str(path))


def compute_video_top_scores(
    segments: Sequence[Dict[str, Any]],
    queries: Sequence[str],
    per_query_top_k: int,
    video_top_k: int,
    threshold: float,
    device: Optional[str],
    model_name: str,
    batch_size: int,
) -> List[float]:
    if not segments or not queries:
        return []
    score_map: Dict[tuple, float] = {}
    for query in queries:
        matches, device, _, _ = rank_with_clip(
            query,
            list(segments),
            threshold,
            per_query_top_k,
            device,
            model_name,
            batch_size=batch_size,
        )
        for match in matches:
            key = (
                match.get("video_id"),
                match.get("start_time"),
                match.get("end_time"),
                match.get("matched_field"),
            )
            score = float(match.get("score", 0.0))
            prev = score_map.get(key)
            if prev is None or score > prev:
                score_map[key] = score
    if not score_map:
        return []
    scores = sorted(score_map.values(), reverse=True)
    return scores[:video_top_k]


def aggregate_domain_scores(
    assignments: Dict[str, str],
    video_dir: Path,
    per_domain_queries: Dict[str, List[str]],
    per_query_top_k: int,
    video_top_k: int,
    threshold: float,
    device: Optional[str],
    model_name: str,
    batch_size: int,
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {
        domain: {
            "assigned_videos": [],
            "video_top_scores": {},
            "missing_videos": [],
        }
        for domain in DOMAINS
    }

    for video_id, domain in assignments.items():
        info = summary.setdefault(
            domain,
            {
                "assigned_videos": [],
                "video_top_scores": {},
                "missing_videos": [],
            },
        )
        segments = load_segments_for_video(video_id, video_dir)
        if not segments:
            info["missing_videos"].append(video_id)
            continue
        top_scores = compute_video_top_scores(
            segments,
            per_domain_queries.get(domain, []),
            per_query_top_k,
            video_top_k,
            threshold,
            device,
            model_name,
            batch_size,
        )
        info["assigned_videos"].append(video_id)
        info["video_top_scores"][video_id] = top_scores

    for domain, info in summary.items():
        all_scores: List[float] = []
        for scores in info["video_top_scores"].values():
            all_scores.extend(scores)
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        info["video_count"] = len(info["assigned_videos"])
        info["tracked_scores"] = len(all_scores)
        info["domain_top3_avg_score"] = avg_score
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain similarity stats for level-1 captions.")
    parser.add_argument(
        "--tree-file",
        type=Path,
        default=Path("/home/kylee/LongVALE/temp_data/Tree-Step3.json"),
        help="Path to Tree-Step3.json",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("/home/kylee/LongVALE/temp_data/after_postprocess"),
        help="Directory containing per-video *_step3.json files",
    )
    parser.add_argument("--output", type=Path, default=Path("domain_topk_stats.json"))
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-keywords", type=int, default=15, help="Keywords to keep per video")
    parser.add_argument("--search-top-k", type=int, default=3, help="Matches to pull per query when scanning segments")
    parser.add_argument("--video-top-k", type=int, default=3, help="Top segment scores per video to keep for averaging")
    parser.add_argument("--search-threshold", type=float, default=0.0, help="Score threshold for rank_with_clip")
    parser.add_argument(
        "--search-model",
        type=str,
        default="BAAI/bge-m3",
        help="SentenceTransformer model name for clip-style search",
    )
    parser.add_argument("--search-device", type=str, default=None, help="Device override for rank_with_clip")
    parser.add_argument("--search-batch-size", type=int, default=64, help="Batch size for clip search encoding")
    args = parser.parse_args()
    search_model_name = args.search_model or args.model

    video_entries = load_level1_keywords(args.tree_file, args.max_keywords)
    if not video_entries:
        raise SystemExit("No usable level-1 captions found in tree file")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model, device=device)

    video_embeddings = encode_texts(model, [entry.keywords for entry in video_entries], args.batch_size)

    domain_query_vectors: Dict[str, np.ndarray] = {}
    domain_queries: Dict[str, List[str]] = {}
    for domain, info in DOMAINS.items():
        queries = info.get("queries", [])
        domain_queries[domain] = queries
        domain_query_vectors[domain] = encode_texts(model, queries, args.batch_size)

    assignments = assign_domains(video_entries, video_embeddings, domain_query_vectors)

    summary = aggregate_domain_scores(
        assignments,
        args.video_dir,
        domain_queries,
        args.search_top_k,
        args.video_top_k,
        args.search_threshold,
        args.search_device,
        search_model_name,
        args.search_batch_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved stats to {args.output}")


if __name__ == "__main__":
    main()
