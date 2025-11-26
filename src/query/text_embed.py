from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:  # torch is optional
    torch = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

from query_utils import dedupe_best, measure_memory, select_top

EMBED_MODEL_CACHE: Dict[Tuple[str, str], "SentenceTransformer"] = {}


def rank_with_text_embed(
    query: str,
    segments: List[Dict[str, Any]],
    threshold: float,
    top_k: int,
    device: Optional[str],
    model_name: str,
    batch_size: int = 64,
) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, Dict[str, float]], Optional[int]]:
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

    def _encode(texts: List[str], tag: str):
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

    for start in range(0, len(candidate_texts), batch_size):
        chunk = candidate_texts[start : start + batch_size]
        text_vecs = _encode(chunk, f"batch_{start // batch_size:03d}")  # (B, d)
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
                    "score_type": "text_embed",
                }
            )

    ranked = select_top(dedupe_best(raw_matches), threshold, top_k)
    return (ranked, resolved_device, text_embed_profile, query_bytes)
