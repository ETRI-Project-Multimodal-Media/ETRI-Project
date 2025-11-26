from typing import Any, Dict, List

from query_utils import dedupe_best, select_top, tokenize


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
