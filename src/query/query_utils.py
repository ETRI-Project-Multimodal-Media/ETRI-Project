import os
import re
import resource
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psutil


WORD_RE = re.compile(r"[0-9A-Za-z\uac00-\ud7a3']+")


def tokenize(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return WORD_RE.findall(text.lower())


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
