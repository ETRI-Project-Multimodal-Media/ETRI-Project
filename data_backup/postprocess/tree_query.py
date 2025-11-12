import argparse
import json
import math
import re
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_DATA_DIR = Path(__file__).resolve().parent


def normalize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def count_relative_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    mag_a = math.sqrt(sum(v * v for v in ca.values()))
    mag_b = math.sqrt(sum(v * v for v in cb.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def find_video_segments(
    payload: Dict[str, Any],
    query: str,
    similarity_threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    video_id = payload.get("video_id")
    root = payload.get("tree", {})
    query = (query or "").strip()
    if not query:
        return []

    is_single_token = len(query.split()) == 1
    query_lower = query.lower()
    query_tokens = normalize(query)
    matches: List[Dict[str, Any]] = []

    def traverse(node: Dict[str, Any]) -> None:
        post = node.get("postprocess", {})
        result = post.get("result", {})
        if result:
            entry = {
                "video_id": result.get("video_id", video_id),
                "start_time": node.get("start_time"),
                "end_time": node.get("end_time"),
                "node": node,
                "score": 0.0,
            }
            if is_single_token:
                tags = result.get("tags", [])
                if any(query_lower in (tag or "").lower() for tag in tags):
                    entry["score"] = 1.0
                    matches.append(entry)
            else:
                lod = result.get("LOD", {})
                candidates = [lod.get("scene_topic", ""), lod.get("summary", "")]
                best = max(
                    (count_relative_similarity(query_tokens, normalize(text)) for text in candidates if text),
                    default=0.0,
                )
                if similarity_threshold is None or best >= similarity_threshold:
                    entry["score"] = best
                    matches.append(entry)

        for child in node.get("children", []):
            traverse(child)

    traverse(root)
    matches.sort(key=lambda item: item["score"], reverse=True)
    if top_k is not None:
        matches = matches[:top_k]
    return matches


def iter_payloads(data_dir: Optional[str] = None) -> Iterable[Tuple[Dict[str, Any], Path]]:
    base_path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    for json_path in sorted(base_path.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            yield payload, json_path


def find_video_segments_in_dir(
    query: str,
    data_dir: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for payload, path in iter_payloads(data_dir):
        segments = find_video_segments(
            payload,
            query,
            similarity_threshold=similarity_threshold,
            top_k=None,
        )
        for segment in segments:
            entry = segment.copy()
            entry["file_path"] = str(path)
            matches.append(entry)

    matches.sort(key=lambda item: item["score"], reverse=True)
    if top_k is not None:
        matches = matches[:top_k]
    return matches


def _prepare_matches_for_save(matches: List[Dict[str, Any]], omit_node: bool) -> List[Dict[str, Any]]:
    if not omit_node:
        return matches
    prepared: List[Dict[str, Any]] = []
    for match in matches:
        prepared.append({key: value for key, value in match.items() if key != "node"})
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search postprocessed tree JSON files using heuristic text similarity."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="Directory containing tree JSON files (default: sibling data/postprocess directory).",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Text query to search for. If omitted, the program prompts for one.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=20,
        help="Maximum number of matches to display (default: 20).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Minimum similarity score to keep a match (omit to keep all results).",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Optional path to write matches and profiling stats as JSON.",
    )
    parser.add_argument(
        "--omit-node",
        action="store_true",
        help="Drop the 'node' field from saved matches (affects --save output only).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or None
    query = (args.query or "").strip()
    if not query:
        try:
            query = input("Enter search query: ").strip()
        except EOFError:
            query = ""

    if not query:
        print("Query is required to perform search.", file=sys.stderr)
        sys.exit(1)

    tracemalloc.start()
    start_time = time.perf_counter()
    matches = find_video_segments_in_dir(
        query=query,
        data_dir=data_dir,
        similarity_threshold=args.threshold,
        top_k=args.top_k,
    )
    elapsed_seconds = time.perf_counter() - start_time
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    current_mib = current_bytes / (1024 ** 2)
    peak_mib = peak_bytes / (1024 ** 2)

    if not matches:
        print("No matches found.")
        print(
            f"Elapsed: {elapsed_seconds:.3f}s | "
            f"Current memory: {current_mib:.2f} MiB | "
            f"Peak memory: {peak_mib:.2f} MiB"
        )
        if args.save:
            matches_for_save: List[Dict[str, Any]] = []
            summary = {
                "query": query,
                "data_dir": data_dir,
                "top_k": args.top_k,
                "similarity_threshold": args.threshold,
                "elapsed_seconds": elapsed_seconds,
                "current_memory_mib": current_mib,
                "peak_memory_mib": peak_mib,
                "match_count": 0,
                "node_included": not args.omit_node,
                "matches": matches_for_save,
            }
            with Path(args.save).open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            print(f"Saved summary to {args.save}")
        return

    for idx, match in enumerate(matches, start=1):
        video_id = match.get("video_id", "")
        start = match.get("start_time")
        end = match.get("end_time")
        score = match.get("score", 0.0)
        file_path = match.get("file_path", "")
        print(
            f"{idx:02d}. score={score:.3f} video={video_id} "
            f"time={start}-{end} file={file_path}"
        )

    print(
        f"\nElapsed: {elapsed_seconds:.3f}s | "
        f"Current memory: {current_mib:.2f} MiB | "
        f"Peak memory: {peak_mib:.2f} MiB"
    )

    if args.save:
        matches_for_save = _prepare_matches_for_save(matches, args.omit_node)
        summary = {
            "query": query,
            "data_dir": data_dir,
            "top_k": args.top_k,
            "similarity_threshold": args.threshold,
            "elapsed_seconds": elapsed_seconds,
            "current_memory_mib": current_mib,
            "peak_memory_mib": peak_mib,
            "match_count": len(matches),
            "node_included": not args.omit_node,
            "matches": matches_for_save,
        }
        with Path(args.save).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"Saved summary to {args.save}")


if __name__ == "__main__":
    main()
