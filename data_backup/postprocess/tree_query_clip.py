import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import clip
import torch

DEFAULT_DATA_DIR = Path(__file__).resolve().parent
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_BATCH_SIZE = 64

_CLIP_MODEL: Optional[torch.nn.Module] = None
_CLIP_DEVICE: Optional[str] = None


def _load_clip_model(device: Optional[str] = None) -> Tuple[torch.nn.Module, str]:
    global _CLIP_MODEL, _CLIP_DEVICE
    if _CLIP_MODEL is None:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = clip.load(CLIP_MODEL_NAME, device=dev)  # type: ignore[misc]
        model.eval()
        _CLIP_MODEL = model
        _CLIP_DEVICE = dev
    assert _CLIP_MODEL is not None
    assert _CLIP_DEVICE is not None
    return _CLIP_MODEL, _CLIP_DEVICE


def _encode_texts(texts: List[str], device: Optional[str] = None) -> torch.Tensor:
    model, dev = _load_clip_model(device)
    features: List[torch.Tensor] = []
    for start in range(0, len(texts), CLIP_BATCH_SIZE):
        batch = texts[start:start + CLIP_BATCH_SIZE]
        tokens = clip.tokenize(batch, truncate=True).to(dev)  # type: ignore[misc]
        with torch.no_grad():
            embeddings = model.encode_text(tokens).float()
        norms = embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings / torch.clamp(norms, min=1e-9)
        features.append(embeddings.cpu())
    return torch.cat(features, dim=0) if features else torch.empty(0)


def _collect_node_texts(node: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    caption = node.get("caption")
    if isinstance(caption, str) and caption.strip():
        texts.append(caption.strip())

    post = node.get("postprocess", {})
    result = post.get("result", {})
    if isinstance(result, dict):
        tags = result.get("tags")
        if tags:
            tag_line = " ".join(str(tag) for tag in tags if tag)
            if tag_line:
                texts.append(tag_line)

        lod = result.get("LOD", {})
        if isinstance(lod, dict):
            for key in ("scene_topic", "summary"):
                value = lod.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())

            modalities = lod.get("modalities", {})
            if isinstance(modalities, dict):
                for value in modalities.values():
                    if isinstance(value, str) and value.strip():
                        texts.append(value.strip())

    # Preserve order while removing duplicates.
    seen = set()
    unique_texts: List[str] = []
    for text in texts:
        normalized = text.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_texts.append(normalized)
    return unique_texts


def _traverse_nodes(root: Dict[str, Any]) -> List[Tuple[Dict[str, Any], List[str]]]:
    collected: List[Tuple[Dict[str, Any], List[str]]] = []

    def _walk(node: Dict[str, Any]) -> None:
        node_texts = _collect_node_texts(node)
        if node_texts:
            collected.append((node, node_texts))
        for child in node.get("children", []):
            if isinstance(child, dict):
                _walk(child)

    if isinstance(root, dict):
        _walk(root)
    return collected


def find_video_segments(
    payload: Dict[str, Any],
    query: str,
    similarity_threshold: float = 0.2,
    top_k: Optional[int] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    video_id = payload.get("video_id")
    root = payload.get("tree", {})
    query = (query or "").strip()
    if not query:
        return []

    query_embeddings = _encode_texts([query], device=device)
    if query_embeddings.numel() == 0:
        return []
    query_vector = query_embeddings[0]

    nodes = _traverse_nodes(root)
    if not nodes:
        return []

    all_texts = [text for _, texts in nodes for text in texts]
    unique_texts = list(dict.fromkeys(text for text in all_texts if text))
    if not unique_texts:
        return []

    text_embeddings = _encode_texts(unique_texts, device=device)
    if text_embeddings.shape[0] != len(unique_texts):
        return []

    text_to_embedding = {
        text: text_embeddings[idx]
        for idx, text in enumerate(unique_texts)
    }

    matches: List[Dict[str, Any]] = []
    for node, texts in nodes:
        best_score = -1.0
        matched_text = None
        for text in texts:
            embedding = text_to_embedding.get(text)
            if embedding is None:
                continue
            score = float(torch.dot(query_vector, embedding))
            if score > best_score:
                best_score = score
                matched_text = text

        if matched_text is None or best_score < similarity_threshold:
            continue

        post = node.get("postprocess", {})
        result = post.get("result", {})
        entry = {
            "video_id": result.get("video_id", video_id),
            "start_time": node.get("start_time"),
            "end_time": node.get("end_time"),
            "node": node,
            "score": best_score,
            "matched_text": matched_text,
        }
        matches.append(entry)

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
    similarity_threshold: float = 0.2,
    top_k: int = 20,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for payload, path in iter_payloads(data_dir):
        segments = find_video_segments(
            payload,
            query,
            similarity_threshold=similarity_threshold,
            top_k=None,
            device=device,
        )
        for segment in segments:
            segment_copy = segment.copy()
            segment_copy["file_path"] = str(path)
            matches.append(segment_copy)

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
        description="Search postprocessed tree JSON files using CLIP text similarity."
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
        default=0.2,
        help="Minimum cosine similarity to keep a match (default: 0.2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Torch device override (default: auto-detect).",
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
    # track time, memory
    tracemalloc.start()
    start_time = time.perf_counter()
    matches = find_video_segments_in_dir(
        query=query,
        data_dir=data_dir,
        similarity_threshold=args.threshold,
        top_k=args.top_k,
        device=args.device,
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
                "device": args.device,
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
        matched_text = match.get("matched_text", "")
        print(f"{idx:02d}. score={score:.3f} video={video_id} "
              f"time={start}-{end} file={file_path}\n    {matched_text}")

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
            "device": args.device,
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
