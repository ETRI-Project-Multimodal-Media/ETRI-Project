import json
import math
import os
import argparse

from collections import Counter
from typing import Any, Dict, List, Optional


KL_SIMILARITY_THRESHOLD = 0.01
KL_EPS = 1e-8
JSD_SIMILARITY_THRESHOLD = 0.25


def _safe_float(value: Optional[float], default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _token_distribution(text: str) -> Dict[str, float]:
    tokens = text.split()
    if not tokens:
        return {}

    counter = Counter(tokens)
    total = sum(counter.values())
    if total == 0:
        return {}

    return {token: count / total for token, count in counter.items()}


def _kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = KL_EPS) -> float:
    if not p:
        return 0.0

    tokens = set(p) | set(q)
    divergence = 0.0
    for token in tokens:
        p_val = p.get(token, eps)
        q_val = q.get(token, eps)
        divergence += p_val * math.log((p_val + eps) / (q_val + eps))
    return divergence


def _js_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = KL_EPS) -> float:
    if not p and not q:
        return 0.0

    tokens = set(p) | set(q)
    P = {t: p.get(t, eps) for t in tokens}
    Q = {t: q.get(t, eps) for t in tokens}
    M = {t: 0.5 * (P[t] + Q[t]) for t in tokens}

    def _kl(A, B):
        s = 0.0
        for t in tokens:
            s += A[t] * math.log((A[t] + eps) / (B[t] + eps))
        return s

    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)


def merge_segments_by_kl(segments, similarity_threshold=KL_SIMILARITY_THRESHOLD):
    if not segments:
        return []

    merged = []
    idx = 0
    total = len(segments)

    while idx < total:
        start, end, text = segments[idx]
        first_text = text
        prev_dist = _token_distribution(text)
        last_end = end
        next_idx = idx + 1

        while next_idx < total:
            next_text = segments[next_idx][2]
            next_dist = _token_distribution(next_text)
            divergence = _kl_divergence(prev_dist, next_dist)
            if divergence >= similarity_threshold:
                break
            last_end = segments[next_idx][1]
            prev_dist = next_dist
            next_idx += 1

        merged.append((start, last_end, first_text.strip()))
        idx = next_idx

    return merged


def merge_segments_by_jsd(segments, similarity_threshold=JSD_SIMILARITY_THRESHOLD):
    if not segments:
        return []

    merged = []
    idx = 0
    total = len(segments)

    while idx < total:
        start, end, text = segments[idx]
        first_text = text
        prev_dist = _token_distribution(text)
        last_end = end
        next_idx = idx + 1

        while next_idx < total:
            next_text = segments[next_idx][2]
            next_dist = _token_distribution(next_text)
            divergence = _js_divergence(prev_dist, next_dist)
            if divergence >= similarity_threshold:
                break
            last_end = segments[next_idx][1]
            prev_dist = next_dist
            next_idx += 1

        merged.append((start, last_end, first_text.strip()))
        idx = next_idx

    return merged


def _child_sort_key(node: Dict[str, Any]):
    return (_safe_float(node.get("start_time")), _safe_float(node.get("end_time")))


def _merge_node_list(nodes: List[Dict[str, Any]], similarity_threshold=JSD_SIMILARITY_THRESHOLD):
    valid_nodes = [node for node in nodes if isinstance(node, dict)]
    if not valid_nodes:
        return []

    sorted_nodes = sorted(valid_nodes, key=_child_sort_key)
    merged = []
    idx = 0
    total = len(sorted_nodes)

    while idx < total:
        current = sorted_nodes[idx]
        level = current.get("level")
        caption = (current.get("caption") or "").strip()
        caption_parts = [caption] if caption else []
        start_time = _safe_float(current.get("start_time"))
        end_time = _safe_float(current.get("end_time"))
        combined_children = list(current.get("children") or [])
        prev_dist = _token_distribution(caption)
        next_idx = idx + 1

        while next_idx < total:
            nxt = sorted_nodes[next_idx]
            next_caption = (nxt.get("caption") or "").strip()
            next_dist = _token_distribution(next_caption)
            divergence = _js_divergence(prev_dist, next_dist)
            if divergence >= similarity_threshold:
                break
            start_time = min(start_time, _safe_float(nxt.get("start_time")))
            end_time = max(end_time, _safe_float(nxt.get("end_time")))
            if next_caption:
                caption_parts.append(next_caption)
                prev_dist = _token_distribution(next_caption)
            combined_children.extend(nxt.get("children") or [])
            next_idx += 1

        current["start_time"] = start_time
        current["end_time"] = end_time
        if caption_parts:
            current["caption"] = caption_parts[0] # can change to summary,.. etc
        current["children"] = sorted(combined_children, key=_child_sort_key) # 합친 node의 자식 node는 합쳐서 시간 순으로 정렬
        current["level"] = level
        merged.append(current)
        idx = next_idx

    return merged


# node 아래의 모든 자식 노드를 level별로 합침.
def merge_children_recursive(node: Dict[str, Any], similarity_threshold=JSD_SIMILARITY_THRESHOLD):
    if not isinstance(node, dict):
        return

    children = node.get("children") or []
    if not children:
        node["children"] = []
        return

    for child in children:
        merge_children_recursive(child, similarity_threshold)

    node["children"] = _merge_node_list(children, similarity_threshold)


def _save_tree(root: Dict[str, Any], save_path: str):
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(root, f, ensure_ascii=False, indent=2)


def merge_tree(
    root: Dict[str, Any],
    similarity_threshold=JSD_SIMILARITY_THRESHOLD,
    save_path: Optional[str] = None,
):
    merge_children_recursive(root, similarity_threshold)
    if save_path:
        _save_tree(root, save_path)
    return root


def parse_args():
    parser = argparse.ArgumentParser(description="KL/JSD-based tree merging utility")
    parser.add_argument("--input", required=False, default="/home/kylee/workspace/LongVALE/data_backup/before_postprocess/Tree-Step3_part1.json",  help="Path to input tree JSON")
    parser.add_argument("--output", required=False, default="/home/kylee/workspace/LongVALE/data_backup/tree_merge.json", help="Path to save merged tree JSON")
    parser.add_argument("--threshold", type=float, default=JSD_SIMILARITY_THRESHOLD,
                        help="JSD similarity threshold")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Loaded {args.input}")
    print(f"[INFO] Merging using JSD threshold = {args.threshold}")

    for video_id, tree in data.items():
        merge_tree(tree, similarity_threshold=args.threshold, save_path=args.output)
    output_path = "/home/kylee/workspace/LongVALE/data_backup/tree_merge_{}.json".format(args.threshold)
    _save_tree(data, output_path)

    print(f"[INFO] Saved merged tree → {args.output}")


if __name__ == "__main__":
    main()

