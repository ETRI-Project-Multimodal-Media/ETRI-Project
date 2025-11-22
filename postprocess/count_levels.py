#!/usr/bin/env python3
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

LEVEL_RE = re.compile(r"Level_(\d+)", re.IGNORECASE)

def collect_counts(base_dir: Path):
    counts = defaultdict(Counter)
    overall = Counter()

    for mp4_path in base_dir.rglob("*.mp4"):
        rel_parts = mp4_path.relative_to(base_dir).parts
        if not rel_parts:
            continue
        video_id = rel_parts[0]
        match = LEVEL_RE.search(mp4_path.stem)
        if not match:
            continue
        level = int(match.group(1))
        counts[video_id][level] += 1
        overall[level] += 1

    return counts, overall

def main():
    parser = argparse.ArgumentParser(description="Compare Level counts between two split_video_segments roots.")
    parser.add_argument("dir_a", type=Path, help="First root directory (e.g. .../20251104_sample_20_twfinch_refine_before)")
    parser.add_argument("dir_b", type=Path, help="Second root directory (e.g. .../20251104_sample_20_kmeans_refine_before)")
    parser.add_argument("--output", type=Path, default=Path("level_counts_comparison.csv"), help="Output CSV path")
    args = parser.parse_args()

    counts_a, overall_a = collect_counts(args.dir_a)
    counts_b, overall_b = collect_counts(args.dir_b)

    if args.output.parent and not args.output.parent.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)

    video_ids = sorted(set(counts_a) | set(counts_b))

    with args.output.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video_id", "level", "count_dir_a", "count_dir_b", "difference (b - a)"])

        for vid in video_ids:
            levels = sorted(set(counts_a.get(vid, Counter())) | set(counts_b.get(vid, Counter())))
            for lvl in levels:
                a_val = counts_a.get(vid, Counter()).get(lvl, 0)
                b_val = counts_b.get(vid, Counter()).get(lvl, 0)
                writer.writerow([vid, lvl, a_val, b_val, b_val - a_val])

        writer.writerow([])
        writer.writerow(["TOTALS"])
        total_levels = sorted(set(overall_a) | set(overall_b))
        for lvl in total_levels:
            a_val = overall_a.get(lvl, 0)
            b_val = overall_b.get(lvl, 0)
            writer.writerow([f"Level {lvl}", "", a_val, b_val, b_val - a_val])

if __name__ == "__main__":
    main()
