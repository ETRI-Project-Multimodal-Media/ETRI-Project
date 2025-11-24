#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

POST_OUTPUT_DIR="./Example/postprocess"
TREE_FILE="./Example/Tree-Step3_part1.json"
VIDEO_DIR="$POST_OUTPUT_DIR"

# 3) benchmark 실험
python src/query/benchmark_queries.py

# 4) domain threshold analysis
python src/query/domain_threshold_analysis.py \
  --tree-file "$TREE_FILE" \
  --video-dir "$VIDEO_DIR" \
  --output "./temp_data/query/domain_topk_stats.json"
