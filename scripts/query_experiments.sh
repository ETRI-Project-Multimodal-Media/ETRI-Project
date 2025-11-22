#!/bin/bash

REPO_ROOT=./  # change properly
POST_OUTPUT_DIR="$REPO_ROOT/Example/postprocess"    
TREE_FILE="$REPO_ROOT/Example/Tree-Step3_part1.json"   
VIDEO_DIR="$POST_OUTPUT_DIR"                           

# 3) benchmark 실험
python query/benchmark_queries.py

# 4) domain threshold analysis

python query/domain_threshold_analysis.py \
  --tree-file "$TREE_FILE" \
  --video-dir "$VIDEO_DIR" \
  --output "$REPO_ROOT/temp_data/query/domain_topk_stats.json"