# 3) benchmark 실험
python query/benchmark_queries.py

# 4) domain threshold analysis
TREE_FILE="$REPO_ROOT/data_backup/before_postprocess/Tree-Step3.json"   # 전체 트리 (level-1 caption용)
VIDEO_DIR="$POST_OUTPUT_DIR"                                            # 1번 output 디렉토리

python query/domain_threshold_analysis.py \
  --tree-file "$TREE_FILE" \
  --video-dir "$VIDEO_DIR" \
  --output "$REPO_ROOT/temp_data/query/domain_topk_stats.json"