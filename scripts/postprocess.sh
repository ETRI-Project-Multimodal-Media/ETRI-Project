#!/bin/bash

REPO_ROOT=./  # change properly

SAVE_PATH=./outputs/log.json

POST_OUTPUT_DIR="$REPO_ROOT/Example/postprocess"    
SPEECH_JSON_DIR="$REPO_ROOT/Example/speech_asr_1171"
DEBUG_LOG="$REPO_ROOT/logs/debug.txt"
VIDEO_JSON="$POST_OUTPUT_DIR/sO3wd7X-l7U.json"   
QUERY_STR="Throws javelin in the air"

cd "$REPO_ROOT"

conda activate postprocess

# 1) postprocess

python src/postprocess/postprocess.py \
  --input "${POST_INPUTS[@]}" \
  --output-dir "$POST_OUTPUT_DIR" \
  --speech-json-dir "$SPEECH_JSON_DIR" \
  --not-json-dir "$DEBUG_LOG"

# 2) query search 

python src/query/search_queries.py \
  --input "$VIDEO_JSON" \
  --query "$QUERY_STR" \
  --mode heuristic \
  --output "$REPO_ROOT/temp_data/query/example_result.json"
