#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

POST_SAVE_DIR=./outputs/postprocess
SPEECH_JSON_DIR="./Example/speech_asr_1171"
VIDEO_JSON="$POST_SAVE_DIR/sO3wd7X-l7U.json"   
QUERY_SAVE_DIR=./outputs/query/example.json
QUERY_STR="Throws javelin in the air"
GPU_ID="7"

CUDA_VISIBLE_DEVICES=$GPU_ID python src/query/search_queries.py \
  --input "$VIDEO_JSON" \
  --query "$QUERY_STR" \
  --mode text_embed \
  --output "$QUERY_SAVE_DIR"
