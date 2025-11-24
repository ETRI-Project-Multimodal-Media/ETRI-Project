#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export HUGGINGFACE_HUB_TOKEN="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"

TREE_SAVE_PATH=./outputs/log.json
POST_SAVE_DIR=./outputs/postprocess
SPEECH_ASR_DIR=./data/features_eval/speech_asr
DEBUG_PATH=./logs/debug.text

GPU_ID="7"

CUDA_VISIBLE_DEVICES=$GPU_ID python src/postprocess/postprocess.py \
    --input "$TREE_SAVE_PATH" \
    --output-dir "$POST_SAVE_DIR" \
    --speech-json-dir "$SPEECH_ASR_DIR" \
    --not-json-dir "$DEBUG_PATH"
