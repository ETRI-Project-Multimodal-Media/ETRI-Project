#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

EXTRACT_MODALITY=${1:-all}

GPU_ID=0

DATA_PATH=./data/annotation.json
VIDEO_DIR=./data/raw_data/video
AUDIO_DIR=./data/raw_data/audio
SAVE_DIR=./data/features_tree

CLIP_CKPT=./checkpoints/ViT-L-14.pt
BEATS_CKPT=./checkpoints/BEATs_iter3_plus_AS20K.pt
WHISPER_CKPT=./checkpoints/openai-whisper-large-v2

python src/preprocess/tree_feature_extract.py \
    --data_path $DATA_PATH \
    --video_dir $VIDEO_DIR \
    --audio_dir $AUDIO_DIR \
    --clip_checkpoint $CLIP_CKPT \
    --beats_checkpoint $BEATS_CKPT \
    --whisper_checkpoint $WHISPER_CKPT \
    --save_dir $SAVE_DIR \
    --extract_modality $EXTRACT_MODALITY \
    --gpu_id $GPU_ID
