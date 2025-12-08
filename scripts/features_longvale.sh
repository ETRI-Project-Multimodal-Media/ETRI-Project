#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

EXTRACT_MODALITY=${1:-all}

GPU_ID=0

DATA_PATH=./data/annotation.json
VIDEO_DIR=./data/raw_data/video
AUDIO_DIR=./data/raw_data/audio
SAVE_DIR=./data/features_model

CLIP_CKPT=./checkpoints/ViT-L-14.pt
BEATS_CKPT=./checkpoints/BEATs_iter3_plus_AS20K.pt
WHISPER_CKPT=./checkpoints/openai-whisper-large-v2

if [[ "$EXTRACT_MODALITY" == "video" || "$EXTRACT_MODALITY" == "all" ]]; then
    echo "Extracting Video features..."
    python src/preprocess/clip_feature_extract.py \
        --annotation $DATA_PATH \
        --video_dir $VIDEO_DIR \
        --save_dir $SAVE_DIR/video_features \
        --checkpoint $CLIP_CKPT \
        --gpu_id $GPU_ID
fi

if [[ "$EXTRACT_MODALITY" == "audio" || "$EXTRACT_MODALITY" == "all" ]]; then
    echo "Extracting Audio features..."
    python src/preprocess/beats_feature_extract.py \
        --annotation $DATA_PATH \
        --audio_dir $AUDIO_DIR \
        --save_dir $SAVE_DIR/audio_features \
        --checkpoint $BEATS_CKPT \
        --gpu_id $GPU_ID
fi

if [[ "$EXTRACT_MODALITY" == "speech" || "$EXTRACT_MODALITY" == "all" ]]; then
    echo "Extracting Speech features..."
    python src/preprocess/whisper_feature_extract.py \
        --annotation $DATA_PATH \
        --audio_dir $AUDIO_DIR \
        --save_dir $SAVE_DIR/speech_features \
        --checkpoint $WHISPER_CKPT \
        --gpu_id $GPU_ID
fi

if [[ "$EXTRACT_MODALITY" == "speech_asr" || "$EXTRACT_MODALITY" == "all" ]]; then
    echo "Extracting Speech ASR..."
    python src/preprocess/whisper_speech_asr.py \
        --annotation $DATA_PATH \
        --audio_dir $AUDIO_DIR \
        --save_dir $SAVE_DIR/speech_asr \
        --checkpoint $WHISPER_CKPT \
        --gpu_id $GPU_ID
fi
