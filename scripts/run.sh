#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

DATA_PATH=./data/longvale-annotations-eval.json
PROMPT_PATH=./data/prompt.json
SAVE_PATH=./outputs/log.json

TREE_V_FEAT=./data/features_tree/video_features
TREE_A_FEAT=./data/features_tree/audio_features
TREE_S_FEAT=./data/features_tree/speech_features

MODEL_V_FEAT=./data/features_eval/video_features
MODEL_A_FEAT=./data/features_eval/audio_features
MODEL_S_FEAT=./data/features_eval/speech_features

MODEL_BASE=./checkpoints/vicuna-7b-v1.5
MODEL_STAGE2=./checkpoints/longvalellm-vicuna-v1-5-7b/longvale-vicuna-v1-5-7b-stage2-bp
MODEL_STAGE3=./checkpoints/longvalellm-vicuna-v1-5-7b/longvale-vicuna-v1-5-7b-stage3-it
MODEL_MM_MLP=./checkpoints/vtimellm_stage1_mm_projector.bin 

GPU_ID=0

source ~/anaconda3/etc/profile.d/conda.sh
conda activate eventtree

python src/eventtree/tree/tree.py \
    --data_path $DATA_PATH \
    --video_feat_folder $TREE_V_FEAT \
    --audio_feat_folder $TREE_A_FEAT \
    --speech_feat_folder $TREE_S_FEAT \
    --save_path $SAVE_PATH

CUDA_VISIBLE_DEVICES=$GPU_ID python src/eventtree/caption_longvale.py \
    --tree_path $SAVE_PATH \
    --prompt_path $PROMPT_PATH \
    --save_path $SAVE_PATH \
    --video_feat_folder $MODEL_V_FEAT \
    --audio_feat_folder $MODEL_A_FEAT \
    --asr_feat_folder $MODEL_S_FEAT \
    --model_base $MODEL_BASE \
    --stage2 $MODEL_STAGE2 \
    --stage3 $MODEL_STAGE3 \
    --pretrain_mm_mlp_adapter $MODEL_MM_MLP \
    --similarity_threshold 0.9

conda activate eventtree-post

CUDA_VISIBLE_DEVICES=$GPU_ID python src/eventtree/summary_llama3.py \
    --tree_path $SAVE_PATH \
    --prompt_path $PROMPT_PATH \
    --save_path $SAVE_PATH \
