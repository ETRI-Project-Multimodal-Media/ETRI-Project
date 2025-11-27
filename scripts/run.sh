#!/bin/bash
source $HOME/anaconda3/etc/profile.d/conda.sh # or miniconda
export PYTHONPATH=src:$PYTHONPATH

GPU_ID=0

DATA_PATH=./data/annotation.json
PROMPT_PATH=./data/prompt.json

TREE_SAVE_PATH=./outputs/log.json
POST_SAVE_DIR=./outputs/postprocess
DEBUG_PATH=./logs/debug.text

TREE_V_FEAT=./data/features_tree/video_features
TREE_A_FEAT=./data/features_tree/audio_features
TREE_S_FEAT=./data/features_tree/speech_features

MODEL_V_FEAT=./data/features_model/video_features
MODEL_A_FEAT=./data/features_model/audio_features
MODEL_S_FEAT=./data/features_model/speech_features
SPEECH_ASR_DIR=./data/features_model/speech_asr

MODEL_BASE=./checkpoints/vicuna-7b-v1.5
MODEL_STAGE2=./checkpoints/longvalellm-vicuna-v1-5-7b/longvale-vicuna-v1-5-7b-stage2-bp
MODEL_STAGE3=./checkpoints/longvalellm-vicuna-v1-5-7b/longvale-vicuna-v1-5-7b-stage3-it
MODEL_MM_MLP=./checkpoints/vtimellm_stage1_mm_projector.bin 

SIMILARITY_THRESHOLD=0.9

if [ -z "$HUGGINGFACE_HUB_TOKEN" ] && [ -n "$HF_TOKEN" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

if [ -z "$HUGGINGFACE_HUB_TOKEN" ] && [ -z "$HF_TOKEN" ]; then
    echo "No HuggingFace token in environment."
    exit 1
fi

echo "Running main pipeline..."
conda activate eventtree

python src/eventtree/tree/tree.py \
    --data_path $DATA_PATH \
    --video_feat_folder $TREE_V_FEAT \
    --audio_feat_folder $TREE_A_FEAT \
    --speech_feat_folder $TREE_S_FEAT \
    --save_path $TREE_SAVE_PATH

CUDA_VISIBLE_DEVICES=$GPU_ID python src/eventtree/caption_longvale.py \
    --tree_path $TREE_SAVE_PATH \
    --prompt_path $PROMPT_PATH \
    --save_path $TREE_SAVE_PATH \
    --video_feat_folder $MODEL_V_FEAT \
    --audio_feat_folder $MODEL_A_FEAT \
    --asr_feat_folder $MODEL_S_FEAT \
    --model_base $MODEL_BASE \
    --stage2 $MODEL_STAGE2 \
    --stage3 $MODEL_STAGE3 \
    --pretrain_mm_mlp_adapter $MODEL_MM_MLP \
    --similarity_threshold $SIMILARITY_THRESHOLD

conda activate eventtree-post

CUDA_VISIBLE_DEVICES=$GPU_ID python src/eventtree/summary_llama3.py \
    --tree_path $TREE_SAVE_PATH \
    --prompt_path $PROMPT_PATH \
    --save_path $TREE_SAVE_PATH \

CUDA_VISIBLE_DEVICES=$GPU_ID python src/postprocess/postprocess.py \
    --input $TREE_SAVE_PATH \
    --output-dir $POST_SAVE_DIR \
    --speech-json-dir $SPEECH_ASR_DIR \
    --not-json-dir $DEBUG_PATH
