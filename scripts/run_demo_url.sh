#!/bin/bash
source $HOME/anaconda3/etc/profile.d/conda.sh # or miniconda
export PYTHONPATH=src:$PYTHONPATH

INPUT_SOURCE=$1 # Input: Video Link 
QUERY_STR=$2 # Input: Query 

GPU_ID=0 # Set this to GPU ID
BASE_DIR=path # Set this to base directory 

DEMO_DIR=$BASE_DIR/demo
TREE_SAVE_PATH=$DEMO_DIR/outputs/log.json 
POST_SAVE_DIR=$DEMO_DIR/outputs 

RESULT_SAVE_DIR=$POST_SAVE_DIR/demo.json
QUERY_SAVE_DIR=$POST_SAVE_DIR/query/demo.json

PROMPT_PATH=$BASE_DIR/data/prompt.json

CLIP_CKPT=$BASE_DIR/checkpoints/ViT-L-14.pt
BEATS_CKPT=$BASE_DIR/checkpoints/BEATs_iter3_plus_AS20K.pt
WHISPER_CKPT=$BASE_DIR/checkpoints/openai-whisper-large-v2

MODEL_BASE=$BASE_DIR/checkpoints/vicuna-7b-v1.5
MODEL_STAGE2=$BASE_DIR/checkpoints/longvalellm-vicuna-v1-5-7b/longvale-vicuna-v1-5-7b-stage2-bp
MODEL_STAGE3=$BASE_DIR/checkpoints/longvalellm-vicuna-v1-5-7b/longvale-vicuna-v1-5-7b-stage3-it
MODEL_MM_MLP=$BASE_DIR/checkpoints/vtimellm_stage1_mm_projector.bin 

SIMILARITY_THRESHOLD=0.9

if [ -z "$HUGGINGFACE_HUB_TOKEN" ] && [ -n "$HF_TOKEN" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

if [ -z "$HUGGINGFACE_HUB_TOKEN" ] && [ -z "$HF_TOKEN" ]; then
    echo "No HuggingFace token in environment."
    exit 1
fi

if [ -z "$INPUT_SOURCE" ] || [ -z "$QUERY_STR" ]; then
    echo "Usage: $0 <VIDEO_URL> <QUERY>"
    echo "Example: $0 https://www.youtube.com/watch?v=abc123 'event'"
    exit 1
fi

for cmd in ffmpeg ffprobe yt-dlp; do
    if ! command -v $cmd >/dev/null 2>&1; then
        echo "Error: $cmd is not installed. Please install $cmd."
        exit 1
    fi
done

TEMP_DIR=$(mktemp -d "${DEMO_DIR}/tmp.XXXXXX")
cleanup() {
    if [ "${KEEP_TEMP:-0}" = "1" ]; then
        echo "KEEP_TEMP=1: keeping temp dir $TEMP_DIR"
    else
        rm -rf "$TEMP_DIR" || true
    fi
}   
trap cleanup EXIT

DATA_PATH=$TEMP_DIR/demo.json
VIDEO_PATH=$TEMP_DIR/demo.mp4
AUDIO_PATH=$TEMP_DIR/demo.wav 

if [[ "$INPUT_SOURCE" =~ ^http.* ]]; then
    echo "Downloading video from URL..."
    yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
        --force-overwrites -o "$VIDEO_PATH" "$INPUT_SOURCE"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download video."
        exit 1
    fi
else
    echo "Error: Input '$INPUT_SOURCE' is not a valid URL."
    exit 1
fi

echo "Generating metadata (.json)..."
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO_PATH")

python3 -c "
import json
data = {
    'demo': {
        'duration': float('${DURATION}')
    }
}
with open('${DATA_PATH}', 'w') as f:
    json.dump(data, f, indent=4)
"

TREE_FEAT=$TEMP_DIR/features_tree
MODEL_FEAT=$TEMP_DIR/features_model
DEBUG_PATH=$TEMP_DIR/debug.text

echo "Extracting audio (.wav)..."
ffmpeg -y -i "$VIDEO_PATH" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_PATH"

echo "Running feature extraction..."
conda activate eventtree

python src/preprocess/tree_feature_extract.py \
    --data_path $DATA_PATH \
    --video_dir $TEMP_DIR \
    --audio_dir $TEMP_DIR \
    --clip_checkpoint $CLIP_CKPT \
    --beats_checkpoint $BEATS_CKPT \
    --whisper_checkpoint $WHISPER_CKPT \
    --save_dir $TREE_FEAT \
    --gpu_id $GPU_ID

python src/preprocess/clip_feature_extract.py \
    --annotation "$DATA_PATH" \
    --video_dir "$TEMP_DIR" \
    --save_dir "$MODEL_FEAT/video_features" \
    --checkpoint "$CLIP_CKPT" \
    --gpu_id $GPU_ID

python src/preprocess/beats_feature_extract.py \
    --annotation "$DATA_PATH" \
    --audio_dir "$TEMP_DIR" \
    --save_dir "$MODEL_FEAT/audio_features" \
    --checkpoint "$BEATS_CKPT" \
    --gpu_id $GPU_ID

python src/preprocess/whisper_feature_extract.py \
    --annotation "$DATA_PATH" \
    --audio_dir "$TEMP_DIR" \
    --save_dir "$MODEL_FEAT/speech_features" \
    --checkpoint "$WHISPER_CKPT" \
    --gpu_id $GPU_ID

python src/preprocess/whisper_speech_asr.py \
    --annotation "$DATA_PATH" \
    --audio_dir "$TEMP_DIR" \
    --save_dir "$MODEL_FEAT/speech_asr" \
    --checkpoint "$WHISPER_CKPT" \
    --gpu_id $GPU_ID

echo "Running main pipeline..."

python src/eventtree/tree/tree.py \
    --data_path $DATA_PATH \
    --video_feat_folder $TREE_FEAT/video_features \
    --audio_feat_folder $TREE_FEAT/audio_features \
    --speech_feat_folder $TREE_FEAT/speech_features \
    --save_path $TREE_SAVE_PATH

CUDA_VISIBLE_DEVICES=$GPU_ID python src/eventtree/caption_longvale.py \
    --tree_path $TREE_SAVE_PATH \
    --prompt_path $PROMPT_PATH \
    --save_path $TREE_SAVE_PATH \
    --video_feat_folder $MODEL_FEAT/video_features \
    --audio_feat_folder $MODEL_FEAT/audio_features \
    --asr_feat_folder $MODEL_FEAT/speech_features \
    --model_base $MODEL_BASE \
    --stage2 $MODEL_STAGE2 \
    --stage3 $MODEL_STAGE3 \
    --pretrain_mm_mlp_adapter $MODEL_MM_MLP \
    --similarity_threshold $SIMILARITY_THRESHOLD

conda activate eventtree-post

CUDA_VISIBLE_DEVICES=$GPU_ID python src/eventtree/summary_llama3.py \
    --tree_path $TREE_SAVE_PATH \
    --prompt_path $PROMPT_PATH \
    --save_path $TREE_SAVE_PATH
    
CUDA_VISIBLE_DEVICES=$GPU_ID python src/postprocess/postprocess.py \
    --input $TREE_SAVE_PATH \
    --output-dir $POST_SAVE_DIR \
    --speech-json-dir $MODEL_FEAT/speech_asr \
    --not-json-dir $DEBUG_PATH

CUDA_VISIBLE_DEVICES=$GPU_ID python src/query/search_queries.py \
    --input "$RESULT_SAVE_DIR" \
    --query "$QUERY_STR" \
    --mode text_embed \
    --output "$QUERY_SAVE_DIR"

echo "Demo completed. (Results are saved in $POST_SAVE_DIR, $QUERY_SAVE_DIR)"
