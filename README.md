# ETRI Project

멀티모달 AI 기반 미디어 핵심 정보 분석 및 요약 기술

## Environment Setup
```bash
# Environment 1 for LongVALE 
conda create --name eventtree python=3.10
conda activate eventtree
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
```

```bash
# Environment 2 for LLaMA3 
conda create --name eventtree2 --clone eventtree
conda activate eventtree2
pip install transformers==4.40.0
```

## Data Setup
```shell
# Tree Features Extraction 
# Type: all, video, audio, speech
bash scripts/features_tree.sh <TYPE>
```

```shell
# LongVALE Features Extraction 
# Type: all, video, audio, speech, speech_asr
# https://github.com/ttgeng233/LongVALE
bash scripts/features_longvale.sh <TYPE>
```

```
data/
├── annotation.json
├── prompt.json
├── raw_data
    ├── video_test/{video_id}.mp4
    ├── audio_test/{video_id}.wav
├──  features_tree
    ├── video_features/{video_id}.npy
    ├── audeo_features/{video_id}.npy
    ├── speech_features/{video_id}.npy
├──  features_eval
    ├── video_features/{video_id}.npy
    ├── audio_features/{video_id}.npy
    ├── speech_features/{video_id}.npy
    ├── speech_asr/{video_id}.json
```

## How to Run
```shell
# Main
bash scripts/run.sh
```

```shell
... 

source ~/anaconda3/etc/profile.d/conda.sh
conda activate eventtree

# Tree Construct 
python src/eventtree/tree/tree.py \
    --data_path $DATA_PATH \
    --video_feat_folder $TREE_V_FEAT \
    --audio_feat_folder $TREE_A_FEAT \
    --speech_feat_folder $TREE_S_FEAT \
    --save_path $SAVE_PATH

# LongVALE - Leaf Node Captioning
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

# LLaMA3 - Internal Node Captioning
conda activate eventtree2

CUDA_VISIBLE_DEVICES=$GPU_ID python src/eventtree/summary_llama3.py \
    --tree_path $SAVE_PATH \
    --prompt_path $PROMPT_PATH \
    --save_path $SAVE_PATH \
```
