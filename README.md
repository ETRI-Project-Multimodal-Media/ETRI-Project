# ETRI Project

멀티모달 AI 기반 미디어 핵심 정보 분석 및 요약 기술

## Environment Setup
```bash
# Environment 1 for LongVALE
# 1-1. Tree Construct
# 1-2. Leaf Node Captioning  
conda create --name eventtree python=3.10
conda activate eventtree
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
```

```bash
# Environment 2 for LLaMA3
# 2-1. Internal Node Captioning
# 2-2. Structured Data Postprocessing
conda create --name eventtree-post python=3.10
conda activate eventtree-post
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install cuda-toolkit==12.8.1
pip install transformers accelerate peft
pip install decord
pip install jsonschema
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

기본 데이터 디렉터리 구성 예시는 다음과 같습니다.

```text
data/
├── annotation.json
├── prompt.json
├── raw_data/
│   ├── video_test/{video_id}.mp4
│   └── audio_test/{video_id}.wav
├── features_tree/
│   ├── video_features/{video_id}.npy
│   ├── audio_features/{video_id}.npy
│   └── speech_features/{video_id}.npy
├── features_eval/
│   ├── video_features/{video_id}.npy
│   ├── audio_features/{video_id}.npy
│   ├── speech_features/{video_id}.npy
│   └── speech_asr/{video_id}.json
```

`scripts/postprocess.sh` 에서 사용하는 예시 데이터/출력 경로는 다음과 같습니다.

```text
Example/
├── Tree-Step3_part1.json
├── Tree-Step3_part2.json
├── Tree-Step3_part3.json
├── Tree-Step3_part4.json
├── speech_asr_1171/
│   └── {video_id}.json        # ASR JSON (SPEECH_JSON_DIR)
└── postprocess/               # postprocess.py 출력 (POST_OUTPUT_DIR)
    └── {video_id}.json

outputs/
└── log.json                   # Tree/LongVALE 파이프라인 결과 (SAVE_PATH)

logs/
└── debug.txt                  # 잘못된 샘플 로그 (DEBUG_LOG)
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

conda activate eventtree-post

# LLaMA3 - Internal Node Captioning
CUDA_VISIBLE_DEVICES=$GPU_ID python src/eventtree/summary_llama3.py \
    --tree_path $SAVE_PATH \
    --prompt_path $PROMPT_PATH \
    --save_path $SAVE_PATH \

# LLaMA3 - Postprocessing
CUDA_VISIBLE_DEVICES=$GPU_ID python src/postprocess/postprocess.py \
  --input "$SAVE_PATH" \
  --output-dir "$POST_OUTPUT_DIR" \
  --speech-json-dir "$SPEECH_JSON_DIR" \
  --not-json-dir "$DEBUG_LOG"

# Query
CUDA_VISIBLE_DEVICES=$GPU_ID python src/query/search_queries.py \
  --input "$VIDEO_JSON" \
  --query "$QUERY_STR" \
  --mode text_embed \
  --output "$REPO_ROOT/query/example_result.json"

# Query Experiment 1
CUDA_VISIBLE_DEVICES=$GPU_ID python src/query/benchmark_queries.py

# Query Experiment 2
CUDA_VISIBLE_DEVICES=$GPU_ID python src/query/domain_threshold_analysis.py \
  --tree-file "$TREE_FILE" \
  --video-dir "$VIDEO_DIR" \
  --output "$REPO_ROOT/query/domain_topk_stats.json"
```
