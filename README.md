# ETRI-Project

멀티모달 AI 기반 미디어 핵심 정보 분석 및 요약 기술

## Environment Setup
```bash
# Environment 1 for LongVALE 
conda create -n eventtree python=3.10
conda activate eventtree
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
```

```bash
# Environment 2 for LLaMA3 
conda create -n eventtree2 python=3.10
conda activate eventtree2
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install transformers==4.40.0
```

## Data/Feature Setup
```shell
# Tree Features Extraction 
bash scripts/features_tree.sh

# LongVALE Features Extraction 
# https://github.com/ttgeng233/LongVALE
bash scripts/features_longvale.sh
```

```
data/
├── annotation.json
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
bash scripts/run.sh
```
