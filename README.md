# ETRI Project

멀티모달 AI 기반 미디어 핵심 정보 분석 및 요약 기술

## Environment Setup
  LongVALE: https://github.com/ttgeng233/LongVALE
```bash
# Environment 1 for LongVALE
# 1-1. Tree Construct
# 1-2. Leaf Node Captioning  
conda create --name eventtree python=3.10
conda activate eventtree
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
pip install soundfile
pip install streamlit
```

  LLaMA3: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
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
pip install sentence-transformers
```

## Data Setup
- `annotation.json`, `{video_id}.mp4`, `{video_id}.wav`이 필요합니다. 
- `annotation.json` 형식은 `data/example.json`와 같으며, video id (YouTube id)와 duration이 필요합니다. 

```shell
# Tree Feature Extraction (features_tree)
# Type: all, video, audio, speech
bash scripts/features_tree.sh <TYPE>
```

```shell
# LongVALE Feature Extraction (features_eval)
# Type: all, video, audio, speech, speech_asr
bash scripts/features_longvale.sh <TYPE>
```

`data` 디렉터리 구성 예시는 다음과 같습니다.

```text
data/
├── annotation.json
├── prompt.json
├── raw_data/
│   ├── video_test/{video_id}.mp4 # input
│   └── audio_test/{video_id}.wav # input
├── features_tree/
│   ├── video_features/{video_id}.npy
│   ├── audio_features/{video_id}.npy
│   └── speech_features/{video_id}.npy
└── features_eval/
    ├── video_features/{video_id}.npy
    ├── audio_features/{video_id}.npy
    ├── speech_features/{video_id}.npy
    └── speech_asr/{video_id}.json
```
    
`scripts/postprocess.sh` 에서 사용하는 예시 데이터/출력 경로는 다음과 같습니다.

```text
outputs/
├── log.json  # Tree/LongVALE 파이프라인 결과 (SAVE_PATH)
├── postprocess/
│   └── {video_id}.json  
└── query/               
    └── example.json
    
logs/
└── debug.txt                  # 잘못된 샘플 로그 (DEBUG_LOG)
```
    
## Checkpoint Setup

| Modality      | Encoder | Checkpoint path                           | Download checkpoint                                                                 |
|---------------|---------|-------------------------------------------|-------------------------------------------------------------------------------------|
| Visual        | CLIP    | `./checkpoints/ViT-L-14.pt`               | [ViT-L/14](https://github.com/openai/CLIP)                                         |
| Audio         | BEATs   | `./checkpoints/BEATs_iter3_plus_AS20K.pt` | [BEATs_iter3_plus_AS20K](https://github.com/microsoft/unilm/tree/master/BEATs)     |
| Speech        | Whisper | `./checkpoints/openai-whisper-large-v2`   | [whisper-large-v2](https://huggingface.co/openai/whisper-large-v2)                 |

- LongVALE: Download [Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) and [vtimellm_stage1](https://huggingface.co/datasets/ttgeng233/LongVALE/blob/main/checkpoints/vtimellm_stage1_mm_projector.bin) weights.
- LongVALE: Download LongVALE-LLM model from [longvalellm-vicuna-v1-5-7b.tar.gz](https://huggingface.co/datasets/ttgeng233/LongVALE/blob/main/checkpoints/longvalellm-vicuna-v1-5-7b.tar.gz).
  
`checkpoints` 디렉터리 구성 예시는 다음과 같습니다.

```text
checkpoints/
├── vicuna-7b-v1.5
├── longvale-vicuna-v1-5-7b-stage2-bp
├── longvale-vicuna-v1-5-7b-stage3-it
├── vtimellm_stage1_mm_projector.bin 
├── ViT-L-14.pt
├── BEATs_iter3_plus_AS20K.pt
└── openai-whisper-large-v2
```

## How to Run

- Main 
```shell
bash scripts/run.sh
```

- Streamlit Demo 
```shell
streamlit run streamlit_demo.py --server.address 0.0.0.0 --server.port 8501
ssh -L 8501:172.17.0.7:8501 Docker_206 # in-case portforward
```

- Demo (Video file)
- Ex. bash scripts/run_demo.sh Abc123 'Event'

```shell
bash scripts/run_demo.sh <VIDEO_ID> <QUERY>
```

- Demo (Video link)
- Ex. bash scripts/run_demo_url.sh https://www.youtube.com/watch?v=Abc123 'Event'

```shell
bash scripts/run_demo_url.sh <VIDEO_LINK> <QUERY>
```
