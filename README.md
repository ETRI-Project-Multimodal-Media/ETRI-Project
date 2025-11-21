# ETRI-Project

멀티모달 AI 기반 미디어 핵심 정보 분석 및 요약 기술

## Usage

1. **Clone this repo**

    ```bash
    git clone https://github.com/Jang-Jinho/ETRI-project.git
    ```

2. **Setting the environment**
    - conda env. 1 for LongVALE
    ```
    conda create -n eventtree python=3.10
    conda activate eventtree
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    pip install flash-attn==2.3.6 --no-build-isolation
    ```

    - conda env. 2 for LLaMA3
    ```
    conda create -n eventtree2 python=3.10
    conda activate eventtree2
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    pip install transformers==4.40.0
    ```
3. **Prepare data and features**
    - Tree Features
    
    ```shell
    bash scripts/features_tree.sh
    ```
    - LongVALE Features
    - https://github.com/ttgeng233/LongVALE
    
    ```shell
    bash scripts/features_longvale.sh
    ```
    - Data Directory Structure
    
    ```
    data
    ┣ annotation.json
    ┣ raw_data
    ┃ ┣ video_test
    ┃ ┣ audio_test
    ┣ features_tree
    ┃ ┣ video_features
    ┃ ┣ audeo_features
    ┃ ┣ speech_features
    ┣ features_eval
    ┃ ┣ video_features
    ┃ ┣ audio_features
    ┃ ┣ speech_features
    ┗ ┗ speech_asr
    ```
4. **Run**

    ```shell
    bash scripts/run.sh
    ```
