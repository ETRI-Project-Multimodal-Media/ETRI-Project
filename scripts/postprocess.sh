set -e
# change properly
REPO_ROOT=/home/kylee/workspace/LongVALE 
cd "$REPO_ROOT"

# change properly
source /root/anaconda3/etc/profile.d/conda.sh
conda activate longvale   

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# 1) postprocess
POST_INPUTS=(
  "$REPO_ROOT/Example/Tree-Step3_part1.json"
  "$REPO_ROOT/Example/Tree-Step3_part2.json"
  "$REPO_ROOT/Example/Tree-Step3_part3.json"
  "$REPO_ROOT/Example/Tree-Step3_part4.json"
)
POST_OUTPUT_DIR="$REPO_ROOT/data/postprocess"        # domain_threshold 의 --video-dir 와 맞춰주는 게 편함
SPEECH_JSON_DIR="$REPO_ROOT/data/speech_asr_1171"
NOT_JSON_LOG="$REPO_ROOT/logs/debug.txt"

python postprocess/postprocess.py \
  --input "${POST_INPUTS[@]}" \
  --output-dir "$POST_OUTPUT_DIR" \
  --speech-json-dir "$SPEECH_JSON_DIR" \
  --not-json-dir "$NOT_JSON_LOG"

# 2) query search (예시 한 번)
VIDEO_JSON="$POST_OUTPUT_DIR/sO3wd7X-l7U.json"   # 보고 싶은 video id 에 맞게
QUERY_STR="Throws javelin in the air"

python query/search_queries.py \
  --input "$VIDEO_JSON" \
  --query "$QUERY_STR" \
  --mode heuristic \
  --output "$REPO_ROOT/temp_data/query/example_result.json"
