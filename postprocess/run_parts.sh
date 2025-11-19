#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/home/kylee/workspace/LongVALE/postprocess/postprocess.py
OUTDIR=/home/kylee/workspace/LongVALE/data_backup/after_postprocess
SPEECH=/home/kylee/workspace/LongVALE/data_backup/speech_asr_1171
NOTJSON=/home/kylee/workspace/LongVALE/logs/wrong_sample_1015.txt

CUDA_VISIBLE_DEVICES=4 python -u "$SCRIPT" --input /home/kylee/workspace/LongVALE/data_backup/before_postprocess/Tree-Step3_part1.json --output-dir "$OUTDIR" --speech-json-dir "$SPEECH" --not-json-dir "$NOTJSON" &
CUDA_VISIBLE_DEVICES=4 python -u "$SCRIPT" --input /home/kylee/workspace/LongVALE/data_backup/before_postprocess/Tree-Step3_part2.json --output-dir "$OUTDIR" --speech-json-dir "$SPEECH" --not-json-dir "$NOTJSON" &
CUDA_VISIBLE_DEVICES=5 python -u "$SCRIPT" --input /home/kylee/workspace/LongVALE/data_backup/before_postprocess/Tree-Step3_part3.json --output-dir "$OUTDIR" --speech-json-dir "$SPEECH" --not-json-dir "$NOTJSON" &
CUDA_VISIBLE_DEVICES=5 python -u "$SCRIPT" --input /home/kylee/workspace/LongVALE/data_backup/before_postprocess/Tree-Step3_part4.json --output-dir "$OUTDIR" --speech-json-dir "$SPEECH" --not-json-dir "$NOTJSON" &

wait
echo "모든 파트 처리 완료"
