# 예: "train.py"라는 문자열이 포함된 프로세스를 계속 체크
while pgrep -f "instruction_tuning.sh" >/dev/null; do
    sleep 60   # 60초마다 확인 (필요에 따라 조정)
done

# 학습이 종료되면 eval 실행
CUDA_VISIBLE_DEVICES=7 python /home/kylee/workspace/LongVALE/longvalellm/eval/eval.py --data_path /home/kylee/workspace/LongVALE/data/longvale-annotations-eval.json --video_feat_folder /home/kylee/workspace/LongVALE/data/features_eval/video_features_1171 --audio_feat_folder /home/kylee/workspace/LongVALE/data/features_eval/audio_features_1171 --asr_feat_folder /home/kylee/workspace/LongVALE/data/features_eval/speech_features_1171 --task all --log_path /home/kylee/workspace/LongVALE/eval_logs
