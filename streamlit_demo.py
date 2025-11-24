import os
import subprocess

import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, env=None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        shell=True,
        capture_output=True,
        text=True,
        env=full_env,
    )
    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    return result.returncode, output


st.title("LongVALE Pipeline Demo (run.sh wrapper)")

st.markdown(
    "이 데모는 `scripts/run.sh`가 수행하는 파이프라인 단계를 "
    "Streamlit UI에서 하나씩 실행해볼 수 있도록 만든 것입니다."
)

# 기본 경로/설정 (run.sh 기준)
st.sidebar.header("기본 설정")
data_path = st.sidebar.text_input(
    "DATA_PATH",
    "./data/longvale-annotations-eval.json",
)
prompt_path = st.sidebar.text_input(
    "PROMPT_PATH",
    "./data/prompt.json",
)
tree_save_path = st.sidebar.text_input(
    "TREE_SAVE_PATH",
    "./outputs/log.json",
)
post_save_dir = st.sidebar.text_input(
    "POST_SAVE_DIR",
    "./outputs/postprocess",
)
debug_path = st.sidebar.text_input(
    "DEBUG_PATH",
    "./logs/debug.text",
)

tree_v_feat = st.sidebar.text_input(
    "TREE_V_FEAT",
    "./data/features_tree/video_features",
)
tree_a_feat = st.sidebar.text_input(
    "TREE_A_FEAT",
    "./data/features_tree/audio_features",
)
tree_s_feat = st.sidebar.text_input(
    "TREE_S_FEAT",
    "./data/features_tree/speech_features",
)

model_v_feat = st.sidebar.text_input(
    "MODEL_V_FEAT",
    "./data/features_eval/video_features",
)
model_a_feat = st.sidebar.text_input(
    "MODEL_A_FEAT",
    "./data/features_eval/audio_features",
)
model_s_feat = st.sidebar.text_input(
    "MODEL_S_FEAT",
    "./data/features_eval/speech_features",
)
speech_asr_dir = st.sidebar.text_input(
    "SPEECH_ASR_DIR",
    "./data/features_eval/speech_asr",
)

model_base = st.sidebar.text_input(
    "MODEL_BASE",
    "./checkpoints/vicuna-7b-v1.5",
)
model_stage2 = st.sidebar.text_input(
    "MODEL_STAGE2",
    "./checkpoints/longvale-vicuna-v1-5-7b-stage2-bp",
)
model_stage3 = st.sidebar.text_input(
    "MODEL_STAGE3",
    "./checkpoints/longvale-vicuna-v1-5-7b-stage3-it",
)
model_mm_mlp = st.sidebar.text_input(
    "MODEL_MM_MLP",
    "./checkpoints/vtimellm_stage1_mm_projector.bin",
)

gpu_id = st.sidebar.text_input("GPU_ID (CUDA_VISIBLE_DEVICES)", "6")

st.sidebar.markdown("---")
st.sidebar.markdown("실행할 단계를 선택하세요:")
run_tree = st.sidebar.checkbox("1. Event Tree 생성 (tree.py)", value=False)
run_caption = st.sidebar.checkbox("2. Tree 캡셔닝 (caption_longvale.py)", value=False)
run_summary = st.sidebar.checkbox("3. Tree 요약 (summary_llama3.py)", value=False)
run_postprocess = st.sidebar.checkbox("4. Postprocess (postprocess.py)", value=True)

if "log_text" not in st.session_state:
    st.session_state.log_text = ""


log_area = st.empty()


def append_log(text):
    if st.session_state.log_text:
        st.session_state.log_text += "\n" + text
    else:
        st.session_state.log_text = text
    log_area.text(st.session_state.log_text)


if st.button("선택한 단계 실행"):
    st.session_state.log_text = ""
    log_area.text("")

    # 1. Event tree 생성
    if run_tree:
        append_log("[1] Event Tree 생성 시작...")
        cmd = (
            "python src/eventtree/tree/tree.py "
            f"--data_path {data_path} "
            f"--video_feat_folder {tree_v_feat} "
            f"--audio_feat_folder {tree_a_feat} "
            f"--speech_feat_folder {tree_s_feat} "
            f"--save_path {tree_save_path}"
        )
        code, out = run_command(cmd)
        append_log(f"$ {cmd}\n{out}")
        append_log(f"[1] 종료 코드: {code}")

    # 2. Caption 생성
    if run_caption:
        append_log("[2] Caption 생성 시작...")
        env = {"CUDA_VISIBLE_DEVICES": gpu_id}
        cmd = (
            "python src/eventtree/caption_longvale.py "
            f"--tree_path {tree_save_path} "
            f"--prompt_path {prompt_path} "
            f"--save_path {tree_save_path} "
            f"--video_feat_folder {model_v_feat} "
            f"--audio_feat_folder {model_a_feat} "
            f"--asr_feat_folder {model_s_feat} "
            f"--model_base {model_base} "
            f"--stage2 {model_stage2} "
            f"--stage3 {model_stage3} "
            f"--pretrain_mm_mlp_adapter {model_mm_mlp} "
            "--similarity_threshold 0.9"
        )
        code, out = run_command(cmd, env=env)
        append_log(f"$ CUDA_VISIBLE_DEVICES={gpu_id} {cmd}\n{out}")
        append_log(f"[2] 종료 코드: {code}")

    # 3. Summary 생성
    if run_summary:
        append_log("[3] Summary 생성 시작...")
        env = {"CUDA_VISIBLE_DEVICES": gpu_id}
        cmd = (
            "python src/eventtree/summary_llama3.py "
            f"--tree_path {tree_save_path} "
            f"--prompt_path {prompt_path} "
            f"--save_path {tree_save_path}"
        )
        code, out = run_command(cmd, env=env)
        append_log(f"$ CUDA_VISIBLE_DEVICES={gpu_id} {cmd}\n{out}")
        append_log(f"[3] 종료 코드: {code}")

    # 4. Postprocess
    if run_postprocess:
        append_log("[4] Postprocess 시작...")
        env = {"CUDA_VISIBLE_DEVICES": gpu_id}
        os.makedirs(os.path.dirname(post_save_dir), exist_ok=True)
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        cmd = (
            "python src/postprocess/postprocess.py "
            f'--input "{tree_save_path}" '
            f'--output-dir "{post_save_dir}" '
            f'--speech-json-dir "{speech_asr_dir}" '
            f'--not-json-dir "{debug_path}"'
        )
        code, out = run_command(cmd, env=env)
        append_log(f"$ CUDA_VISIBLE_DEVICES={gpu_id} {cmd}\n{out}")
        append_log(f"[4] 종료 코드: {code}")

    append_log("선택한 단계 실행이 모두 완료되었습니다.")
