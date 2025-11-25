import os
import subprocess
import json
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, env=None):
    full_env = os.environ.copy()
    # Ensure local Python packages (e.g., longvalellm) are importable
    src_path = os.path.join(BASE_DIR, "src")
    if full_env.get("PYTHONPATH"):
        full_env["PYTHONPATH"] = f"{src_path}{os.pathsep}{full_env['PYTHONPATH']}"
    else:
        full_env["PYTHONPATH"] = src_path
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
    "./data/annotation.json",
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
query_save_dir = st.sidebar.text_input(
    "QUERY_SAVE_DIR",
    "./outputs/query/example.json",
)
video_json = st.sidebar.text_input(
    "VIDEO_JSON_PATH",
    "./outputs/postprocess/olZPuJTwh0s.json",
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
    "./data/features_model/video_features",
)
model_a_feat = st.sidebar.text_input(
    "MODEL_A_FEAT",
    "./data/features_model/audio_features",
)
model_s_feat = st.sidebar.text_input(
    "MODEL_S_FEAT",
    "./data/features_model/speech_features",
)
speech_asr_dir = st.sidebar.text_input(
    "SPEECH_ASR_DIR",
    "./data/features_model/speech_asr",
)

model_base = st.sidebar.text_input(
    "MODEL_BASE",
    "./checkpoints/vicuna-7b-v1.5",
)
model_stage2 = st.sidebar.text_input(
    "MODEL_STAGE2",
    "./checkpoints/longvalellm-vicuna-v1-5-7b/longvale-vicuna-v1-5-7b-stage2-bp",
)
model_stage3 = st.sidebar.text_input(
    "MODEL_STAGE3",
    "./checkpoints/longvalellm-vicuna-v1-5-7b/longvale-vicuna-v1-5-7b-stage3-it",
)
model_mm_mlp = st.sidebar.text_input(
    "MODEL_MM_MLP",
    "./checkpoints/vtimellm_stage1_mm_projector.bin",
)
hf_token = st.sidebar.text_input(
    "HF_TOKEN",
    "",
)
mode = st.sidebar.text_input(
    "MODE_text_embed_or_heuristic",
    "text_embed",
)
query_str = st.sidebar.text_input(
    "QUERY_STR",
    "indoor market",
)


gpu_id = st.sidebar.text_input("GPU_ID (CUDA_VISIBLE_DEVICES)", "6")

st.sidebar.markdown("---")
st.sidebar.markdown("실행할 단계를 선택하세요:")
run_tree = st.sidebar.checkbox("1. Event Tree 생성 (tree.py)", value=False)
run_caption = st.sidebar.checkbox("2. Tree 캡셔닝 (caption_longvale.py)", value=False)
run_summary = st.sidebar.checkbox("3. Tree 요약 (summary_llama3.py)", value=False)
run_postprocess = st.sidebar.checkbox("4. Postprocess (postprocess.py)", value=True)
query_process = st.sidebar.checkbox("5. Query (search_queries.py)", value=True)

if "log_text" not in st.session_state:
    st.session_state.log_text = ""


log_area = st.empty()
tree_preview_area = st.empty()
caption_preview_area = st.empty()
summary_preview_area = st.empty()
post_preview_area = st.empty()


def append_log(text):
    if st.session_state.log_text:
        st.session_state.log_text += "\n" + text
    else:
        st.session_state.log_text = text
    log_area.text(st.session_state.log_text)


def show_tree_preview(path):
    if not os.path.isfile(path):
        tree_preview_area.info(f"Tree 파일을 찾을 수 없습니다: {path}")
        return
    with open(path, "r") as f:
        data = json.load(f)
    if not data:
        tree_preview_area.info("Tree 파일이 비어 있습니다.")
        return
    first_video_id = next(iter(data))
    tree_preview_area.markdown(f"**[1] Event Tree 미리보기 - video_id: {first_video_id}**")
    tree_preview_area.json(data[first_video_id])


def show_caption_preview(path):
    if not os.path.isfile(path):
        caption_preview_area.info(f"캡션이 포함된 Tree 파일이 없습니다: {path}")
        return
    with open(path, "r") as f:
        data = json.load(f)
    if not data:
        caption_preview_area.info("캡션이 포함된 Tree 데이터가 비어 있습니다.")
        return
    first_video_id = next(iter(data))
    caption_preview_area.markdown(f"**[2] Caption 미리보기 - video_id: {first_video_id}**")
    caption_preview_area.json(data[first_video_id])


def show_summary_preview(path):
    if not os.path.isfile(path):
        summary_preview_area.info(f"Summary가 저장된 Tree 파일이 없습니다: {path}")
        return
    with open(path, "r") as f:
        data = json.load(f)
    if not data:
        summary_preview_area.info("Summary 데이터가 비어 있습니다.")
        return
    first_video_id = next(iter(data))
    summary_preview_area.markdown(f"**[3] Summary 미리보기 - video_id: {first_video_id}**")
    summary_preview_area.json(data[first_video_id])


def show_postprocess_preview(output_dir):
    if not os.path.isdir(output_dir):
        post_preview_area.info(f"Postprocess 출력 디렉토리를 찾을 수 없습니다: {output_dir}")
        return
    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    if not json_files:
        post_preview_area.info("Postprocess 결과 JSON 파일이 없습니다.")
        return
    first_file = sorted(json_files)[0]
    first_path = os.path.join(output_dir, first_file)
    with open(first_path, "r") as f:
        data = json.load(f)
    post_preview_area.markdown(f"**[4] Postprocess 미리보기 - {first_file}**")
    post_preview_area.json(data)

def show_query_preview(output_dir):
    output_dir = os.path.dirname(output_dir)
    if not os.path.isdir(output_dir):
        post_preview_area.info(f"query 출력 디렉토리를 찾을 수 없습니다: {output_dir}")
        return
    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    if not json_files:
        post_preview_area.info("query 결과 JSON 파일이 없습니다.")
        return
    first_file = sorted(json_files)[0]
    first_path = os.path.join(output_dir, first_file)
    with open(first_path, "r") as f:
        data = json.load(f)
    post_preview_area.markdown(f"**[5] query 미리보기 - {first_file}**")
    post_preview_area.json(data)

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
        if code == 0:
            show_tree_preview(tree_save_path)

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
        if code == 0:
            show_caption_preview(tree_save_path)

    # 3. Summary 생성 (eventtree-post 환경)
    if run_summary:
        append_log("[3] Summary 생성 시작 (conda env: eventtree-post)...")
        cmd = (
            "bash -lc "
            "\"source ~/anaconda3/etc/profile.d/conda.sh && "
            "conda activate eventtree-post && "
            f"HF_TOKEN={hf_token} "
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            "python src/eventtree/summary_llama3.py "
            f"--tree_path {tree_save_path} "
            f"--prompt_path {prompt_path} "
            f"--save_path {tree_save_path}\""
        )
        code, out = run_command(cmd)
        append_log(f"$ {cmd}\n{out}")
        append_log(f"[3] 종료 코드: {code}")
        if code == 0:
            show_summary_preview(tree_save_path)

    # 4. Postprocess (eventtree-post 환경)
    if run_postprocess:
        append_log("[4] Postprocess 시작 (conda env: eventtree-post)...")
        os.makedirs(os.path.dirname(post_save_dir), exist_ok=True)
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        cmd = (
            "bash -lc "
            "\"source ~/anaconda3/etc/profile.d/conda.sh && "
            "conda activate eventtree-post && "
            f"HUGGINGFACE_HUB_TOKEN={hf_token} "
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            "python src/postprocess/postprocess.py "
            f'--input \\"{tree_save_path}\\" '
            f'--output-dir \\"{post_save_dir}\\" '
            f'--speech-json-dir \\"{speech_asr_dir}\\" '
            f'--not-json-dir \\"{debug_path}\\"\"'
        )
             
        code, out = run_command(cmd)
        append_log(f"$ {cmd}\n{out}")
        append_log(f"[4] 종료 코드: {code}")
        if code == 0:
            show_postprocess_preview(post_save_dir)
            
    # 5. Query (eventtree-post 환경)
    if query_process:
        append_log("[5] Query 시작 (conda env: eventtree-post)...")
        os.makedirs(os.path.dirname(query_save_dir), exist_ok=True)
        cmd = (
            "bash -lc "
            "\"source ~/anaconda3/etc/profile.d/conda.sh && "
            "conda activate eventtree-post && "
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            "python src/query/search_queries.py "
            f'--input \\"{video_json}\\" '
            f'--query \\"{query_str}\\" '
            f'--mode \\"{mode}\\" '
            f'--output \\"{query_save_dir}\\"\"'
        )
             
        code, out = run_command(cmd)
        append_log(f"$ {cmd}\n{out}")
        append_log(f"[5] 종료 코드: {code}")
        if code == 0:
            show_query_preview(query_save_dir)

    append_log("선택한 단계 실행이 모두 완료되었습니다.")
