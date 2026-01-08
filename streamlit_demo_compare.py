import os
import subprocess
import time
import base64
import json
import math
import html as html_module
import shlex
import csv
import hashlib
from typing import List
from string import Template

import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, env=None):
    full_env = os.environ.copy()
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


def has_any_files(folder: str, exts: List[str]) -> bool:
    if not os.path.isdir(folder):
        return False
    for name in os.listdir(folder):
        lower = name.lower()
        if any(lower.endswith(ext) for ext in exts):
            return True
    return False


def get_video_duration(video_path: str):
    """ffprobe를 사용해 비디오 duration(초)을 구합니다. 실패 시 None 반환."""
    if not os.path.isfile(video_path):
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except Exception:
        return None


def get_frame_image_path(video_path: str, video_id: str, time_sec: float) -> str | None:
    """지정된 시점의 첫 프레임을 jpg로 추출하고 경로를 반환합니다."""
    if not os.path.isfile(video_path):
        return None

    safe_time = max(time_sec, 0.0)
    frame_dir = os.path.join(BASE_DIR, "outputs", "frames", video_id)
    os.makedirs(frame_dir, exist_ok=True)
    frame_name = f"{int(round(safe_time))}.jpg"
    frame_path = os.path.join(frame_dir, frame_name)

    if os.path.isfile(frame_path):
        return frame_path

    cmd = (
        f'ffmpeg -y -loglevel error -ss {safe_time:.2f} -i "{video_path}" '
        f'-frames:v 1 -q:v 2 "{frame_path}"'
    )
    code, _ = run_command(cmd)
    if code != 0 or not os.path.isfile(frame_path):
        return None
    return frame_path


def image_file_to_base64(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_frame_b64_cached(video_path: str, video_id: str, time_sec: float) -> str | None:
    """지정 시점의 프레임을 추출해 base64로 캐싱하여 새로고침/재렌더링에서도 이미지가 유지되도록."""
    frame_path = get_frame_image_path(video_path, video_id, time_sec)
    if not frame_path:
        return None
    return image_file_to_base64(frame_path)


def get_video_duration(video_path: str):
    """ffprobe를 사용해 비디오 duration(초)을 구합니다. 실패 시 None 반환."""
    if not os.path.isfile(video_path):
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except Exception:
        return None


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


st.title("LongVALE Pipeline Demo (2-Step: Tree & Query)")

st.markdown(
    "이 데모는 LongVALE 전체 파이프라인을 **두 단계**로 실행합니다.\n\n"
    "1단계: Feature Extraction → Tree 생성 → Caption → Summary → Postprocess\n\n"
    "2단계: Query 검색 및 해당 구간 영상/JSON 확인"
)

st.sidebar.header("기본 설정")
hf_token = st.sidebar.text_input("HF_TOKEN", os.getenv("HF_TOKEN", ""))
gpu_id = st.sidebar.text_input("GPU_ID (CUDA_VISIBLE_DEVICES)", "0")
video_dir = st.sidebar.text_input(
    "VIDEO_DIR (원본 mp4 폴더)", "./data/raw_data/video"
)
audio_dir = st.sidebar.text_input(
    "AUDIO_DIR (원본 wav 폴더)", "./data/raw_data/audio"
)
checkpoint_dir = st.sidebar.text_input("CHECKPOINT_DIR", "./checkpoints")

# 고정 경로 설정 (좌측 경로 입력 대신 기본값 사용)
# annotation.json 없이, 선택한 비디오로부터 임시 annotation을 생성해 사용합니다.
prompt_path = "./data/prompt.json"
tree_save_dir = "./outputs/tree"
post_save_dir = "./outputs/postprocess"
debug_path = "./logs/debug.text"
query_base_dir = "./outputs/query"
query_recommendation_csv = os.path.join(BASE_DIR, "query_recommendations_wide.csv")

tree_v_feat = "./data/features_tree/video_features"
tree_a_feat = "./data/features_tree/audio_features"
tree_s_feat = "./data/features_tree/speech_features"

model_v_feat = "./data/features_model/video_features"
model_a_feat = "./data/features_model/audio_features"
model_s_feat = "./data/features_model/speech_features"
speech_asr_dir = "./data/features_model/speech_asr"

model_base = os.path.join(checkpoint_dir, "vicuna-7b-v1.5")
model_stage2 = os.path.join(
    checkpoint_dir,
    "longvalellm-vicuna-v1-5-7b",
    "longvale-vicuna-v1-5-7b-stage2-bp",
)
model_stage3 = os.path.join(
    checkpoint_dir,
    "longvalellm-vicuna-v1-5-7b",
    "longvale-vicuna-v1-5-7b-stage3-it",
)
model_mm_mlp = os.path.join(checkpoint_dir, "vtimellm_stage1_mm_projector.bin")

st.sidebar.markdown("---")
st.sidebar.markdown("Tree 파라미터")
caption_similarity_threshold = st.sidebar.slider(
    "Caption similarity_threshold (caption_longvale.py)", min_value=0.0, max_value=1.0, value=0.9, step=0.05
)
tree_merge_threshold = st.sidebar.slider(
    "Tree merge similarity_threshold (postprocess.py)", min_value=0.0, max_value=1.0, value=0.8, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("Query 파라미터 (기본값)")
default_query_str = st.sidebar.text_input("DEFAULT_QUERY_STR", "indoor market")
default_query_mode = st.sidebar.selectbox(
    "DEFAULT_QUERY_MODE", ["text_embed", "heuristic"], index=0
)
default_query_top_k = st.sidebar.number_input(
    "DEFAULT_QUERY_TOP_K", min_value=1, max_value=50, value=3, step=1
)
default_query_threshold = st.sidebar.slider(
    "DEFAULT_QUERY_THRESHOLD", min_value=0.0, max_value=1.0, value=0.2, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("TRACE Temporal Grounding")
trace_repo_root = st.sidebar.text_input(
    "TRACE_REPO_ROOT", "./TRACE"
)
default_trace_model_path = os.path.join(trace_repo_root, "trace-uni")
trace_model_path = st.sidebar.text_input(
    "TRACE_MODEL_PATH", default_trace_model_path
)
# trace_conda_env = st.sidebar.text_input("TRACE_CONDA_ENV", "trace")
TRACE_CONDA_ENV = "trace"
trace_device = st.sidebar.text_input("TRACE_DEVICE", "cuda:0")
trace_num_frames = st.sidebar.number_input(
    "TRACE_NUM_FRAMES", min_value=8, max_value=128, value=64, step=8
)
trace_max_new_tokens = st.sidebar.number_input(
    "TRACE_MAX_NEW_TOKENS", min_value=64, max_value=4096, value=512, step=64
)

if "log_text" not in st.session_state:
    st.session_state.log_text = ""

if "trace_cache" not in st.session_state:
    st.session_state.trace_cache = {}

log_area = st.empty()


def append_log(text):
    if st.session_state.log_text:
        st.session_state.log_text += "\n" + str(text)
    else:
        st.session_state.log_text = str(text)
    log_area.text(st.session_state.log_text)


def get_primary_gpu_index(gpu_text: str | None) -> int:
    if not gpu_text:
        return 0
    token = str(gpu_text).split(",")[0].strip()
    try:
        return int(token)
    except ValueError:
        return 0


def resolve_path(path: str) -> str:
    """상대 경로를 LongVALE_new 기준 절대 경로로 변환."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(BASE_DIR, path))


def ensure_tree_features(
    tree_v_feat_dir: str,
    tree_a_feat_dir: str,
    tree_s_feat_dir: str,
    video_id: str | None,
    annotation_path: str | None,
    video_dir_path: str,
    audio_dir_path: str,
    gpu_id_value: str,
    checkpoint_dir_path: str,
) -> int:
    """선택된 video_id에 대한 Tree feature 존재 여부 확인 및 필요 시 생성."""

    if not video_id:
        append_log("[0] Tree features 추출 불가: video_id가 없습니다.")
        return -1

    def _has_features_for_video(vid: str) -> bool:
        required_files = [
            os.path.join(tree_v_feat_dir, f"{vid}.npy"),
            os.path.join(tree_a_feat_dir, f"{vid}.npy"),
            os.path.join(tree_s_feat_dir, f"{vid}.npy"),
        ]
        return all(os.path.isfile(path) for path in required_files)

    if _has_features_for_video(video_id):
        append_log(f"[0] Tree features 이미 존재 ({video_id}) → 추출 스킵")
        return 0

    if not annotation_path or not os.path.isfile(annotation_path):
        append_log("[0] Tree features 추출 실패: 임시 annotation 파일이 없습니다.")
        return -1

    video_dir_abs = resolve_path(video_dir_path)
    audio_dir_abs = resolve_path(audio_dir_path)
    annotation_abs = resolve_path(annotation_path)
    tree_feat_root = resolve_path(os.path.dirname(tree_v_feat_dir))
    clip_ckpt = resolve_path(os.path.join(checkpoint_dir_path, "ViT-L-14.pt"))
    beats_ckpt = resolve_path(os.path.join(checkpoint_dir_path, "BEATs_iter3_plus_AS20K.pt"))
    whisper_ckpt = resolve_path(os.path.join(checkpoint_dir_path, "openai-whisper-large-v2"))
    gpu_index = get_primary_gpu_index(gpu_id_value)

    if not os.path.isdir(video_dir_abs):
        append_log(f"[0] Tree features 추출 실패: VIDEO_DIR 경로가 잘못되었습니다 ({video_dir_abs})")
        return -1
    if not os.path.isdir(audio_dir_abs):
        append_log(f"[0] Tree features 추출 실패: AUDIO_DIR 경로가 잘못되었습니다 ({audio_dir_abs})")
        return -1

    os.makedirs(tree_feat_root, exist_ok=True)

    cmd = (
        "python src/preprocess/tree_feature_extract.py "
        f"--data_path {shlex.quote(annotation_abs)} "
        f"--video_dir {shlex.quote(video_dir_abs)} "
        f"--audio_dir {shlex.quote(audio_dir_abs)} "
        f"--save_dir {shlex.quote(tree_feat_root)} "
        f"--clip_checkpoint {shlex.quote(clip_ckpt)} "
        f"--beats_checkpoint {shlex.quote(beats_ckpt)} "
        f"--whisper_checkpoint {shlex.quote(whisper_ckpt)} "
        "--extract_modality all "
        f"--gpu_id {gpu_index}"
    )

    append_log(f"[0] Tree features 추출 시작 (video_id={video_id})...")
    code, out = run_command(cmd)
    append_log(f"$ {cmd}\n{out}")
    append_log(f"[0] Tree features 종료 코드: {code}")
    return code


def ensure_model_features(
    model_v_feat_dir: str,
    model_a_feat_dir: str,
    model_s_feat_dir: str,
    speech_asr_dir_path: str,
    video_id: str | None,
    annotation_path: str | None,
    video_dir_path: str,
    audio_dir_path: str,
    gpu_id_value: str,
    checkpoint_dir_path: str,
) -> int:
    """선택된 video_id에 대한 모델 feature 존재 여부 확인 및 필요 시 생성."""

    if not video_id:
        append_log("[0] Model features 추출 불가: video_id가 없습니다.")
        return -1

    video_feat_path = os.path.join(model_v_feat_dir, f"{video_id}.npy")
    audio_feat_path = os.path.join(model_a_feat_dir, f"{video_id}.npy")
    speech_feat_path = os.path.join(model_s_feat_dir, f"{video_id}.npy")
    speech_asr_path = os.path.join(speech_asr_dir_path, f"{video_id}.json")

    missing_video = not os.path.isfile(video_feat_path)
    missing_audio = not os.path.isfile(audio_feat_path)
    missing_speech = not os.path.isfile(speech_feat_path)
    missing_asr = not os.path.isfile(speech_asr_path)

    if not any([missing_video, missing_audio, missing_speech, missing_asr]):
        append_log(f"[0] Model features 이미 존재 ({video_id}) → 추출 스킵")
        return 0

    if not annotation_path or not os.path.isfile(annotation_path):
        append_log("[0] Model features 추출 실패: 임시 annotation 파일이 없습니다.")
        return -1

    video_dir_abs = resolve_path(video_dir_path)
    audio_dir_abs = resolve_path(audio_dir_path)
    annotation_abs = resolve_path(annotation_path)
    clip_ckpt = resolve_path(os.path.join(checkpoint_dir_path, "ViT-L-14.pt"))
    beats_ckpt = resolve_path(os.path.join(checkpoint_dir_path, "BEATs_iter3_plus_AS20K.pt"))
    whisper_ckpt = resolve_path(os.path.join(checkpoint_dir_path, "openai-whisper-large-v2"))
    gpu_index = get_primary_gpu_index(gpu_id_value)

    if not os.path.isdir(video_dir_abs):
        append_log(f"[0] Model features 추출 실패: VIDEO_DIR 경로가 잘못되었습니다 ({video_dir_abs})")
        return -1
    if not os.path.isdir(audio_dir_abs):
        append_log(f"[0] Model features 추출 실패: AUDIO_DIR 경로가 잘못되었습니다 ({audio_dir_abs})")
        return -1

    commands: list[tuple[str, str]] = []
    if missing_video:
        os.makedirs(model_v_feat_dir, exist_ok=True)
        cmd = (
            "python src/preprocess/clip_feature_extract.py "
            f"--annotation {shlex.quote(annotation_abs)} "
            f"--video_dir {shlex.quote(video_dir_abs)} "
            f"--save_dir {shlex.quote(resolve_path(model_v_feat_dir))} "
            f"--checkpoint {shlex.quote(clip_ckpt)} "
            f"--gpu_id {gpu_index}"
        )
        commands.append(("Video", cmd))

    if missing_audio:
        os.makedirs(model_a_feat_dir, exist_ok=True)
        cmd = (
            "python src/preprocess/beats_feature_extract.py "
            f"--annotation {shlex.quote(annotation_abs)} "
            f"--audio_dir {shlex.quote(audio_dir_abs)} "
            f"--save_dir {shlex.quote(resolve_path(model_a_feat_dir))} "
            f"--checkpoint {shlex.quote(beats_ckpt)} "
            f"--gpu_id {gpu_index}"
        )
        commands.append(("Audio", cmd))

    if missing_speech:
        os.makedirs(model_s_feat_dir, exist_ok=True)
        cmd = (
            "python src/preprocess/whisper_feature_extract.py "
            f"--annotation {shlex.quote(annotation_abs)} "
            f"--audio_dir {shlex.quote(audio_dir_abs)} "
            f"--save_dir {shlex.quote(resolve_path(model_s_feat_dir))} "
            f"--checkpoint {shlex.quote(whisper_ckpt)} "
            f"--gpu_id {gpu_index}"
        )
        commands.append(("Speech", cmd))

    if missing_asr:
        os.makedirs(speech_asr_dir_path, exist_ok=True)
        cmd = (
            "python src/preprocess/whisper_speech_asr.py "
            f"--annotation {shlex.quote(annotation_abs)} "
            f"--audio_dir {shlex.quote(audio_dir_abs)} "
            f"--save_dir {shlex.quote(resolve_path(speech_asr_dir_path))} "
            f"--checkpoint {shlex.quote(whisper_ckpt)} "
            f"--gpu_id {gpu_index}"
        )
        commands.append(("Speech ASR", cmd))

    for desc, cmd in commands:
        append_log(f"[0] Model features {desc} 추출 시작 (video_id={video_id})...")
        code, out = run_command(cmd)
        append_log(f"$ {cmd}\n{out}")
        append_log(f"[0] Model features {desc} 종료 코드: {code}")
        if code != 0:
            return code

    return 0



def list_video_files(video_dir_path: str) -> List[str]:
    if not os.path.isdir(video_dir_path):
        return []
    files = []
    for name in os.listdir(video_dir_path):
        if name.lower().endswith(".mp4"):
            files.append(name)
    return sorted(files)


def list_video_ids_from_annotation(annotation_path: str) -> List[str]:
    if not os.path.isfile(annotation_path):
        return []
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    ids: List[str] = []
    if isinstance(data, dict):
        ids = list(data.keys())
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            vid = item.get("video_id") or item.get("id")
            if vid:
                ids.append(str(vid))
    return sorted(set(ids))


def list_postprocess_jsons(output_dir: str) -> List[str]:
    if not os.path.isdir(output_dir):
        return []
    files = []
    for name in os.listdir(output_dir):
        if name.lower().endswith(".json"):
            files.append(os.path.join(output_dir, name))
    return sorted(files)


@st.cache_data(show_spinner=False)
def load_query_recommendations(csv_path: str):
    """CSV에서 video_id별 good/bad Query 추천을 불러옵니다."""
    if not os.path.isfile(csv_path):
        return {}

    recommendations: dict[str, dict[str, list[tuple[str, str]]]] = {}
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = (row.get("video_id") or "").strip()
                if not video_id:
                    continue
                video_entry = recommendations.setdefault(
                    video_id, {"good": [], "bad": []}
                )

                def _add(field_name: str, quality: str):
                    value = (row.get(field_name) or "").strip()
                    if not value:
                        return
                    label_type = "sentence" if "sentence" in field_name else "word"
                    label = f"{quality.title()} ({label_type}) · {value}"
                    video_entry[quality].append((label, value))

                for idx in range(1, 3):
                    _add(f"sentence_good_{idx}", "good")
                    _add(f"sentence_bad_{idx}", "bad")
                    _add(f"word_good_{idx}", "good")
                    _add(f"word_bad_{idx}", "bad")
    except Exception:
        return {}

    return recommendations


query_recommendations_data = load_query_recommendations(query_recommendation_csv)


def flatten_tree_segments(tree_data: dict):
    segments = []

    def _dfs(node):
        post = node.get("postprocess") or {}
        result = post.get("result") or {}
        lod = result.get("LOD") or result.get("lod") or {}
        scene_topic = lod.get("scene_topic")

        children = node.get("children") or []
        is_leaf = len(children) == 0
        seg = {
            "node": node,
            "level": node.get("level", 0),
            "start": node.get("start_time", 0.0),
            "end": node.get("end_time", 0.0),
            "summary": scene_topic
            or node.get("summary")
            or node.get("caption")
            or "",
            "is_leaf": is_leaf,
        }
        segments.append(seg)
        for child in children:
            if isinstance(child, dict):
                _dfs(child)

    _dfs(tree_data)
    return segments


def collect_tree_segments_with_event(tree_data: dict):
    segments = []

    def _dfs(node):
        post = node.get("postprocess") or {}
        result = post.get("result") or {}
        lod = result.get("LOD") or result.get("lod") or {}
        scene_topic = lod.get("scene_topic")
        event_id = post.get("event_id")
        seg = {
            "node": node,
            "level": node.get("level", 0),
            "start": node.get("start_time", 0.0),
            "end": node.get("end_time", 0.0),
            "summary": scene_topic
            or node.get("summary")
            or node.get("caption")
            or "",
            "event_id": event_id,
        }
        segments.append(seg)
        for child in node.get("children") or []:
            if isinstance(child, dict):
                _dfs(child)

    _dfs(tree_data)
    return segments


def get_query_video_selection_data(post_dir: str, query_dir: str):
    post_files = list_postprocess_jsons(post_dir)
    query_files = list_postprocess_jsons(query_dir)
    query_file_map = {
        os.path.splitext(os.path.basename(path))[0]: path for path in query_files
    }
    postprocess_video_ids = {
        os.path.splitext(os.path.basename(path))[0] for path in post_files
    }
    available_video_ids = sorted(postprocess_video_ids.union(query_file_map.keys()))
    return post_files, query_file_map, available_video_ids


def render_query_inputs_for_video(
    video_id: str | None,
    prefix: str,
    existing_query_text: str | None = None,
):
    """현재 선택된 video_id 기준으로 Query 입력 UI를 구성하고 값을 반환."""
    video_query_recos = (
        query_recommendations_data.get(video_id) if video_id else None
    )
    has_recommendation = (
        video_query_recos
        and (video_query_recos.get("good") or video_query_recos.get("bad"))
    )
    query_str = existing_query_text or default_query_str

    if has_recommendation:
        available_quality_options = []
        if video_query_recos.get("good"):
            available_quality_options.append("Good")
        if video_query_recos.get("bad"):
            available_quality_options.append("Bad")

        if available_quality_options:
            query_quality = st.radio(
                "추천 Query 유형",
                available_quality_options,
                horizontal=True,
                key=f"{prefix}_query_quality_{video_id or 'none'}",
            )
            selected_options = list(
                video_query_recos.get(query_quality.lower()) or []
            )
            if existing_query_text:
                existing_label = f"저장된 Query 결과 · {existing_query_text}"
                if not any(existing_query_text == opt[1] for opt in selected_options):
                    selected_options.insert(0, (existing_label, existing_query_text))
            if selected_options:
                query_str = selected_options[
                    st.selectbox(
                        "Query 문자열",
                        range(len(selected_options)),
                        format_func=lambda i: selected_options[i][0],
                        key=f"{prefix}_query_value_{video_id or 'none'}_{query_quality}",
                    )
                ][1]
            else:
                st.info("선택한 유형의 추천 Query가 없어 직접 입력해야 합니다.")
                query_str = st.text_input(
                    "Query 문자열",
                    value=existing_query_text or default_query_str,
                    key=f"{prefix}_query_manual_{video_id or 'none'}",
                )
        else:
            query_str = st.text_input(
                "Query 문자열",
                value=existing_query_text or default_query_str,
                key=f"{prefix}_query_manual_{video_id or 'none'}",
            )
    else:
        query_str = st.text_input(
            "Query 문자열",
            value=existing_query_text or default_query_str,
            key=f"{prefix}_query_manual_{video_id or 'none'}",
        )

    query_mode = st.selectbox(
        "Query mode",
        ["text_embed", "heuristic"],
        index=["text_embed", "heuristic"].index(default_query_mode),
        key=f"{prefix}_query_mode",
    )
    query_top_k = st.number_input(
        "top_k",
        min_value=1,
        max_value=50,
        value=default_query_top_k,
        step=1,
        key=f"{prefix}_query_top_k",
    )
    query_threshold = st.slider(
        "similarity_threshold",
        min_value=0.0,
        max_value=1.0,
        value=default_query_threshold,
        step=0.05,
        key=f"{prefix}_query_threshold",
    )

    return {
        "query_str": query_str,
        "mode": query_mode,
        "top_k": int(query_top_k),
        "threshold": float(query_threshold),
    }


def run_query_command(
    video_id: str | None,
    query_str: str,
    query_mode: str,
    query_top_k: int,
    query_threshold: float,
    gpu_id_value: str,
    post_dir: str,
    query_dir: str,
    log_prefix: str = "[5]",
    success_message: str = "2단계 Query 검색이 완료되었습니다.",
):
    """선택된 video_id에 대해 LongVALE Query 파이프라인을 실행."""
    append_log(f"{log_prefix} Query 시작 (conda env: eventtree-post)...")

    if not video_id:
        append_log("video_id가 선택되지 않아 Query를 실행할 수 없습니다.")
        return -1, ""

    video_json_path = os.path.join(post_dir, f"{video_id}.json")
    if not os.path.isfile(video_json_path):
        append_log(
            f"{video_id}에 대한 Postprocess JSON을 찾을 수 없습니다. 1단계를 실행하세요."
        )
        return -1, ""

    query_save_path = os.path.join(query_dir, f"{video_id}.json")
    os.makedirs(os.path.dirname(query_save_path), exist_ok=True)
    cmd = (
        "bash -lc "
        "\"source ~/anaconda3/etc/profile.d/conda.sh && "
        "conda activate eventtree-post && "
        f"CUDA_VISIBLE_DEVICES={gpu_id_value} "
        "python src/query/search_queries.py "
        f'--input \\"{video_json_path}\\" '
        f'--query \\"{query_str}\\" '
        f'--mode \\"{query_mode}\\" '
        f'--top-k {query_top_k} '
        f'--threshold {query_threshold} '
        f'--output \\"{query_save_path}\\"\"'
    )
    step_start = time.time()
    code, out = run_command(cmd)
    append_log(f"$ {cmd}\n{out}")
    append_log(
        f"{log_prefix} 종료 코드: {code} (경과 {time.time() - step_start:.1f}초)"
    )
    if code == 0:
        append_log(success_message)
    return code, query_save_path


def extract_video_subclip(
    video_path: str,
    video_id: str,
    start_time: float | None,
    end_time: float | None,
    label: str,
) -> str | None:
    if not os.path.isfile(video_path):
        return None
    if start_time is None or end_time is None:
        return None
    try:
        start = float(start_time)
        end = float(end_time)
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None

    duration = get_video_duration(video_path)
    if duration is None:
        return None

    safe_start = max(0.0, min(start, duration))
    safe_end = max(safe_start + 0.1, min(end, duration))
    clip_dir = os.path.join(BASE_DIR, "outputs", "comparison_clips", video_id)
    os.makedirs(clip_dir, exist_ok=True)
    digest = hashlib.md5(
        f"{label}_{safe_start:.2f}_{safe_end:.2f}".encode("utf-8")
    ).hexdigest()[:10]
    clip_path = os.path.join(clip_dir, f"{label}_{digest}.mp4")
    if os.path.isfile(clip_path):
        return clip_path

    cmd = (
        f'ffmpeg -y -loglevel error -ss {safe_start:.3f} -to {safe_end:.3f} '
        f'-i "{video_path}" -c copy "{clip_path}"'
    )
    code, _ = run_command(cmd)
    if code != 0 or not os.path.isfile(clip_path):
        return None
    return clip_path


def run_trace_temporal_grounding(
    video_path: str,
    query_text: str,
    repo_root: str,
    model_path: str,
    device: str,
    max_new_tokens: int,
    num_frames: int,
    conda_env: str,
):
    repo_abs = os.path.abspath(repo_root)
    script_path = os.path.join(repo_abs, "scripts", "inference", "inference.py")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f"TRACE inference.py를 찾을 수 없습니다: {script_path}"
        )
    video_abs = os.path.abspath(video_path)
    if not os.path.isfile(video_abs):
        raise FileNotFoundError(f"비디오 파일이 없습니다: {video_abs}")
    model_abs = os.path.abspath(model_path) if model_path else None
    if not model_abs:
        model_abs = os.path.join(repo_abs, "trace-uni")

    run_id = hashlib.md5(
        f"{video_abs}|{query_text}|{model_abs}|{device}|{max_new_tokens}|{num_frames}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    output_dir = os.path.join(BASE_DIR, "outputs", "trace_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{run_id}.json")

    cmd = (
        "bash -lc "
        "\"source ~/anaconda3/etc/profile.d/conda.sh && "
        f"conda activate {shlex.quote(conda_env)} && "
        f"python {shlex.quote(script_path)} "
        f"--video_path {shlex.quote(video_abs)} "
        f"--query {shlex.quote(query_text)} "
        f"--model_path {shlex.quote(model_abs)} "
        f"--device {shlex.quote(device)} "
        f"--num_frames {int(num_frames)} "
        f"--max_new_tokens {int(max_new_tokens)} "
        f"--output {shlex.quote(output_path)}\""
    )
    code, out = run_command(cmd)
    if code != 0:
        raise RuntimeError(f"TRACE 추론 실패 (code={code}):\n{out}")
    if not os.path.isfile(output_path):
        raise RuntimeError("TRACE 추론 결과 파일을 찾을 수 없습니다.")
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


TREE_NODE_STYLE_BLOCK = """
<style>
:root {
  --tree-node-width: 192px;
  --tree-node-image-width: 144px;
  --tree-node-image-height: 80px;
}
.tree-container {
  position: relative;
  overflow: auto;
  padding: 20px 28px 24px 160px;
  background-color: #f8fafc;
  border: 2px solid #4a90e2;
  border-radius: 6px;
}
.tree-layout {
  position: relative;
  min-width: 100%;
  min-height: 380px;
}
.tree-level-labels {
  position: absolute;
  top: 0;
  left: -130px;
  width: 110px;
  pointer-events: none;
}
.tree-level-label {
  position: absolute;
  width: 100%;
  text-align: right;
  font-size: 11px;
  font-weight: 600;
  color: #4c5d78;
  letter-spacing: 0.1px;
}
.tree-edges-overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 1;
}
.tree-edges-overlay svg {
  width: 100%;
  height: 100%;
}
.tree-node {
  position: absolute;
  width: var(--tree-node-width);
  transform: translate(-50%, 0);
  font-size: 11px;
  z-index: 2;
}
.tree-node-level-badge {
  position: absolute;
  top: -18px;
  left: 0;
  font-size: 10px;
  color: #6b7a99;
  font-weight: 600;
}
.tree-node-box {
  position: relative;
  padding: 6px;
  width: var(--tree-node-width);
  background: #fff;
  border-radius: 10px;
  border: 1px solid #c6d0e3;
  box-shadow: 0 1px 4px rgba(15,32,77,0.15);
  overflow: visible;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.tree-node-image {
  width: var(--tree-node-image-width);
  height: var(--tree-node-image-height);
  object-fit: cover;
  border: 1px solid #c0c6d4;
  border-radius: 6px;
  display: block;
  margin: 0 auto;
}
.tree-node-tooltip {
  display: none;
  position: absolute;
  top: 0;
  left: 0;
  background: rgba(0,0,0,0.9);
  color: #fff;
  padding: 8px;
  z-index: 100;
  min-width: 240px;
  max-width: 440px;
  white-space: pre-wrap;
  font-size: 10px;
  border-radius: 4px;
  word-break: break-word;
}
.tree-node-placeholder {
  width: var(--tree-node-image-width);
  height: var(--tree-node-image-height);
  background: #eee;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 8px;
  color: #666;
  border-radius: 4px;
  border: 1px dashed #ccc;
}
.tree-node-box.selected {
  opacity: 1.0;
  filter: none;
  box-shadow: 0 0 6px rgba(255, 215, 0, 0.85);
}
.tree-node-box.dimmed {
  opacity: 0.25;
  filter: grayscale(0.9);
}
.tree-node-box:hover .tree-node-tooltip {
  display: block;
}
.tree-node-box::after {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: 6px;
  box-shadow: inset 0 0 0 0 rgba(0,0,0,0.1);
  pointer-events: none;
}
.tree-node-box:hover {
  transform: translateY(-2px);
}
</style>
<script>
(function() {
  const STORAGE_KEY = "longvale_tree_scroll";
  function attach() {
    const container = document.querySelector(".tree-container");
    if (!container) return;
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved !== null) {
      const v = parseInt(saved, 10);
      if (!Number.isNaN(v)) {
        container.scrollTop = v;
      }
    }
    container.addEventListener("scroll", function() {
      try {
        window.localStorage.setItem(STORAGE_KEY, String(container.scrollTop));
      } catch (e) {
        // ignore
      }
    }, { passive: true });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attach);
  } else {
    attach();
  }
})();
</script>
"""


def ensure_tree_node_style():
    """전체 Tree 노드 시각화를 위한 CSS 주입."""
    st.markdown(TREE_NODE_STYLE_BLOCK, unsafe_allow_html=True)
    # 매 rerun 시에도 스타일을 다시 주입해 트리가 깨지지 않도록 함.


def render_tree_structure(
    tree_data: dict,
    video_id: str,
    video_dir_path: str,
    selected_event_id=None,
    selected_time_range: tuple[float | None, float | None] | None = None,
):
    """
    전체 Tree를 level(세로 축) 기준으로 썸네일 트리로 시각화.
    각 노드는 첫 프레임 이미지이며 hover 시 정보를 보여줍니다.
    Query 매치 노드는 강조 표시합니다.
    """
    video_path = os.path.join(video_dir_path, f"{video_id}.mp4")
    if not os.path.isfile(video_path):
        st.info(f"비디오 파일 없음: {video_path}")
        return

    video_duration = get_video_duration(video_path)

    ensure_tree_node_style()

    def _safe_level(value, default):
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    nodes_by_id: dict[str, dict] = {}
    edges: list[tuple[str, str]] = []
    min_level = math.inf
    max_level = -math.inf

    def _build_node(node: dict, fallback_level: int, node_id: str, parent_id: str | None):
        nonlocal min_level, max_level
        level_val = _safe_level(node.get("level"), fallback_level)
        min_level = min(min_level, level_val)
        max_level = max(max_level, level_val)

        start = float(node.get("start_time", 0.0) or 0.0)
        end = float(node.get("end_time", 0.0) or 0.0)
        post = node.get("postprocess") or {}
        result = post.get("result") or {}
        lod = result.get("LOD") or result.get("lod") or {}
        tags = result.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        scene_topic = lod.get("scene_topic") if isinstance(lod, dict) else None
        modalities = lod.get("modalities") if isinstance(lod, dict) else {}
        modalities = modalities if isinstance(modalities, dict) else {}
        node_event_id = result.get("event_id") or post.get("event_id")
        scene_topic_text = (
            scene_topic
            if isinstance(scene_topic, str)
            else (str(scene_topic) if scene_topic else None)
        )
        label_text = (
            scene_topic_text
            or node.get("summary")
            or node.get("caption")
            or "-"
        )
        if not isinstance(label_text, str):
            label_text = str(label_text)

        # 구간의 중앙 프레임 사용
        if end > start:
            frame_time = (start + end) / 2.0
        else:
            frame_time = start
        if video_duration is not None:
            frame_time = max(0.0, min(frame_time, max(video_duration - 0.1, 0.0)))

        frame_b64 = get_frame_b64_cached(video_path, video_id, frame_time)
        tags_text = ", ".join(str(t) for t in tags if t) if tags else "-"
        tooltip_parts = [
            f"<strong>Level</strong>: L{level_val}",
            f"<strong>구간</strong>: {start:.1f}–{end:.1f}s",
            f"<strong>scene_topic</strong>: {html_module.escape(scene_topic_text or '-')}",
            f"<strong>tags</strong>: {html_module.escape(tags_text)}",
        ]
        if node_event_id is not None:
            tooltip_parts.append(f"<strong>event_id</strong>: {html_module.escape(str(node_event_id))}")
        if isinstance(modalities, dict):
            for key in ["visual", "audio", "speech"]:
                if modalities.get(key):
                    tooltip_parts.append(
                        f"<strong>{key}</strong>: {html_module.escape(str(modalities[key]))}"
                    )
        tooltip_html = "<br/>".join(tooltip_parts)

        children = node.get("children") or []
        filtered_children = [c for c in children if isinstance(c, dict)]
        filtered_children.sort(key=lambda c: c.get("start_time", 0.0) or 0.0)

        node_info = {
            "id": node_id,
            "parent_id": parent_id,
            "level": level_val,
            "start": start,
            "end": end,
            "event_id": node_event_id,
            "label": label_text,
            "tooltip": tooltip_html,
            "frame_b64": frame_b64,
            "children": [],
        }
        nodes_by_id[node_id] = node_info

        for idx, child in enumerate(filtered_children):
            child_id = f"{node_id}.{idx}"
            node_info["children"].append(child_id)
            edges.append((node_id, child_id))
            _build_node(child, level_val + 1, child_id, node_id)

    root_level = _safe_level(tree_data.get("level"), 0)
    _build_node(tree_data, root_level, "0", None)

    if not nodes_by_id:
        st.info("Tree 노드가 없습니다.")
        return

    def _assign_positions(node_id: str):
        node = nodes_by_id[node_id]
        children = node.get("children", [])
        if not children:
            idx = _assign_positions.counter
            node["x_index"] = idx
            _assign_positions.counter += 1
            return idx
        positions = []
        for child_id in children:
            positions.append(_assign_positions(child_id))
        node["x_index"] = sum(positions) / len(positions)
        return node["x_index"]

    _assign_positions.counter = 0
    _assign_positions("0")
    max_x_index = max((node.get("x_index", 0.0) for node in nodes_by_id.values()), default=0.0)

    horizontal_spacing = 220
    vertical_spacing = 160
    padding_x = 90
    padding_y = 60
    level_label_width = 90
    level_span = max(1, int(max_level - min_level + 1)) if math.isfinite(min_level) else 1

    layout_width = int(
        padding_x * 2 + level_label_width + (max_x_index + 1) * horizontal_spacing
    )
    layout_height = int(padding_y * 2 + level_span * vertical_spacing)
    container_height = max(420, min(900, layout_height + 80))
    iframe_height = min(1200, container_height + 150)

    selected_event_str = str(selected_event_id) if selected_event_id is not None else None
    highlight_node_id: str | None = None
    if selected_event_str is not None:
        for node_id, node in nodes_by_id.items():
            node_event = node.get("event_id")
            if node_event is not None and str(node_event) == selected_event_str:
                highlight_node_id = node_id
                break

    def _best_range_match():
        sel_range = selected_time_range
        if not sel_range:
            return None
        sel_start, sel_end = sel_range
        if sel_start is None or sel_end is None:
            return None
        sel_start = float(sel_start)
        sel_end = float(sel_end)
        if sel_end <= sel_start:
            return None
        best_score = 0.0
        best_id = None
        for node_id, node in nodes_by_id.items():
            node_start = float(node.get("start", 0.0) or 0.0)
            node_end = float(node.get("end", 0.0) or 0.0)
            if node_end <= node_start:
                continue
            overlap = max(0.0, min(node_end, sel_end) - max(node_start, sel_start))
            if overlap <= 0.0:
                continue
            union = (sel_end - sel_start) + (node_end - node_start) - overlap
            if union <= 0.0:
                continue
            iou = overlap / union
            if iou > best_score:
                best_score = iou
                best_id = node_id
        return best_id if best_score > 0 else None

    if highlight_node_id is None:
        highlight_node_id = _best_range_match()
    nodes_html_parts: list[str] = []
    for node_info in sorted(
        nodes_by_id.values(), key=lambda n: (n["level"], n.get("x_index", 0.0))
    ):
        level_val = node_info["level"]
        level_offset = level_val - min_level if math.isfinite(min_level) else level_val
        top_px = padding_y + level_offset * vertical_spacing
        left_px = (
            padding_x
            + level_label_width
            + node_info.get("x_index", 0.0) * horizontal_spacing
        )
        label_text = html_module.escape((node_info["label"] or "-")[:60])
        range_text = f"{node_info['start']:.1f}–{node_info['end']:.1f}s"
        tooltip_html = node_info["tooltip"]
        frame_b64 = node_info.get("frame_b64")
        if frame_b64:
            img_html = f'<img src="data:image/jpeg;base64,{frame_b64}" class="tree-node-image" />'
        else:
            img_html = '<div class="tree-node-placeholder">No Frame</div>'

        is_highlight = highlight_node_id is not None and node_info["id"] == highlight_node_id
        classes = ["tree-node-box"]
        if highlight_node_id is not None and not is_highlight:
            classes.append("dimmed")
        if is_highlight:
            classes.append("selected")

        node_html = f"""
<div class="tree-node" data-node-id="{node_info['id']}" style="top:{top_px}px; left:{left_px}px;">
  <div class="tree-node-level-badge">L{max(level_offset, 0)}</div>
  <div class="{' '.join(classes)}">
    {img_html}
    <div class="tree-node-tooltip">{tooltip_html}</div>
  </div>
</div>
        """
        nodes_html_parts.append(node_html)

    level_labels_html: list[str] = []
    if math.isfinite(min_level) and math.isfinite(max_level):
        for offset, level_val in enumerate(range(int(min_level), int(max_level) + 1)):
            top_px = padding_y + (level_val - min_level) * vertical_spacing
            level_labels_html.append(
                f'<div class="tree-level-label" style="top:{top_px}px;">L{offset}</div>'
            )

    edges_json = json.dumps(edges)
    container_id = f"tree-container-{video_id}-{int(time.time()*1000)}"
    html_template = Template(
        """
$style_block
<div id="$cid" class="tree-container" style="height: ${container_height}px;">
  <div class="tree-layout" style="width: ${layout_width}px; height: ${layout_height}px;">
    <div class="tree-level-labels">$level_labels</div>
    <div class="tree-edges-overlay"><svg></svg></div>
    $nodes
  </div>
</div>
<script>
(() => {
  const edges = $edges;
  const container = document.getElementById("$cid");
  if (!container) return;
  const layout = container.querySelector('.tree-layout');
  if (!layout) return;
  const svg = layout.querySelector('.tree-edges-overlay svg');
  if (!svg) return;

  const draw = () => {
    const layoutRect = layout.getBoundingClientRect();
    const nodes = {};
    layout.querySelectorAll('[data-node-id]').forEach(el => {
      const rect = el.getBoundingClientRect();
      nodes[el.dataset.nodeId] = {
        x: rect.left + rect.width / 2 - layoutRect.left,
        yTop: rect.top - layoutRect.top,
        yBottom: rect.bottom - layoutRect.top
      };
    });
    svg.setAttribute('width', layoutRect.width);
    svg.setAttribute('height', layoutRect.height);
    svg.setAttribute('viewBox', '0 0 ' + layoutRect.width + ' ' + layoutRect.height);
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }
    edges.forEach(([p, c]) => {
      const parent = nodes[p];
      const child = nodes[c];
      if (!parent || !child) return;
      const midY = (parent.yBottom + child.yTop) / 2;
      const d = 'M ' + parent.x + ' ' + parent.yBottom +
                ' C ' + parent.x + ' ' + midY + ', ' +
                child.x + ' ' + midY + ', ' +
                child.x + ' ' + child.yTop;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', d);
      path.setAttribute('stroke', '#93a6c7');
      path.setAttribute('stroke-width', '1');
      path.setAttribute('fill', 'none');
      svg.appendChild(path);
    });
  };

  requestAnimationFrame(draw);
  const ro = new ResizeObserver(draw);
  ro.observe(layout);
  container.addEventListener('scroll', draw, { passive: true });
})();
</script>
"""
    )

    html_output = html_template.substitute(
        cid=container_id,
        nodes="".join(nodes_html_parts),
        edges=edges_json,
        container_height=int(container_height),
        layout_width=int(layout_width),
        layout_height=int(layout_height),
        level_labels="".join(level_labels_html),
        style_block=TREE_NODE_STYLE_BLOCK,
    )

    components.html(html_output, height=iframe_height, scrolling=True)


def show_tree_step_visual(video_json_path: str, video_dir_path: str, container):
    if not os.path.isfile(video_json_path):
        with container:
            st.info(f"Postprocess JSON을 찾을 수 없습니다: {video_json_path}")
        return

    with open(video_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_id = data.get("video_id")
    tree_data = data.get("tree")
    if not video_id or not tree_data:
        with container:
            st.info("video_id 또는 tree 정보가 없습니다.")
        return

    video_path = os.path.join(video_dir_path, f"{video_id}.mp4")
    segments = flatten_tree_segments(tree_data)
    leaf_segments = [s for s in segments if s["is_leaf"]]
    if not leaf_segments:
        with container:
            st.info("Tree leaf segment가 없습니다.")
        return

    with container:
        options = {
            f"L{s['level']} | {s['start']:.1f}–{s['end']:.1f}s | {s['summary'][:40]}": idx
            for idx, s in enumerate(leaf_segments)
        }
        label = st.selectbox("1단계 구간 선택 (leaf)", list(options.keys()))
        selected = leaf_segments[options[label]]

        col_video, col_info = st.columns([2, 3])
        with col_video:
            if os.path.isfile(video_path):
                st.video(video_path, start_time=int(selected["start"]))
            else:
                st.info(f"비디오 파일 없음: {video_path}")

        with col_info:
            st.markdown("**선택 노드 Level 및 Caption**")
            st.write(f"Level: L{selected['level']}")
            st.write(selected["summary"])

            st.markdown("**선택 구간 Node JSON**")
            st.json(selected["node"])

        st.markdown("**전체 Tree 구조 (노드 뷰)**")
        render_tree_structure(tree_data, video_id, video_dir_path)


def show_query_step_visual(video_json_path: str, query_json_path: str, video_dir_path: str, container):
    if not os.path.isfile(video_json_path):
        with container:
            st.info(f"Postprocess JSON을 찾을 수 없습니다: {video_json_path}")
        return
    if not os.path.isfile(query_json_path):
        with container:
            st.info(f"Query JSON을 찾을 수 없습니다: {query_json_path}")
        return

    with open(video_json_path, "r", encoding="utf-8") as f:
        tree_data_all = json.load(f)
    with open(query_json_path, "r", encoding="utf-8") as f:
        q = json.load(f)

    video_id = tree_data_all.get("video_id")
    tree_data = tree_data_all.get("tree")
    if not video_id or not tree_data:
        with container:
            st.info("video_id 또는 tree 정보가 없습니다.")
        return

    video_path = os.path.join(video_dir_path, f"{video_id}.mp4")

    matches = q.get("matches", [])
    if not matches:
        with container:
            st.info("Query 매치 결과가 없습니다.")
        return

    segments = collect_tree_segments_with_event(tree_data)

    with container:
        options = {
            f"[score={m.get('score',0):.3f}] {m.get('start_time',0):.1f}–{m.get('end_time',0):.1f}s | {(m.get('scene_topic') or m.get('summary',''))[:40]}": idx
            for idx, m in enumerate(matches)
        }
        label = st.selectbox("Query 결과 구간 선택", list(options.keys()))
        selected = matches[options[label]]
        event_id = selected.get("event_id")

        # 선택된 event_id에 해당하는 Tree 노드 level 찾기
        selected_level = None
        if event_id is not None:
            for seg in segments:
                if seg["event_id"] == event_id:
                    selected_level = seg["level"]
                    break

        col_video, col_json = st.columns([2, 3])
        with col_video:
            if os.path.isfile(video_path):
                st.video(video_path, start_time=int(selected.get("start_time", 0)))
            else:
                st.info(f"비디오 파일 없음: {video_path}")

        with col_json:
            st.markdown("**선택된 Query match 정보**")
            if selected_level is not None:
                st.write(f"Level: L{selected_level}")
            st.write(selected.get("summary", ""))

            st.markdown("**선택된 Query match JSON**")
            st.json(selected)

        st.markdown("**전체 Tree 구조 (Query 매치 하이라이트)**")
        selected_range = (
            float(selected.get("start_time", 0.0)) if selected.get("start_time") is not None else None,
            float(selected.get("end_time", 0.0)) if selected.get("end_time") is not None else None,
        )
        render_tree_structure(
            tree_data,
            video_id,
            video_dir_path,
            selected_event_id=event_id,
            selected_time_range=selected_range,
        )


tab_video, tab_tree, tab_query, tab_compare = st.tabs(
    [
        "0. 비디오 선택/미리보기",
        "1. Tree 생성 (Feature 포함)",
        "2. Query 검색",
        "3. Query 비교 (TRACE)"
    ]
)

with tab_video:
    st.markdown(
        "### 0단계: 비디오 선택 및 미리보기\n"
        "- VIDEO_DIR 안의 mp4 파일 중에서 하나를 선택해 미리 재생합니다.\n"
        "- 선택한 video_id는 1단계/2단계에서 기본값으로 사용됩니다."
    )
    available_videos = list_video_files(video_dir)
    if available_videos:
        selected_video_name = st.selectbox(
            "비디오 선택 (mp4 파일명)",
            available_videos,
            key="video_select_tab0",
        )
        if selected_video_name:
            selected_video_id = os.path.splitext(selected_video_name)[0]
            st.session_state["selected_video_id"] = selected_video_id
            video_path = os.path.join(video_dir, selected_video_name)
            if os.path.isfile(video_path):
                st.video(video_path)
            else:
                st.info(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    else:
        st.info(f"VIDEO_DIR 경로에 mp4 파일이 없습니다: {video_dir}")


with tab_tree:
    st.markdown(
        "### 1단계: Feature Extraction + Tree 전체 파이프라인\n"
        "- Feature가 이미 존재하면 재계산하지 않습니다.\n"
        "- 순서: features_tree.sh → features_longvale.sh → tree.py → caption_longvale.py → summary_llama3.py → postprocess.py"
    )

    st.markdown("#### 현재 선택된 비디오 ID")
    selected_video_id = st.session_state.get("selected_video_id")
    if selected_video_id:
        st.info(f"현재 선택된 비디오 ID: {selected_video_id}")
    else:
        st.warning("0단계에서 비디오를 먼저 선택해야 1단계를 실행할 수 있습니다.")

    st.markdown("#### 1단계 실행")
    # 1단계 실행 플로우에서 사용하는 상태 코드 기본값
    code = 0
    run_clicked = st.button("1단계 실행")
    if run_clicked:
        st.session_state.log_text = ""
        log_area.text("")

        code = 0

        # 1) 선택된 비디오 ID 확인
        selected_video_id = st.session_state.get("selected_video_id")
        if not selected_video_id:
            append_log("0단계에서 비디오를 먼저 선택하세요.")
            code = -1
        else:
            # 2) 비디오 duration 계산 후, 임시 annotation JSON 생성
            video_path = os.path.join(video_dir, f"{selected_video_id}.mp4")
            if not os.path.isfile(video_path):
                append_log(f"선택된 비디오 파일을 찾을 수 없습니다: {video_path}")
                code = -1
            else:
                duration = get_video_duration(video_path)
                if duration is None:
                    append_log(f"비디오 duration을 계산하지 못했습니다: {video_path}")
                    code = -1
                else:
                    tmp_anno_dir = os.path.join(BASE_DIR, "outputs", "tmp_annotation")
                    os.makedirs(tmp_anno_dir, exist_ok=True)
                    data_path = os.path.join(
                        tmp_anno_dir, f"{selected_video_id}_annotation.json"
                    )
                    with open(data_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {selected_video_id: {"duration": duration}},
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    append_log(
                        f"임시 annotation 생성: {data_path} (duration={duration:.2f}s)"
                    )

        if run_clicked and code == 0:
            step_start = time.time()
            code = ensure_tree_features(
                tree_v_feat,
                tree_a_feat,
                tree_s_feat,
                video_id=selected_video_id,
                annotation_path=data_path,
                video_dir_path=video_dir,
                audio_dir_path=audio_dir,
                gpu_id_value=gpu_id,
                checkpoint_dir_path=checkpoint_dir,
            )
            if code != 0:
                append_log(
                    f"Tree feature 추출 실패로 1단계를 중단합니다. (경과 {time.time() - step_start:.1f}초)"
                )

        if run_clicked and code == 0:
            step_start = time.time()
            code = ensure_model_features(
                model_v_feat,
                model_a_feat,
                model_s_feat,
                speech_asr_dir,
                video_id=selected_video_id,
                annotation_path=data_path,
                video_dir_path=video_dir,
                audio_dir_path=audio_dir,
                gpu_id_value=gpu_id,
                checkpoint_dir_path=checkpoint_dir,
            )
            if code != 0:
                append_log(
                    f"Model feature 추출 실패로 1단계를 중단합니다. (경과 {time.time() - step_start:.1f}초)"
                )

        if run_clicked and code == 0:
            if code == 0:
                effective_tree_path = os.path.join(
                    tree_save_dir, f"{selected_video_id}.json"
                )
                if not os.path.isfile(effective_tree_path):
                    append_log(
                        f"Tree 결과 파일을 찾을 수 없습니다: {effective_tree_path}"
                    )
                    code = -1
            append_log("[1] Event Tree 생성 시작...")
            step_start = time.time()
            cmd = (
                "python src/eventtree/tree/tree.py "
                f"--data_path {data_path} "
                f"--video_feat_folder {tree_v_feat} "
                f"--audio_feat_folder {tree_a_feat} "
                f"--speech_feat_folder {tree_s_feat} "
                f"--save_path {effective_tree_path} "
            )

            code, out = run_command(cmd)
            append_log(f"$ {cmd}\n{out}")
            append_log(
                f"[1] 종료 코드: {code} (경과 {time.time() - step_start:.1f}초)"
            )

        if run_clicked and code == 0:
            append_log("[2] Caption 생성 시작...")
            env = {"CUDA_VISIBLE_DEVICES": gpu_id}
            resolved_model_base = resolve_path(model_base)
            resolved_model_stage2 = resolve_path(model_stage2)
            resolved_model_stage3 = resolve_path(model_stage3)
            resolved_model_mm_mlp = resolve_path(model_mm_mlp)
            cmd = (
                "python src/eventtree/caption_longvale.py "
                f"--tree_path {effective_tree_path} "
                f"--prompt_path {prompt_path} "
                f"--save_path {effective_tree_path} "
                f"--video_feat_folder {model_v_feat} "
                f"--audio_feat_folder {model_a_feat} "
                f"--asr_feat_folder {model_s_feat} "
                f"--model_base {resolved_model_base} "
                f"--stage2 {resolved_model_stage2} "
                f"--stage3 {resolved_model_stage3} "
                f"--pretrain_mm_mlp_adapter {resolved_model_mm_mlp} "
                f"--similarity_threshold {caption_similarity_threshold}"
            )
            step_start = time.time()
            code, out = run_command(cmd, env=env)
            append_log(f"$ CUDA_VISIBLE_DEVICES={gpu_id} {cmd}\n{out}")
            append_log(
                f"[2] 종료 코드: {code} (경과 {time.time() - step_start:.1f}초)"
            )

        if run_clicked and code == 0:
            append_log("[3] Summary 생성 시작 (conda env: eventtree-post)...")
            cmd = (
                "bash -lc "
                "\"source ~/anaconda3/etc/profile.d/conda.sh && "
                "conda activate eventtree-post && "
                f"HF_TOKEN={hf_token} "
                f"CUDA_VISIBLE_DEVICES={gpu_id} "
                "python src/eventtree/summary_llama3.py "
                f"--tree_path {effective_tree_path} "
                f"--prompt_path {prompt_path} "
                f"--save_path {effective_tree_path}\""
            )
            step_start = time.time()
            code, out = run_command(cmd)
            append_log(f"$ {cmd}\n{out}")
            append_log(
                f"[3] 종료 코드: {code} (경과 {time.time() - step_start:.1f}초)"
            )

        if run_clicked and code == 0:
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
                f'--input \\"{effective_tree_path}\\" '
                f'--output-dir \\"{post_save_dir}\\" '
                f'--speech-json-dir \\"{speech_asr_dir}\\" '
                f'--merge-threshold {tree_merge_threshold} '
                f'--not-json-dir \\"{debug_path}\\"\"'
            )
            step_start = time.time()
            code, out = run_command(cmd)
            append_log(f"$ {cmd}\n{out}")
            append_log(
                f"[4] 종료 코드: {code} (경과 {time.time() - step_start:.1f}초)"
            )

        if run_clicked and code == 0:
            append_log("1단계 전체 파이프라인이 완료되었습니다.")

    st.markdown("#### 시각화할 Postprocess JSON (video_id) 선택")
    # 1단계 실행 여부와 관계없이, 현재 존재하는 Postprocess 결과를 항상 시각화
    json_files = list_postprocess_jsons(post_save_dir)
    if json_files:
        labels = [os.path.basename(p) for p in json_files]

        # Streamlit selectbox에서 기본 선택을 사용자가 선택한 video_id로 맞추기
        default_index = 0
        current_video_id = st.session_state.get("selected_video_id")
        if current_video_id is not None:
            for i, lbl in enumerate(labels):
                if os.path.splitext(lbl)[0] == str(current_video_id):
                    default_index = i
                    break

        selected_index = st.selectbox(
            "시각화할 Postprocess JSON (video_id) 선택",
            range(len(json_files)),
            index=default_index,
            format_func=lambda i: labels[i],
        )
        selected_video_json = json_files[selected_index]

        tree_vis_container = st.container()
        show_tree_step_visual(selected_video_json, video_dir, tree_vis_container)
    else:
        st.info("Postprocess 결과 JSON이 없습니다. 1단계를 먼저 실행하세요.")


with tab_query:
    st.markdown(
        "### 2단계: Query 검색\n"
        "- 1단계 Postprocess 결과(JSON)와 Query 결과(JSON)를 이용해 시각화합니다."
    )

    post_json_files = list_postprocess_jsons(post_save_dir)
    if not post_json_files:
        st.info("Postprocess 결과 JSON이 없습니다. 1단계를 먼저 실행하세요.")
    else:
        post_video_ids = [
            os.path.splitext(os.path.basename(path))[0] for path in post_json_files
        ]
        post_video_ids.sort()
        label_map = {}
        for vid in post_video_ids:
            query_path = os.path.join(query_base_dir, f"{vid}.json")
            exists = os.path.isfile(query_path)
            suffix = "" if exists else " (Query 결과 없음)"
            label_map[vid] = f"{vid}.json{suffix}"

        default_index = 0
        current_video_id = st.session_state.get("selected_video_id")
        if current_video_id in post_video_ids:
            default_index = post_video_ids.index(current_video_id)

        selected_q_index = st.selectbox(
            "시각화할 Query JSON (video_id) 선택",
            range(len(post_video_ids)),
            index=default_index,
            format_func=lambda i: label_map[post_video_ids[i]],
        )
        selected_query_video_id = post_video_ids[selected_q_index]
        selected_query_json_path = os.path.join(
            query_base_dir, f"{selected_query_video_id}.json"
        )

        existing_query_text = None
        if os.path.isfile(selected_query_json_path):
            try:
                with open(selected_query_json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                existing_query_text = (payload.get("query") or "").strip()
                if not existing_query_text:
                    existing_query_text = None
            except Exception:
                existing_query_text = None

        query_inputs = render_query_inputs_for_video(
            selected_query_video_id,
            "query_tab",
            existing_query_text=existing_query_text,
        )
        query_str_input = query_inputs["query_str"]
        query_str = query_str_input.strip()
        query_mode = query_inputs["mode"]
        query_top_k = query_inputs["top_k"]
        query_threshold = query_inputs["threshold"]

        query_vis_container = st.container()

        run_query_clicked = st.button("2단계 실행 (Query 검색)", key="query_run_button")
        if run_query_clicked:
            st.session_state.log_text = ""
            log_area.text("")
            run_query_command(
                selected_query_video_id,
                query_str_input,
                query_mode,
                query_top_k,
                query_threshold,
                gpu_id,
                post_save_dir,
                query_base_dir,
                log_prefix="[5]",
                success_message="2단계 Query 검색이 완료되었습니다.",
            )

        video_json_path = os.path.join(
            post_save_dir, f"{selected_query_video_id}.json"
        )
        if not os.path.isfile(video_json_path):
            with query_vis_container:
                st.info(
                    f"{selected_query_video_id}에 대한 Postprocess JSON을 찾을 수 없습니다. 1단계를 실행하세요."
                )
        elif not os.path.isfile(selected_query_json_path):
            with query_vis_container:
                st.info(
                    f"{selected_query_video_id}에 대한 Query 결과 JSON이 없습니다. 2단계를 실행하세요."
                )
        else:
            try:
                with open(selected_query_json_path, "r", encoding="utf-8") as f:
                    query_payload = json.load(f)
            except Exception:
                query_payload = {}
            payload_query_text = (query_payload.get("query") or "").strip()
            if not payload_query_text or payload_query_text != query_str:
                with query_vis_container:
                    st.info(
                        f"선택한 Query 문자열({query_str})에 대한 결과가 없습니다. 2단계를 실행하세요."
                    )
            else:
                show_query_step_visual(
                    video_json_path,
                    selected_query_json_path,
                    video_dir,
                    query_vis_container,
                )


with tab_compare:
    st.markdown(
        "### 3단계: Query 결과 비교 (LongVALE vs TRACE)\n"
        "- Postprocess가 완료된 video_id에 한해 LongVALE 2단계와 TRACE 결과를 동시에 확인합니다.\n"
        "- 비교 실행 버튼을 누르면 선택한 설정대로 LongVALE Query와 TRACE 추론을 연속 실행합니다."
    )

    post_json_files = list_postprocess_jsons(post_save_dir)
    if not post_json_files:
        st.info("Postprocess 결과 JSON이 없습니다. 1단계를 먼저 실행하세요.")
    else:
        post_video_ids = [
            os.path.splitext(os.path.basename(path))[0] for path in post_json_files
        ]
        post_video_ids.sort()
        query_files = list_postprocess_jsons(query_base_dir)
        query_file_map = {
            os.path.splitext(os.path.basename(path))[0]: path for path in query_files
        }

        label_map = {}
        for vid in post_video_ids:
            label_suffix = "" if vid in query_file_map else " (Query 결과 없음)"
            label_map[vid] = f"{vid}.json{label_suffix}"

        default_index = 0
        current_video_id = st.session_state.get("selected_video_id")
        if current_video_id in post_video_ids:
            default_index = post_video_ids.index(current_video_id)

        selected_compare_index = st.selectbox(
            "비교할 video_id (Query JSON)",
            range(len(post_video_ids)),
            index=default_index,
            format_func=lambda i: label_map[post_video_ids[i]],
        )
        compare_video_id = post_video_ids[selected_compare_index]
        video_json_path = os.path.join(post_save_dir, f"{compare_video_id}.json")
        video_path = os.path.join(video_dir, f"{compare_video_id}.mp4")
        compare_query_json_path = os.path.join(query_base_dir, f"{compare_video_id}.json")

        existing_compare_query_text = None
        if os.path.isfile(compare_query_json_path):
            try:
                with open(compare_query_json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                existing_compare_query_text = (
                    payload.get("query") or ""
                ).strip() or None
            except Exception:
                existing_compare_query_text = None

        compare_query_inputs = render_query_inputs_for_video(
            compare_video_id,
            "compare_tab",
            existing_query_text=existing_compare_query_text,
        )
        compare_query_str_input = compare_query_inputs["query_str"]
        compare_query_str = compare_query_str_input.strip()
        compare_query_mode = compare_query_inputs["mode"]
        compare_query_top_k = compare_query_inputs["top_k"]
        compare_query_threshold = compare_query_inputs["threshold"]

        repo_abs_for_key = os.path.abspath(trace_repo_root)
        model_path_for_key = (
            os.path.abspath(trace_model_path)
            if trace_model_path
            else os.path.join(repo_abs_for_key, "trace-uni")
        )
        cache_key = "|".join(
            [
                compare_video_id,
                compare_query_str,
                repo_abs_for_key,
                model_path_for_key,
                trace_device,
                TRACE_CONDA_ENV,
                str(int(trace_num_frames)),
                str(int(trace_max_new_tokens)),
                compare_query_mode,
                str(int(compare_query_top_k)),
                f"{compare_query_threshold:.3f}",
            ]
        )
        trace_result = st.session_state.trace_cache.get(cache_key)

        run_compare_clicked = st.button(
            "비교 실행",
            key=f"compare_run_{compare_video_id}",
        )
        if run_compare_clicked:
            st.session_state.log_text = ""
            log_area.text("")
            code, _ = run_query_command(
                compare_video_id,
                compare_query_str_input,
                compare_query_mode,
                compare_query_top_k,
                compare_query_threshold,
                gpu_id,
                post_save_dir,
                query_base_dir,
                log_prefix="[5-compare]",
                success_message="비교용 Query 검색이 완료되었습니다.",
            )
            if code == 0 and os.path.isfile(video_path):
                try:
                    with st.spinner("TRACE 모델 추론 중..."):
                        inference_result = run_trace_temporal_grounding(
                            video_path,
                            compare_query_str_input,
                            trace_repo_root,
                            trace_model_path,
                            trace_device,
                            int(trace_max_new_tokens),
                            int(trace_num_frames),
                            TRACE_CONDA_ENV,
                        )
                    outputs = (inference_result.get("outputs") or {})
                    timestamps = outputs.get("timestamps") or []
                    parsed_start = None
                    parsed_end = None
                    for event in timestamps:
                        if isinstance(event, list) and len(event) >= 2:
                            parsed_start = safe_float(event[0])
                            parsed_end = safe_float(event[1])
                            break
                    inference_result["parsed_start_time"] = parsed_start
                    inference_result["parsed_end_time"] = parsed_end
                    captions = outputs.get("captions") or []
                    inference_result["parsed_caption"] = captions[0] if captions else ""
                    st.session_state.trace_cache[cache_key] = inference_result
                    trace_result = inference_result
                except Exception as exc:  # pylint: disable=broad-except
                    st.session_state.trace_cache[cache_key] = {"error": str(exc)}
                    trace_result = st.session_state.trace_cache[cache_key]
            elif code == 0 and not os.path.isfile(video_path):
                st.info(f"비디오 파일을 찾을 수 없습니다: {video_path}")

        st.markdown("**비교 Query:**")
        st.write(compare_query_str)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### LongVALE 2단계 결과")
            if not os.path.isfile(video_json_path):
                st.info(
                    f"{compare_video_id}에 대한 Postprocess JSON이 없습니다. 1단계를 먼저 실행하세요."
                )
            elif not os.path.isfile(compare_query_json_path):
                st.info("비교할 Query JSON이 없습니다. 비교 실행 버튼을 눌러 생성하세요.")
            else:
                try:
                    with open(compare_query_json_path, "r", encoding="utf-8") as f:
                        compare_query_payload = json.load(f)
                except Exception:
                    compare_query_payload = {}
                payload_query_text = (compare_query_payload.get("query") or "").strip()
                matches = compare_query_payload.get("matches") or []
                if not matches:
                    st.info("비교 Query 결과가 없습니다. 비교 실행 버튼을 눌러 새로 생성하세요.")
                elif payload_query_text != compare_query_str:
                    st.info(
                        "선택한 Query 문자열에 대한 결과가 없습니다. 비교 실행 버튼을 눌러 갱신하세요."
                    )
                else:
                    match_index = st.selectbox(
                        "비교할 LongVALE 구간 선택",
                        range(len(matches)),
                        format_func=lambda i: (
                            f"[score={matches[i].get('score', 0):.3f}] "
                            f"{matches[i].get('start_time', 0):.1f}–"
                            f"{matches[i].get('end_time', 0):.1f}s"
                        ),
                        key=f"compare_pipeline_match_{compare_video_id}",
                    )
                    pipeline_match = matches[match_index]
                    pipeline_start = safe_float(pipeline_match.get("start_time"))
                    pipeline_end = safe_float(
                        pipeline_match.get("end_time"), pipeline_start
                    )
                    pipeline_score = pipeline_match.get("score")
                    pipeline_score_str = (
                        f"{pipeline_score:.3f}" if pipeline_score is not None else "N/A"
                    )
                    pipeline_summary = (
                        pipeline_match.get("scene_topic")
                        or pipeline_match.get("summary")
                        or pipeline_match.get("matched_text")
                        or ""
                    )
                    st.write(
                        f"시간: {pipeline_start:.2f}–{pipeline_end:.2f}s "
                        f"(score={pipeline_score_str})"
                    )
                    if pipeline_summary:
                        st.write(pipeline_summary)
                    pipeline_clip = extract_video_subclip(
                        video_path,
                        compare_video_id,
                        pipeline_start,
                        pipeline_end,
                        "longvale",
                    )
                    if pipeline_clip:
                        st.video(pipeline_clip)
                    elif os.path.isfile(video_path):
                        st.video(video_path, start_time=int(pipeline_start))
                    else:
                        st.info(f"비디오 파일을 찾을 수 없습니다: {video_path}")

        with col_right:
            st.markdown("#### TRACE 결과")
            if not os.path.isfile(video_path):
                st.info(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            elif not trace_result:
                st.info("비교 실행 버튼을 눌러 TRACE 결과를 생성하세요.")
            elif trace_result.get("error"):
                st.error(f"추론 실패: {trace_result['error']}")
            else:
                tz_start = trace_result.get("parsed_start_time")
                tz_end = trace_result.get("parsed_end_time")
                tz_raw = json.dumps(trace_result, ensure_ascii=False, indent=2)
                if tz_start is None or tz_end is None:
                    st.warning("TRACE 모델이 올바른 시간 구간을 반환하지 않았습니다.")
                else:
                    tz_start_val = safe_float(tz_start)
                    tz_end_val = safe_float(tz_end, tz_start_val + 0.1)
                    st.write(f"시간: {tz_start_val:.2f}–{tz_end_val:.2f}s")
                    tz_clip = extract_video_subclip(
                        video_path,
                        compare_video_id,
                        tz_start_val,
                        tz_end_val,
                        "trace",
                    )
                    if tz_clip:
                        st.video(tz_clip)
                    else:
                        st.video(video_path, start_time=int(tz_start_val))
                caption_text = trace_result.get("parsed_caption")
                if caption_text:
                    st.write(caption_text)
                if tz_raw:
                    with st.expander("TRACE 원본 출력"):
                        st.code(tz_raw, language="json")
