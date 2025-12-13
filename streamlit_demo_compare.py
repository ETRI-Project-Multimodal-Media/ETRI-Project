import os
import re
import subprocess
import json
from pathlib import Path

import cv2
import graphviz
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_CACHE_DIR = os.path.join(BASE_DIR, "outputs", "frame_cache")
CLIP_CACHE_DIR = os.path.join(BASE_DIR, "outputs", "clip_cache")
HF_VTG_MODEL_ID = "RuizheChen/ColdStart_Temporal_GroundQA_Grounding_512"

_VTG_MODEL = None
_VTG_PROCESSOR = None
_VTG_DEVICE = None


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
tree_view_area = st.empty()


def append_log(text):
    if st.session_state.log_text:
        st.session_state.log_text += "\n" + text
    else:
        st.session_state.log_text = text
    log_area.text(st.session_state.log_text)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _mid_frame_capture(video_path: str, start_time: float, end_time: float) -> str | None:
    """주어진 구간의 중간 프레임을 추출해 캐시 폴더에 저장하고 경로를 반환한다."""
    if not os.path.isfile(video_path):
        return None
    _ensure_dir(FRAME_CACHE_DIR)
    stem = Path(video_path).stem
    mid = (start_time + end_time) / 2.0
    cache_name = f"{stem}_{start_time:.3f}_{end_time:.3f}.jpg"
    cache_path = os.path.join(FRAME_CACHE_DIR, cache_name)
    if os.path.isfile(cache_path):
        return cache_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame = int(mid * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    success, frame = cap.read()
    if not success or frame is None:
        cap.release()
        return None

    # 크기 축소 (너비 140px 기준 - 트리에서 너무 크지 않게)
    height, width = frame.shape[:2]
    target_w = 140
    scale = target_w / width if width > 0 else 1.0
    resized = cv2.resize(frame, (target_w, int(height * scale)))
    cv2.imwrite(cache_path, resized)
    cap.release()
    return cache_path


def _get_video_duration(video_path: str) -> float | None:
    if not os.path.isfile(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0.0 or frame_count <= 0.0:
        return None
    return float(frame_count / fps)


def _extract_video_clip(video_path: str, start_time: float, end_time: float) -> str | None:
    """ffmpeg을 사용해 해당 구간의 비디오 클립을 잘라 캐시 폴더에 저장하고 경로를 반환한다."""
    if not os.path.isfile(video_path):
        return None

    _ensure_dir(CLIP_CACHE_DIR)
    stem = Path(video_path).stem
    safe_start = max(0.0, float(start_time))
    safe_end = max(safe_start, float(end_time))
    if safe_end <= safe_start:
        safe_end = safe_start + 2.0

    clip_name = f"{stem}_{safe_start:.3f}_{safe_end:.3f}.mp4"
    clip_path = os.path.join(CLIP_CACHE_DIR, clip_name)
    if os.path.isfile(clip_path):
        return clip_path

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{safe_start:.3f}",
        "-to",
        f"{safe_end:.3f}",
        "-i",
        video_path,
        "-c",
        "copy",
        clip_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        st.error("ffmpeg 명령을 찾을 수 없습니다. ffmpeg를 설치해 주세요.")
        return None

    if result.returncode != 0:
        st.error("ffmpeg 실행 중 오류가 발생했습니다. stderr를 확인해 주세요.")
        return None

    return clip_path
def _load_video_frames(video_path: str, num_frames: int = 16):
    """비디오에서 균일하게 num_frames 장의 프레임을 추출해 PIL 이미지 리스트로 반환."""
    try:
        from PIL import Image
    except ImportError:
        st.error("Pillow(PIL)가 설치되어 있지 않습니다. `pip install pillow` 후 다시 시도하세요.")
        return None

    if not os.path.isfile(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if frame_count <= 0:
        cap.release()
        return None

    step = max(int(frame_count // num_frames), 1)
    frames = []
    idx = 0
    while len(frames) < num_frames and idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success or frame is None:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        idx += step

    cap.release()
    if not frames:
        return None
    return frames


def _parse_temporal_json(text: str, duration: float) -> tuple[float, float] | None:
    """모델 출력 텍스트에서 {\"start\": float, \"end\": float} JSON을 파싱."""
    text = text.strip()
    if not text:
        return None

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
    else:
        candidate = text

    try:
        data = json.loads(candidate)
        start = float(data.get("start", 0.0))
        end = float(data.get("end", start))
    except Exception:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if len(nums) >= 2:
            start = float(nums[0])
            end = float(nums[1])
        elif len(nums) == 1:
            start = float(nums[0])
            end = start + 2.0
        else:
            return None

    start = max(0.0, start)
    end = max(start, end)
    if duration > 0:
        start = min(start, duration)
        end = min(end, duration)
    if end <= start:
        end = min(duration, start + 2.0) if duration > 0 else start + 2.0
    return start, end


def _run_hf_temporal_grounding(video_path: str, query_text: str) -> tuple[float, float] | None:
    """HuggingFace Qwen2.5-VL 기반 Video Temporal Grounding (ColdStart_Temporal_GroundQA_Grounding_512) 호출."""
    if not os.path.isfile(video_path):
        st.warning(f"원본 비디오 파일을 찾을 수 없습니다: {video_path}")
        return None

    duration = _get_video_duration(video_path)
    if duration is None or duration <= 0.0:
        st.warning("비디오 길이를 계산할 수 없습니다.")
        return None

    frames = _load_video_frames(video_path, num_frames=16)
    if not frames:
        st.warning("비디오 프레임을 읽어오지 못했습니다.")
        return None

    global _VTG_MODEL, _VTG_PROCESSOR, _VTG_DEVICE

    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError:
        st.error(
            "Temporal Grounding 모델을 사용하려면 `torch`와 `transformers`가 필요합니다. "
            "`pip install torch transformers` 후 다시 시도하세요."
        )
        return None

    if _VTG_MODEL is None or _VTG_PROCESSOR is None:
        try:
            _VTG_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            _VTG_PROCESSOR = AutoProcessor.from_pretrained(
                HF_VTG_MODEL_ID,
                trust_remote_code=True,
            )
            _VTG_MODEL = AutoModelForVision2Seq.from_pretrained(
                HF_VTG_MODEL_ID,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            if not torch.cuda.is_available():
                _VTG_MODEL.to(_VTG_DEVICE)
        except Exception as e:
            st.error(f"HuggingFace Temporal Grounding 모델 로드 중 오류가 발생했습니다: {e}")
            return None

    try:
        prompt = (
            "You are a video temporal grounding model. "
            "Given the following video and a natural language query, "
            "return only a JSON object with the start and end time (in seconds) "
            "when the query is best grounded in the video. "
            'Use the format: {\"start\": <float>, \"end\": <float>} with no extra text.\n\n'
            f"Query: {query_text}"
        )

        inputs = _VTG_PROCESSOR(
            videos=[frames],
            text=[prompt],
            return_tensors="pt",
        )

        if _VTG_DEVICE:
            inputs = {k: v.to(_VTG_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = _VTG_MODEL.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        output_text = _VTG_PROCESSOR.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
    except Exception as e:
        st.error(f"HuggingFace Temporal Grounding 추론 중 오류가 발생했습니다: {e}")
        return None

    parsed = _parse_temporal_json(output_text, duration)
    if not parsed:
        st.warning(
            "Temporal Grounding 모델 출력에서 구간을 파싱하지 못했습니다. "
            "기본값 0~5초 구간을 사용합니다."
        )
        default_end = min(duration, 5.0)
        return 0.0, default_end

    return parsed


def _node_tooltip_text(node: dict) -> str:
    # caption이 우선, 없으면 postprocess LOD summary 사용
    if node.get("caption"):
        return node["caption"]
    lod = node.get("postprocess", {}).get("result", {}).get("LOD", {})
    return lod.get("summary") or lod.get("scene_topic") or ""


def _build_graph_from_tree(tree_root: dict, video_path: str) -> graphviz.Digraph:
    """트리 JSON을 Graphviz 트리로 변환 (레벨별 가로 정렬, 부모-자식 연결)."""
    g = graphviz.Digraph(format="svg")
    g.attr(rankdir="TB", splines="ortho")
    g.attr("node", shape="box", style="rounded")

    level_nodes: dict[int, list[str]] = {}
    counter = {"n": 0}

    def walk(node: dict, parent_id: str | None):
        counter["n"] += 1
        node_id = f"n{counter['n']}"
        level = int(node.get("level", 0))
        start_time = float(node.get("start_time", 0.0))
        end_time = float(node.get("end_time", start_time))
        tooltip = _node_tooltip_text(node)
    img_path = _mid_frame_capture(video_path, start_time, end_time)

    node_kwargs = {
        "tooltip": tooltip,
        "imagescale": "true",
        "fixedsize": "true",
        # 노드 크기를 줄여 전체 트리가 잘 보이도록 조정
        "height": "0.9",
        "width": "1.4",
        }
        if img_path:
            node_kwargs.update({"label": "", "image": img_path})
        else:
            node_kwargs.update(
                {
                    "label": f"{start_time:.1f}-{end_time:.1f}",
                    "style": "filled,rounded",
                    "fillcolor": "#f5f5f5",
                }
            )

        g.node(node_id, **node_kwargs)
        if parent_id:
            g.edge(parent_id, node_id)

        level_nodes.setdefault(level, []).append(node_id)
        for child in node.get("children", []):
            walk(child, node_id)

    walk(tree_root, None)

    # 같은 level을 같은 rank로 맞춰 가로 정렬
    for level, nodes in sorted(level_nodes.items(), key=lambda x: x[0]):
        with g.subgraph(name=f"rank_{level}") as sub:
            sub.attr(rank="same")
            for nid in nodes:
                sub.node(nid)
    return g


def render_tree_view(tree_json_path: str, video_dir: str):
    if not os.path.isfile(tree_json_path):
        st.warning(f"트리 JSON을 찾을 수 없습니다: {tree_json_path}")
        return
    with open(tree_json_path, "r") as f:
        data = json.load(f)

    # postprocess 결과 형식: {"video_id": "...", "tree": {...}}
    if "tree" in data:
        video_id = data.get("video_id") or Path(tree_json_path).stem
        tree_root = data["tree"]
    else:
        # tree 단계 형식: {video_id: tree}
        video_id, tree_root = next(iter(data.items()))

    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    graph = _build_graph_from_tree(tree_root, video_path)
    tree_view_area.graphviz_chart(graph, use_container_width=True)


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


st.markdown("---")
tab_tree, tab_compare = st.tabs(["트리 구조(노드 뷰)", "Query 결과 비교"])

with tab_tree:
    st.markdown("### 트리 구조(노드 뷰)")

    tree_view_json = st.text_input(
        "트리 JSON 경로 (postprocess 또는 tree 결과)",
        video_json,
        key="tree_view_json",
    )
    video_raw_dir = st.text_input(
        "원본 비디오 디렉터리 (video_id.mp4가 위치한 경로)",
        "./data/raw_data/video",
        key="video_raw_dir",
    )

    if st.button("트리 시각화 갱신"):
        try:
            render_tree_view(tree_view_json, video_raw_dir)
        except Exception as e:
            st.error(f"트리 시각화 중 오류가 발생했습니다: {e}")


with tab_compare:
    st.markdown("### Query 결과 비교 (기존 파이프라인 vs 다른 모델)")

    compare_video_raw_dir = st.text_input(
        "원본 비디오 디렉터리 (video_id.mp4가 위치한 경로)",
        "./data/raw_data/video",
        key="compare_video_raw_dir",
    )

    available_video_ids: list[str] = []
    if os.path.isdir(post_save_dir):
        for fname in sorted(os.listdir(post_save_dir)):
            if fname.endswith(".json"):
                vid = os.path.splitext(fname)[0]
                available_video_ids.append(vid)
    else:
        st.info(f"Postprocess 출력 디렉터리를 찾을 수 없습니다: {post_save_dir}")

    selected_video_id = st.selectbox(
        "video_id 선택 (Postprocess JSON 기준)",
        available_video_ids,
        index=0 if available_video_ids else None,
    ) if available_video_ids else ""

    compare_query_text = st.text_input(
        "Query 텍스트",
        value=query_str,
        key="compare_query_text",
    )

    if st.button("Query 결과 비교 실행"):
        if not selected_video_id:
            st.warning("video_id를 선택하세요.")
        elif not compare_query_text:
            st.warning("Query 텍스트를 입력하세요.")
        else:
            post_json_path = os.path.join(post_save_dir, f"{selected_video_id}.json")
            if not os.path.isfile(post_json_path):
                st.error(f"Postprocess JSON을 찾을 수 없습니다: {post_json_path}")
            else:
                query_dir = os.path.dirname(query_save_dir) or "./outputs/query"
                _ensure_dir(query_dir)
                compare_query_output = os.path.join(
                    query_dir, f"{selected_video_id}_compare.json"
                )

                cmd = (
                    "bash -lc "
                    "\"source ~/anaconda3/etc/profile.d/conda.sh && "
                    "conda activate eventtree-post && "
                    f"CUDA_VISIBLE_DEVICES={gpu_id} "
                    "python src/query/search_queries.py "
                    f'--input \\\"{post_json_path}\\\" '
                    f'--query \\\"{compare_query_text}\\\" '
                    f'--mode \\\"{mode}\\\" '
                    f'--output \\\"{compare_query_output}\\\"\"'
                )

                code, out = run_command(cmd)
                if code != 0:
                    st.error("기존 파이프라인 Query 실행 중 오류가 발생했습니다.")
                    st.text(out)
                else:
                    try:
                        with open(compare_query_output, "r", encoding="utf-8") as f:
                            query_result = json.load(f)
                    except Exception as e:
                        st.error(f"Query 결과 JSON을 읽는 중 오류가 발생했습니다: {e}")
                        query_result = None

                    pipeline_segment = None
                    if query_result and isinstance(query_result, dict):
                        matches = query_result.get("matches") or []
                        if matches:
                            top = matches[0]
                            pipeline_segment = {
                                "start_time": float(top.get("start_time", 0.0)),
                                "end_time": float(top.get("end_time", 0.0)),
                                "score": top.get("score"),
                                "matched_text": top.get("matched_text"),
                                "mode": query_result.get("mode"),
                            }

                    video_path = os.path.join(
                        compare_video_raw_dir, f"{selected_video_id}.mp4"
                    )
                    hf_segment = _run_hf_temporal_grounding(
                        video_path, compare_query_text
                    )

                    st.markdown(f"**Query:** {compare_query_text} (video_id: {selected_video_id})")

                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown("#### 기존 파이프라인 결과")
                        if not pipeline_segment:
                            st.info("기존 파이프라인에서 매칭된 구간을 찾지 못했습니다.")
                        else:
                            pl_start = pipeline_segment["start_time"]
                            pl_end = pipeline_segment["end_time"]
                            pl_clip = _extract_video_clip(video_path, pl_start, pl_end)

                            st.markdown(
                                f"- 시간 구간: {pl_start:.2f} ~ {pl_end:.2f}초"
                            )
                            if pipeline_segment.get("score") is not None:
                                st.markdown(
                                    f"- score: {pipeline_segment['score']:.3f}"
                                )
                            if pipeline_segment.get("matched_text"):
                                st.markdown(
                                    f"- matched text: {pipeline_segment['matched_text']}"
                                )
                            if pl_clip:
                                st.video(pl_clip)
                            else:
                                st.info("비디오 클립을 생성하지 못했습니다.")

                    with col_right:
                        st.markdown("#### 다른 모델 결과 (HuggingFace VTG)")
                        if not hf_segment:
                            st.info(
                                "다른 모델(HF Video Temporal Grounding) 결과를 사용할 수 없습니다."
                            )
                        else:
                            hf_start, hf_end = hf_segment
                            hf_clip = _extract_video_clip(
                                video_path, hf_start, hf_end
                            )
                            st.markdown(
                                f"- 시간 구간: {hf_start:.2f} ~ {hf_end:.2f}초"
                            )
                            if hf_clip:
                                st.video(hf_clip)
                            else:
                                st.info("비디오 클립을 생성하지 못했습니다.")
