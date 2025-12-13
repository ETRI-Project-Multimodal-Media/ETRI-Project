#!/usr/bin/env python3
"""
Batch runner that mirrors the Streamlit demo pipeline.

Edit VIDEO_IDS and GPU_IDS below, then run:
    python batch_pipeline_runner.py

For every video_id the script executes:
Feature Extraction → Tree → Caption → Summary → Postprocess.
Outputs under outputs/postprocess so Streamlit can jump straight to step 2.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# TODO: Replace the example IDs below with the actual targets before running.
# ---------------------------------------------------------------------------
VIDEO_IDS: List[str] = [
    "1zjizzVY_H4",
    "-powI0eeLbw",
    "15GCaMdApS4",
    "1Gf2W3eYP28",
    "2Vh-xCGMLzg",
    "2lJMU_78m7A",
    "9AffWPWL9RM",
    "-QEfaBLGPQk",
    "-eoe6CBR1wE",
    "1sM9Qs-0Vdk",
]
# GPU IDs as they should appear in CUDA_VISIBLE_DEVICES.
GPU_IDS: List[str] = ["3", "4", "5", "6", "7"]
# Max simultaneous pipelines per GPU.
MAX_PROCESSES_PER_GPU = 2
# Skip computation if postprocess output already exists.
SKIP_EXISTING_OUTPUT = True

# Tokens can also be supplied via environment variables.
HF_TOKEN = os.environ.get(
    "HF_TOKEN", "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"
)
HUB_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN", HF_TOKEN)

CAPTION_SIMILARITY_THRESHOLD = 0.9
TREE_MERGE_THRESHOLD = 0.65

VIDEO_DIR = BASE_DIR / "data/raw_data/video"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PROMPT_PATH = BASE_DIR / "data/prompt.json"
TREE_SAVE_DIR = BASE_DIR / "outputs/tree"
POST_SAVE_DIR = BASE_DIR / "outputs/postprocess"
TMP_ANNOTATION_DIR = BASE_DIR / "outputs/tmp_annotation"
DEBUG_PATH = BASE_DIR / "logs/debug.text"

TREE_V_FEAT = BASE_DIR / "data/features_tree/video_features"
TREE_A_FEAT = BASE_DIR / "data/features_tree/audio_features"
TREE_S_FEAT = BASE_DIR / "data/features_tree/speech_features"

MODEL_V_FEAT = BASE_DIR / "data/features_model/video_features"
MODEL_A_FEAT = BASE_DIR / "data/features_model/audio_features"
MODEL_S_FEAT = BASE_DIR / "data/features_model/speech_features"
SPEECH_ASR_DIR = BASE_DIR / "data/features_model/speech_asr"

MODEL_BASE = CHECKPOINT_DIR / "vicuna-7b-v1.5"
MODEL_STAGE2 = (
    CHECKPOINT_DIR
    / "longvalellm-vicuna-v1-5-7b"
    / "longvale-vicuna-v1-5-7b-stage2-bp"
)
MODEL_STAGE3 = (
    CHECKPOINT_DIR
    / "longvalellm-vicuna-v1-5-7b"
    / "longvale-vicuna-v1-5-7b-stage3-it"
)
MODEL_MM_MLP = CHECKPOINT_DIR / "vtimellm_stage1_mm_projector.bin"

LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    with LOG_LOCK:
        print(f"[{now}] {message}", flush=True)


def build_env(extra: Dict[str, str] | None = None) -> Dict[str, str]:
    env = os.environ.copy()
    src_path = str(BASE_DIR / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{current}" if current else src_path
    if extra:
        env.update(extra)
    return env


def run_command(cmd, env: Dict[str, str] | None = None) -> Tuple[int, str]:
    shell = isinstance(cmd, str)
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        shell=shell,
        text=True,
        capture_output=True,
        env=build_env(env),
    )
    output = result.stdout or ""
    if result.stderr:
        output += ("\n" + result.stderr)
    return result.returncode, output.strip()


def has_any_files(folder: Path, exts: Iterable[str]) -> bool:
    if not folder.is_dir():
        return False
    for name in folder.iterdir():
        if name.is_file() and name.suffix.lower() in exts:
            return True
    return False


def ensure_tree_features() -> bool:
    if all(
        has_any_files(path, [".npy"])
        for path in (TREE_V_FEAT, TREE_A_FEAT, TREE_S_FEAT)
    ):
        log("[features/tree] Existing files detected; skip extraction.")
        return True
    log("[features/tree] Running scripts/features_tree.sh all ...")
    code, out = run_command("bash scripts/features_tree.sh all")
    log(out)
    if code != 0:
        log(f"[features/tree] Failed with exit code {code}.")
        return False
    return True


def ensure_model_features() -> bool:
    video_ok = has_any_files(MODEL_V_FEAT, [".npy"])
    audio_ok = has_any_files(MODEL_A_FEAT, [".npy"])
    speech_ok = has_any_files(MODEL_S_FEAT, [".npy"])
    asr_ok = has_any_files(SPEECH_ASR_DIR, [".json"])
    if video_ok and audio_ok and speech_ok and asr_ok:
        log("[features/model] Existing files detected; skip extraction.")
        return True
    log("[features/model] Running scripts/features_longvale.sh all ...")
    code, out = run_command("bash scripts/features_longvale.sh all")
    log(out)
    if code != 0:
        log(f"[features/model] Failed with exit code {code}.")
        return False
    return True


def get_video_duration(video_path: Path) -> float:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}\n{result.stderr}")
    return float(result.stdout.strip())


def create_annotation(video_id: str, duration: float) -> Path:
    TMP_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    anno_path = TMP_ANNOTATION_DIR / f"{video_id}_annotation.json"
    payload = {video_id: {"duration": duration}}
    with open(anno_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return anno_path


def quote_path(path: Path) -> str:
    return shlex.quote(str(path))


def run_tree_stage(annotation_path: Path) -> Tuple[int, str]:
    cmd = (
        "python src/eventtree/tree/tree.py "
        f"--data_path {quote_path(annotation_path)} "
        f"--video_feat_folder {quote_path(TREE_V_FEAT)} "
        f"--audio_feat_folder {quote_path(TREE_A_FEAT)} "
        f"--speech_feat_folder {quote_path(TREE_S_FEAT)} "
        f"--save_dir {quote_path(TREE_SAVE_DIR)}"
    )
    return run_command(cmd)


def run_caption_stage(tree_path: Path, gpu_id: str) -> Tuple[int, str]:
    cmd = (
        "python src/eventtree/caption_longvale.py "
        f"--tree_path {quote_path(tree_path)} "
        f"--prompt_path {quote_path(PROMPT_PATH)} "
        f"--save_path {quote_path(tree_path)} "
        f"--video_feat_folder {quote_path(MODEL_V_FEAT)} "
        f"--audio_feat_folder {quote_path(MODEL_A_FEAT)} "
        f"--asr_feat_folder {quote_path(MODEL_S_FEAT)} "
        f"--model_base {quote_path(MODEL_BASE)} "
        f"--stage2 {quote_path(MODEL_STAGE2)} "
        f"--stage3 {quote_path(MODEL_STAGE3)} "
        f"--pretrain_mm_mlp_adapter {quote_path(MODEL_MM_MLP)} "
        f"--similarity_threshold {CAPTION_SIMILARITY_THRESHOLD}"
    )
    env = {"CUDA_VISIBLE_DEVICES": gpu_id}
    return run_command(cmd, env=env)


def run_summary_stage(tree_path: Path, gpu_id: str) -> Tuple[int, str]:
    inner = (
        "source ~/anaconda3/etc/profile.d/conda.sh && "
        "conda activate eventtree-post && "
        f"HF_TOKEN={shlex.quote(HF_TOKEN)} "
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        "python src/eventtree/summary_llama3.py "
        f"--tree_path {quote_path(tree_path)} "
        f"--prompt_path {quote_path(PROMPT_PATH)} "
        f"--save_path {quote_path(tree_path)}"
    )
    return run_command(["bash", "-lc", inner])


def run_postprocess_stage(tree_path: Path, gpu_id: str) -> Tuple[int, str]:
    POST_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    inner = (
        "source ~/anaconda3/etc/profile.d/conda.sh && "
        "conda activate eventtree-post && "
        f"HUGGINGFACE_HUB_TOKEN={shlex.quote(HUB_TOKEN)} "
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        "python src/postprocess/postprocess.py "
        f"--input {quote_path(tree_path)} "
        f"--output-dir {quote_path(POST_SAVE_DIR)} "
        f"--speech-json-dir {quote_path(SPEECH_ASR_DIR)} "
        f"--merge-threshold {TREE_MERGE_THRESHOLD} "
        f"--not-json-dir {quote_path(DEBUG_PATH)}"
    )
    return run_command(["bash", "-lc", inner])


def run_pipeline_for_video(video_id: str, gpu_id: str) -> Tuple[bool, str]:
    prefix = f"[video:{video_id}][gpu:{gpu_id}]"
    try:
        TREE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        TMP_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        postprocess_output = POST_SAVE_DIR / f"{video_id}.json"
        if SKIP_EXISTING_OUTPUT and postprocess_output.is_file():
            msg = f"{prefix} Skipping (postprocess already exists)."
            log(msg)
            return True, msg

        video_path = VIDEO_DIR / f"{video_id}.mp4"
        log(f"{prefix} Calculating duration for {video_path} ...")
        duration = get_video_duration(video_path)
        annotation_path = create_annotation(video_id, duration)
        log(f"{prefix} Created annotation {annotation_path}.")

        log(f"{prefix} Starting tree step ...")
        code, output = run_tree_stage(annotation_path)
        log(f"{prefix} tree output:\n{output}")
        if code != 0:
            raise RuntimeError(
                f"{prefix} Step 'tree' failed with exit code {code}."
            )

        tree_file = TREE_SAVE_DIR / f"{video_id}.json"
        if not tree_file.is_file():
            raise FileNotFoundError(f"{prefix} Tree output missing: {tree_file}")

        steps = [
            ("caption", run_caption_stage),
            ("summary", run_summary_stage),
            ("postprocess", run_postprocess_stage),
        ]

        for step_name, func in steps:
            log(f"{prefix} Starting {step_name} step ...")
            code, output = func(tree_file, gpu_id)
            log(f"{prefix} {step_name} output:\n{output}")
            if code != 0:
                raise RuntimeError(
                    f"{prefix} Step '{step_name}' failed with exit code {code}."
                )

        msg = f"{prefix} Completed. Postprocess → {postprocess_output}"
        log(msg)
        return True, msg
    except Exception as exc:
        error_msg = f"{prefix} Failed: {exc}"
        log(error_msg)
        return False, error_msg


def distribute_videos(video_ids: List[str], gpu_ids: List[str]) -> Dict[str, List[str]]:
    assignments: Dict[str, List[str]] = {gpu: [] for gpu in gpu_ids}
    if not gpu_ids:
        return assignments
    for idx, video_id in enumerate(video_ids):
        gpu = gpu_ids[idx % len(gpu_ids)]
        assignments[gpu].append(video_id)
    return assignments


def run_gpu_worker(gpu_id: str, videos: List[str], results: List[Tuple[str, str, bool, str]], lock: threading.Lock):
    if not videos:
        log(f"[gpu:{gpu_id}] No videos assigned.")
        return
    log(f"[gpu:{gpu_id}] Scheduled videos: {videos}")
    with ThreadPoolExecutor(max_workers=MAX_PROCESSES_PER_GPU) as executor:
        future_map = {
            executor.submit(run_pipeline_for_video, video_id, gpu_id): video_id
            for video_id in videos
        }
        for future in as_completed(future_map):
            video_id = future_map[future]
            success, message = future.result()
            with lock:
                results.append((video_id, gpu_id, success, message))


def main():
    if not VIDEO_IDS:
        log("VIDEO_IDS is empty. Edit batch_pipeline_runner.py before running.")
        return
    if not GPU_IDS:
        log("GPU_IDS is empty. Specify at least one GPU.")
        return

    TREE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    POST_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    if not ensure_tree_features():
        log("Tree feature extraction failed. Aborting.")
        return
    if not ensure_model_features():
        log("Model feature extraction failed. Aborting.")
        return

    assignments = distribute_videos(VIDEO_IDS, GPU_IDS)
    results: List[Tuple[str, str, bool, str]] = []
    results_lock = threading.Lock()

    threads = []
    for gpu_id, videos in assignments.items():
        thread = threading.Thread(
            target=run_gpu_worker, args=(gpu_id, videos, results, results_lock)
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    order = {video_id: idx for idx, video_id in enumerate(VIDEO_IDS)}
    results.sort(key=lambda item: order.get(item[0], len(VIDEO_IDS)))

    log("--- Pipeline summary ---")
    for video_id, gpu_id, success, message in results:
        status = "OK" if success else "FAILED"
        log(f"{status} | video_id={video_id} | gpu={gpu_id} | {message}")
    log("Done.")


if __name__ == "__main__":
    main()
