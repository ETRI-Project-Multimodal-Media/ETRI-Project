from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..") # add project root dir to sys.path
import sys
sys.path.append(root_dir)

import re
import argparse
import torch
import wandb
import time
import json
import numpy as np
from tqdm import tqdm
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/path/to/longvale-annotations-eval.json")
    parser.add_argument("--asr_feat_folder", type=str, default="/path/to/features_eval/speech_features_1171") 
    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = parse_args()
    # wandb.init(project="longvale-eval", config=vars(args))
    lengths = []
    dims = set()

    js = json.load(open(args.data_path))
    for id, data in tqdm(js.items()):
        asr_features = None
       
        if args.asr_feat_folder is not None:
            asr_feat_path = os.path.join(args.asr_feat_folder, f"{id}.npy")
            if os.path.isfile(asr_feat_path):
                asr_features = torch.from_numpy(np.load(asr_feat_path)).cuda()
                
                if asr_features.ndim == 1:
                    seq_len, hidden_dim = 1, asr_features.shape[0]
                elif asr_features.ndim == 2:
                    seq_len, hidden_dim = asr_features.shape
                else:
                    print(f"[Warning] : unexpected shape {asr_features.shape}")
                    continue

                lengths.append(seq_len)
                dims.add(hidden_dim)
    # 길이 분포 요약
    counter = Counter(lengths)
    print("\n=== Sequence length distribution ===")
    for l, c in counter.most_common():
        print(f"Length {l:4d}: {c} files")

    print("\n=== Hidden dim set ===")
    print(dims)

    print(f"\nTotal files checked: {len(lengths)}")
    print(f"Min length: {min(lengths)}, Max length: {max(lengths)}")


    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")


    # decoder 시작 토큰 준비
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

    if asr_features.ndim == 2:
        asr_features = asr_features.unsqueeze(0)  # (1, seq_len, hidden_dim)


    outputs = model.generate(
        encoder_outputs=asr_features,
        decoder_input_ids=decoder_input_ids,
        max_length=100
    )
    transcription = processor.batch_decode(outputs, skip_special_tokens=False)
    print(transcription)
