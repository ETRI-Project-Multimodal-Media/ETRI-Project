import sys
sys.path.append('./')
import os 
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from longvalellm.model.beats.BEATs import BEATs, BEATsConfig
from transformers.models.whisper.modeling_whisper import WhisperModel
from transformers import WhisperFeatureExtractor
import decord

class AVDataset(Dataset):
    def __init__(self, data_path, video_dir, audio_dir, save_dir, frame_fps=1.0, segment_len=1.0, extract_modality='all'):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.data = list(self.data.items())

        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.save_dir = save_dir
        self.extract_modality = extract_modality
        
        self.frame_fps = frame_fps
        self.segment_len = segment_len
        self.sampling_rate = 16000
        
        os.makedirs(os.path.join(save_dir, "video_features"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "audio_features"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "speech_features"), exist_ok=True)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_id, item = self.data[idx]
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        audio_path = os.path.join(self.audio_dir, f"{video_id}.wav")
        
        save_v_path = os.path.join(self.save_dir, "video_features", f"{video_id}.npy")
        save_a_path = os.path.join(self.save_dir, "audio_features", f"{video_id}.npy")
        save_s_path = os.path.join(self.save_dir, "speech_features", f"{video_id}.npy")
        
        if os.path.exists(save_v_path) and os.path.exists(save_a_path) and os.path.exists(save_s_path):
            return None
        
        duration = item['duration']
        timestamps = np.arange(0, duration, 1.0 / self.frame_fps)
        
        frames = None
        waveform = None
        
        if self.extract_modality in ['video', 'all'] and not os.path.exists(save_v_path):
            try: 
                video_reader = decord.VideoReader(video_path, num_threads=1)
                fps_ori = video_reader.get_avg_fps()
                total_frames = len(video_reader)
                
                frame_indices = np.round(timestamps * fps_ori).astype(int)
                frame_indices = np.clip(frame_indices, 0, total_frames - 1)
                
                frames = video_reader.get_batch(frame_indices).asnumpy() # (N, H, W, C)
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # (N, C, H, W)
                frames = frames.float() / 255.0
                
            except Exception as e:
                print(f"Error reading video {video_id}: {e}")
                frames = None
                
        if self.extract_modality in ['audio', 'speech', 'all']:
            if not os.path.exists(save_a_path) or not os.path.exists(save_s_path):
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    
                    if waveform.dim() == 2:
                        waveform = waveform.mean(dim=0)  
                    if sr != self.sampling_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                        waveform = resampler(waveform)
                    
                    waveform = waveform
                    
                except Exception as e:
                    print(f"Error reading audio {video_id}: {e}")
                    waveform = None

        return {
            'video_id': video_id,
            'video_frames': frames,
            'audio_waveform': waveform,
            'timestamps': timestamps,
            'save_v_path': save_v_path,
            'save_a_path': save_a_path,
            'save_s_path': save_s_path
        }

def collate_fn(batch):
    return batch[0]

def extract_video_features(model, frames, device): 
    clip_transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    frames = clip_transform(frames).to(device)
    with torch.no_grad():
        features = model.encode_image(frames)
        features = features.cpu().numpy()
    return features

def extract_audio_features(
    model, waveform, timestamps, 
    segment_len, sampling_rate, device
):
    fbank_mean = 15.41663
    fbank_std = 6.55582
    
    waveform = waveform * (2**15)

    fbank = ta_kaldi.fbank(
        waveform.unsqueeze(0),
        num_mel_bins=128, 
        sample_frequency=sampling_rate,
        frame_length=25, 
        frame_shift=10
    ) 
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    
    frames_per_sec = 100 # 10ms hop size
    frame_size = int(segment_len * frames_per_sec)
    total_fbank_frames = fbank.shape[0]
    
    fbanks = []
    
    for timestamp in timestamps:
        center_frame = int(timestamp.item() * frames_per_sec)
        start_frame = center_frame - (frame_size // 2)
        end_frame = start_frame + frame_size
    
        valid_start = max(0, start_frame)
        valid_end = min(end_frame, total_fbank_frames)
        
        if valid_start >= valid_end:
            chunk = torch.zeros((frame_size, fbank.shape[1]), dtype=fbank.dtype, device=fbank.device)
        else:
            chunk = fbank[valid_start:valid_end]
            
            pad_left = valid_start - start_frame
            pad_right = end_frame - valid_end
            
            if pad_left > 0 or pad_right > 0:
                chunk = F.pad(chunk, (0, 0, pad_left, pad_right))
            
        fbanks.append(chunk)
    fbanks = torch.stack(fbanks).to(device)
    
    with torch.no_grad():
        beats_feats = model.extract_features(fbanks)[0]
        beats_feats = beats_feats.mean(dim=1)
    return beats_feats.cpu().numpy()

def extract_speech_features(
    model, transform, waveform, timestamps, 
    segment_len, sampling_rate, device
):
    chunk_size = 30 * sampling_rate
    
    if len(waveform) % chunk_size != 0:
        pad_len = chunk_size - (len(waveform) % chunk_size)
        waveform = F.pad(waveform, (0, pad_len))
        
    waveform = waveform.view(-1, chunk_size) 
    
    spectrograms = transform(
        list(waveform.numpy()),
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).input_features.to(device)
    
    with torch.no_grad():
        whisper_feats = model(spectrograms).last_hidden_state  
    whisper_feats = whisper_feats.view(-1, whisper_feats.size(-1)).cpu()  
    
    total_tokens = whisper_feats.shape[0]
    sec_per_token = 0.02 # 30s / 1500 tokens 
    
    sliced_feats = [] 
    
    for timestamp in timestamps:
        start_time = timestamp.item() - segment_len / 2.0
        end_time = timestamp.item() + segment_len / 2.0
        
        start_idx = int(max(0, start_time) / sec_per_token)
        end_idx = int(end_time / sec_per_token)

        start_idx = min(start_idx, total_tokens)
        end_idx = min(end_idx, total_tokens)
        
        if start_idx >= end_idx:
            feature = torch.zeros((whisper_feats.size(-1),), dtype=whisper_feats.dtype, device=whisper_feats.device)
        else:
            feature = whisper_feats[start_idx:end_idx].mean(dim=0)
            
        sliced_feats.append(feature)  
    sliced_feats = torch.stack(sliced_feats)
    return sliced_feats.cpu().numpy()

def prepare_models(clip_ckpt, beats_ckpt, whisper_ckpt, extract_modality, device):
    clip_model, beats_model, whisper_model, whisper_transform = None, None, None, None
    
    # CLIP
    if extract_modality in ['video', 'all']:
        clip_model, _ = clip.load(clip_ckpt, device=device)
        clip_model.eval()

    # BEATs
    if extract_modality in ['audio', 'all']:
        beats_checkpoint = torch.load(beats_ckpt, map_location='cpu')
        beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
        beats_model = BEATs(beats_cfg)
        beats_model.load_state_dict(beats_checkpoint['model'])
        beats_model.to(device)
        beats_model.eval()

    # Whisper
    if extract_modality in ['speech', 'all']:
        whisper_model = WhisperModel.from_pretrained(whisper_ckpt).encoder
        whisper_model.to(device)
        whisper_model.eval()
        whisper_transform = WhisperFeatureExtractor.from_pretrained(whisper_ckpt)

    return clip_model, beats_model, whisper_model, whisper_transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser.add_argument("--data_path", type=str, default="./data/longvale-annotations-eval.json")
    parser.add_argument("--video_dir", type=str, default="./data/raw_data/video_test")
    parser.add_argument("--audio_dir", type=str, default="./data/raw_data/audio_test")
    parser.add_argument("--clip_checkpoint", type=str, default="./checkpoints/ViT-L-14.pt")
    parser.add_argument("--beats_checkpoint", type=str, default="./checkpoints/BEATs_iter3_plus_AS20K.pt")
    parser.add_argument("--whisper_checkpoint", type=str, default="./checkpoints/openai/whisper-large-v2")
    parser.add_argument("--save_dir", type=str, default="./data/features_tree")
    parser.add_argument("--extract_modality", type=str, choices=['video', 'audio','speech', 'all'], default="all")
    parser.add_argument("--frame_fps", type=float, default=1.0)
    parser.add_argument("--segment_len", type=float, default=1.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    
    with open(args.data_path, 'r') as f:
        data = json.load(f)
        
    device = torch.device(f'cuda:{args.gpu_id}')    
    clip_model, beats_model, whisper_model, whisper_transform = prepare_models(
        args.clip_checkpoint, args.beats_checkpoint, args.whisper_checkpoint, 
        args.extract_modality, device
    )
    
    dataset = AVDataset(
        data_path=args.data_path,
        video_dir=args.video_dir,
        audio_dir=args.audio_dir,
        save_dir=args.save_dir,
        frame_fps=args.frame_fps,
        segment_len=args.segment_len,
        extract_modality=args.extract_modality
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    for batch in tqdm(dataloader):
        if batch is None: 
            continue
        
        video_id = batch['video_id']
        video_frames = batch['video_frames']
        audio_waveform = batch['audio_waveform']
        timestamps = batch['timestamps']
        save_v_path = batch['save_v_path']
        save_a_path = batch['save_a_path']
        save_s_path = batch['save_s_path']

        if args.extract_modality in ['video', 'all'] and video_frames is not None:
            if not os.path.exists(save_v_path):
                video_features = extract_video_features(
                    clip_model, video_frames, device
                ) 
                np.save(save_v_path, video_features)
            
        if args.extract_modality in ['audio', 'speech', 'all'] and audio_waveform is not None:
            if args.extract_modality in ['audio', 'all'] and not os.path.exists(save_a_path):
                audio_features = extract_audio_features(
                    beats_model, audio_waveform, timestamps, 
                    segment_len=dataset.segment_len,
                    sampling_rate=dataset.sampling_rate,
                    device=device
                )
                np.save(save_a_path, audio_features)
            
            if args.extract_modality in ['speech', 'all'] and not os.path.exists(save_s_path):
                speech_features = extract_speech_features(
                    whisper_model, whisper_transform, 
                    audio_waveform, timestamps, 
                    segment_len=dataset.segment_len,
                    sampling_rate=dataset.sampling_rate,
                    device=device
                )
                np.save(save_s_path, speech_features)
    
