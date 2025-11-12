import sys
sys.path.append('./')
import torch
import argparse
import json
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import decord
from moviepy.editor import VideoFileClip
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi

import clip
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

from longvalellm.model.beats.BEATs import BEATs, BEATsConfig
from transformers.models.whisper.modeling_whisper import WhisperModel
from transformers import WhisperFeatureExtractor

def prepare_models(clip_ckpt, beats_ckpt, whisper_ckpt, extract_mode, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    clip_model, beats_model, whisper_model = None, None, None

    # CLIP
    if extract_mode in ['all', 'video']:
        clip_model, _ = clip.load(clip_ckpt, device=device)
        clip_model.eval()

    # BEATs
    if extract_mode in ['all', 'audio']:
        beats_checkpoint = torch.load(beats_ckpt, map_location='cpu')
        beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
        beats_model = BEATs(beats_cfg)
        beats_model.load_state_dict(beats_checkpoint['model'])
        beats_model.to(device)
        beats_model.eval()

    # Whisper
    if extract_mode in ['all', 'speech']:
        whisper_model = WhisperModel.from_pretrained(whisper_ckpt).encoder
        whisper_model.to(device)
        whisper_model.eval()

    return clip_model, beats_model, whisper_model, device

class AVDataset(Dataset):
    def __init__(self, annotation, video_dir, audio_dir, whisper_ckpt, n_frames=100, audio_segment_len=2.0, extract_mode='all'):
        with open(annotation, 'r') as f:
            self.data = json.load(f)
        self.data = list(self.data.items())

        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.extract_mode = extract_mode

        self.n_frames = n_frames
        self.audio_segment_len = audio_segment_len
        self.sampling_rate = 16000

        # BEATs settings
        self.fbank_mean = 15.41663
        self.fbank_std = 6.55582

        # CLIP settings
        self.clip_transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # Whisper settings
        self.whisper_transform = WhisperFeatureExtractor.from_pretrained(whisper_ckpt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, item = self.data[index]
        video_path = os.path.join(self.video_dir, f'{video_id}.mp4')

        try:
            video_reader = decord.VideoReader(video_path, num_threads=1)
            total_frames = len(video_reader)
            duration = item["duration"]
            
            # Sample video frames 
            sampled_indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=np.int32)
            if self.extract_mode in ['all', 'video']:
                video_frames = video_reader.get_batch(sampled_indices).asnumpy()
                video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2) # (N, C, H, W)
                video_frames = self.clip_transform(video_frames / 255.0)
            else:
                video_frames = torch.empty(0)

            # Extract audio segments
            if self.audio_dir: 
                audio_path = os.path.join(self.audio_dir, f'{video_id}.wav')
                waveforms, sr = torchaudio.load(audio_path)
                if sr != self.sampling_rate:
                        waveforms = torchaudio.functional.resample(waveforms, sr, self.sampling_rate)

                if waveforms.dim() == 2:
                    waveforms = waveforms.mean(dim=0)  # Convert to mono
            else: 
                with VideoFileClip(video_path) as video_clip:
                    audio = video_clip.audio
                    if audio is None:
                        raise ValueError("Audio track not found")
                    
                    waveforms = audio.to_soundarray(fps=self.sampling_rate)
                    waveforms = torch.from_numpy(waveforms).float()

            # timestamps = torch.arange(0, self.n_frames, dtype=torch.float32) * (duration / self.n_frames)
            timestamps = torch.tensor([video_reader.get_frame_timestamp(idx)[0] for idx in sampled_indices])
            segment_len = int(self.sampling_rate * self.audio_segment_len)

            start_times = ((timestamps - self.audio_segment_len / 2) * self.sampling_rate).long()
            end_times = start_times + segment_len

            left_pad = max(0, -start_times.min().item())
            right_pad = max(0, end_times.max().item() - len(waveforms))

            start_times += left_pad
            indices = start_times.unsqueeze(1) + torch.arange(segment_len).unsqueeze(0)

            padded_waveforms = torch.nn.functional.pad(waveforms, (left_pad, right_pad))
            audio_waveforms = padded_waveforms[indices] # (N, T)
            return video_id, video_frames, audio_waveforms

        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            return video_id, None, None
        
    def process_for_beats(self, waveforms):
        # waveforms: (N, T)
        waveforms = waveforms * (2**15)

        def process_fbank(waveform):
            return ta_kaldi.fbank(
                waveform.unsqueeze(0),
                num_mel_bins=128,
                sample_frequency=self.sampling_rate,
                frame_length=25,
                frame_shift=10,
            )
        
        from torch import vmap
        fbanks = vmap(process_fbank)(waveforms) 
        fbanks = (fbanks - self.fbank_mean) / (2 * self.fbank_std)
        return fbanks  # (N, T', D)
    
    def process_for_whisper(self, waveforms):
        # waveforms: (N, T)
        waveforms = list(waveforms.cpu().numpy().astype(np.float32))
        spectrograms = self.whisper_transform(
            waveforms,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).input_features  
        return spectrograms  # (N, F, T')
    
def collate_fn(batch):
    video_ids, video_frames, audio_waveforms = zip(*batch)
    video_ids = list(video_ids)
    audio_waveforms = torch.cat(audio_waveforms, dim=0)

    if not (video_frames[0].numel() == 0):
        video_frames = torch.cat(video_frames, dim=0)
    else:
        video_frames = torch.empty(0)
    return video_ids, video_frames, audio_waveforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str, default='/root/datasets/jinho/LongVALE/data/longvale-annotations-training.json')
    parser.add_argument("--video_dir", type=str, default='/root/datasets/jinho/LongVALE/data/raw_videos/video_train_7240')
    parser.add_argument("--audio_dir", type=str, default='/root/datasets/jinho/LongVALE/data/raw_videos/wav_train_7240')  
    parser.add_argument("--video_feat_dir", type=str, default='/root/datasets/jinho/LongVALE/data/features_training/video_features')
    parser.add_argument("--audio_feat_dir", type=str, default='/root/datasets/jinho/LongVALE/data/features_training/synced_audio_features')
    parser.add_argument("--speech_feat_dir", type=str, default='/root/datasets/jinho/LongVALE/data/features_training/synced_speech_features')
    parser.add_argument("--clip_checkpoint", type=str, default='/root/datasets/jinho/LongVALE/checkpoints/ViT-L-14.pt')
    parser.add_argument("--beats_checkpoint", type=str, default='/root/datasets/jinho/LongVALE/checkpoints/BEATs_iter3_plus_AS20K.pt')
    parser.add_argument("--whisper_checkpoint", type=str, default='/root/datasets/jinho/LongVALE/checkpoints/openai-whisper-large-v2')
    parser.add_argument("--extract_mode", type=str, default='all', choices=['all', 'video', 'audio', 'speech'])
    parser.add_argument("--n_frames", type=int, default=100)
    parser.add_argument("--audio_segment_len", type=float, default=2.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    if args.extract_mode in ['all', 'video']:
        os.makedirs(args.video_feat_dir, exist_ok=True)
    if args.extract_mode in ['all', 'audio']:
        os.makedirs(args.audio_feat_dir, exist_ok=True)
    if args.extract_mode in ['all', 'speech']:
        os.makedirs(args.speech_feat_dir, exist_ok=True)

    clip_model, beats_model, whisper_model, device = prepare_models(
        args.clip_checkpoint, args.beats_checkpoint, args.whisper_checkpoint, args.extract_mode, args.gpu_id
    )

    dataset = AVDataset(args.annotation, args.video_dir, args.audio_dir, 
                        whisper_ckpt=args.whisper_checkpoint, n_frames=args.n_frames, audio_segment_len=args.audio_segment_len)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn) 
    assert data_loader.batch_size == 1, "batch_size must be 1"

    with torch.no_grad():
        for video_ids, video_frames, audio_waveforms in tqdm(data_loader):
            video_id = video_ids[0]
            
            save_path_video = os.path.join(args.video_feat_dir, f'{video_id}.npy')
            save_path_audio = os.path.join(args.audio_feat_dir, f'{video_id}.npy')
            save_path_speech = os.path.join(args.speech_feat_dir, f'{video_id}.npy')

            if os.path.exists(save_path_video) and os.path.exists(save_path_audio) and os.path.exists(save_path_speech):
                continue

            video_frames = video_frames.to(device)
        
            if args.extract_mode in ['all', 'video'] and not os.path.exists(save_path_video):
                video_features = clip_model.encode_image(video_frames) 
                video_features = video_features.cpu().numpy()
                np.save(save_path_video, video_features)

            if args.extract_mode in ['all', 'audio'] and not os.path.exists(save_path_audio):
                fbanks = dataset.process_for_beats(audio_waveforms).to(device)
                audio_features = beats_model.extract_features(fbanks)[0] 
                audio_features = audio_features.mean(dim=1)
                audio_features = audio_features.cpu().numpy()
                np.save(save_path_audio, audio_features)

            if args.extract_mode in ['all', 'speech'] and not os.path.exists(save_path_speech):
                spectrograms = dataset.process_for_whisper(audio_waveforms) 
                speech_features = []
                for spectrogram_batch in np.array_split(spectrograms, 3):  # GPU memory issue, process in 3 groups
                    spectrogram_batch = spectrogram_batch.to(device)
                    speech_feature = whisper_model(spectrogram_batch).last_hidden_state
                    speech_feature = speech_feature.mean(dim=1).squeeze() 
                    speech_features.append(speech_feature.cpu().numpy())
                speech_features = np.concatenate(speech_features, axis=0) 
                np.save(save_path_speech, speech_features)
