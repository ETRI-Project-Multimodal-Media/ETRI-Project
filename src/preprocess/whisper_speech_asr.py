import sys
sys.path.append('./')
from longvalellm.mm_utils import SpeechExtractor
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader 
import torch
import json
import os
import argparse
from tqdm import tqdm

def prepare_model(whisper_path, gpu_id):
    device = torch.device('cuda:{}'.format(gpu_id))
    model = WhisperForConditionalGeneration.from_pretrained(whisper_path).to(device)
    processor = WhisperProcessor.from_pretrained(whisper_path)
    return model, processor, device

class AudioDataset(Dataset):
    def __init__(self, annotation, audio_dir, whisper_path):
        with open(annotation, 'r') as f:
            self.data = json.load(f)
        self.data = list(self.data.items())
        self.processor = SpeechExtractor(whisper_path)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        video_id, _ = self.data[index]
        video_path = os.path.join(self.audio_dir, '{}.wav'.format(video_id))
        sample = {'video': video_path}
        spectrogram = self.processor.extract(sample)
        if spectrogram is None:
            return None
        return spectrogram, video_id


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), []
    spectrogram, video_ids = zip(*batch)
    spectrogram = torch.cat(spectrogram, dim=0)
    video_ids = list(video_ids)
    return spectrogram, video_ids

def create_data_loader(annotation, audio_dir, whisper_path, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = AudioDataset(annotation, audio_dir, whisper_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str, default="./data/longvale-annotations-eval.json")
    parser.add_argument("--audio_dir", type=str, default="./data/raw_data/audio_test")
    parser.add_argument("--save_dir", type=str, default="./data/features_eval/speech_asr")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/openai/whisper-large-v2")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model, processor, device = prepare_model(args.checkpoint, args.gpu_id)
    data_loader = create_data_loader(args.annotation, args.audio_dir, args.checkpoint)

    with torch.no_grad():
        for (spectrogram, video_ids) in tqdm(data_loader):
            if spectrogram.numel() == 0: 
                continue
            video_id = video_ids[0]
            save_path = os.path.join(args.save_dir, '{}.json'.format(video_id))
            
            if os.path.exists(save_path):
                continue
            
            spectrogram = spectrogram.to(device)
            if spectrogram.dim() == 2:  
                spectrogram = spectrogram.unsqueeze(0)  
            
            transcriptions = model.generate(input_features=spectrogram)
            transcriptions = processor.batch_decode(transcriptions, skip_special_tokens=True)
                
            full_transcription = " ".join(transcriptions).strip()

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({"video_id": video_id, "transcription": full_transcription}, f, ensure_ascii=False, indent=2)
