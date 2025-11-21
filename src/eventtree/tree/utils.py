import sys
sys.path.append('./')
import os 
import json 
import argparse
import numpy as np 
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def normalize_feature(feature, eps=1e-8):
    feature = feature.astype(np.float32)
    mean = np.mean(feature, axis=0, keepdims=True)
    std = np.std(feature, axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (feature - mean) / std

def load_avs_features(video_id, video_path, audio_path, speech_path):
    try:         
        video_features = np.load(os.path.join(video_path, f"{video_id}.npy"))
        audio_features = np.load(os.path.join(audio_path, f"{video_id}.npy"))
        speech_features = np.load(os.path.join(speech_path, f"{video_id}.npy"))
            
        video_features = normalize_feature(video_features)
        audio_features = normalize_feature(audio_features)
        speech_features = normalize_feature(speech_features)
        
        return {
            'video_id': video_id,
            'video_features': video_features,
            'audio_features': audio_features,
            'speech_features': speech_features
        }
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{video_id} video feature not found: {e}")

def split_video_segments(tree_path, video_dir, save_dir):
    with open(tree_path, 'r') as f:
        tree = json.load(f)
    os.makedirs(save_dir, exist_ok=True)
    
    def collect_node(node):
        nodes = []
        for child in node.get('children', []):
            nodes.append(child)
            nodes.extend(collect_node(child))
        return nodes

    for video_id, root in tqdm(tree.items()):       
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        video_save_dir = os.path.join(save_dir, "video_segments", video_id)
        audio_save_dir = os.path.join(save_dir, "audio_segments", video_id)
        
        os.makedirs(video_save_dir, exist_ok=True)
        os.makedirs(audio_save_dir, exist_ok=True)
        
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
            
        nodes = collect_node(root)
        try:
            with VideoFileClip(video_path) as video:
                for node in nodes: 
                    start_time = node['start_time']
                    end_time = node['end_time']
                    level = node.get('level')
                    
                    segment_clip = video.subclip(start_time, end_time)
                    
                    if segment_clip.audio: 
                        video_out_name = f"Level_{level}_{start_time:.2f}_to_{end_time:.2f}.mp4"
                        video_out_path = os.path.join(video_save_dir, video_out_name)
                        segment_clip.write_videofile(video_out_path, codec='libx264', audio_codec='aac', logger=None)
                        
                        audio_out_name = f"Level_{level}_{start_time:.2f}_to_{end_time:.2f}.wav"
                        audio_out_path = os.path.join(audio_save_dir, audio_out_name)
                        segment_clip.audio.write_audiofile(audio_out_path, fps=16000, codec='pcm_s16le', logger=None)
                    else:
                        print(f"No audio track found in segment: {video_id}, Level {level}, {start_time}-{end_time}")
                    
        except Exception as e:
            print(f"Failed to process video {video_id}: {e}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_path", type=str, required=True, help="Path to the JSON file containing the hierarchical tree structure.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing the video files.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the segmented video and audio files.")
    args = parser.parse_args()
    
    split_video_segments(
        tree_path=args.tree_path,
        video_dir=args.video_dir,
        save_dir=args.save_dir
    )