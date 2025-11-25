import os
import sys
sys.path.append('./')
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from longvalellm.model.builder import load_pretrained_model
from longvalellm.utils import disable_torch_init
from longvalellm.inference import inference
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

def time_to_frame_index(duration, time):
    time = time / duration * 100
    frame_idx = str(min(round(time), 99))
    if len(frame_idx) == 1:
        frame_idx = "0" + frame_idx
    return frame_idx

def segment_caption_node(
    node, model, tokenizer, data, 
    video_features, audio_features, asr_features, question
):
    start_time = node['start_time']
    end_time = node['end_time']
    duration = data['end_time']
    
    start_frame = time_to_frame_index(duration, start_time)
    end_frame = time_to_frame_index(duration, end_time)
    
    query = question.replace('<start>', start_frame).replace('<end>', end_frame)
    answer = inference(model, video_features, audio_features, asr_features, "<video>\n " + query, tokenizer)
    
    node['caption'] = answer.strip()
    
def caption_leaf_nodes(
    node, model, tokenizer, data, 
    video_features, audio_features, asr_features, question
):
    if not node['children']:
        segment_caption_node(
            node, model, tokenizer, data, 
            video_features, audio_features, asr_features, question
        )
        return 
    
    for child in node['children']:
        caption_leaf_nodes(
            child, model, tokenizer, data, 
            video_features, audio_features, asr_features, question
        )

def get_max_depth(node):
    if not node['children']:
        return node['level']

    return max([get_max_depth(child) for child in node['children']])
    
def find_nodes_at_max_depth(node, depth, node_list):
    if not node['children']:
        return 
    
    if node['children'] and node['children'][0]['level'] == depth:
        node_list.append(node)
        return
    
    for child in node['children']:
        find_nodes_at_max_depth(child, depth, node_list)
 
def refine_leaf_nodes(
    root_node, model, tokenizer, sim_model, data, 
    video_features, audio_features, asr_features, 
    question, similarity_threshold=0.9
):
    max_depth = get_max_depth(root_node)
    nodes_to_refine = []
    find_nodes_at_max_depth(root_node, max_depth, nodes_to_refine)
    
    for parent_node in nodes_to_refine: 
        childrens = parent_node['children']

        new_groups = [] 
        current_group = [childrens[0]]

        for i in range(len(childrens)-1):  
            caption1 = childrens[i]['caption']
            caption2 = childrens[i+1]['caption']

            similarity = caption_similarity(sim_model, caption1, caption2)
            if similarity >= similarity_threshold:
                current_group.append(childrens[i+1])
            else:
                new_groups.append(current_group)
                current_group = [childrens[i+1]]
                
        new_groups.append(current_group)
        if len(new_groups) == len(childrens):
            continue
        
        if len(new_groups) == 1:
            segment_caption_node(
                parent_node, model, tokenizer, data, 
                video_features, audio_features, asr_features, question
            )
            parent_node['children'] = [] 
        else: 
            new_children = []
            for group in new_groups:
                if len(group) == 1:
                    new_children.append(group[0])
                    continue
                
                merged_node = {
                    'level': group[0]['level'],
                    'start_time': group[0]['start_time'],
                    'end_time': group[-1]['end_time'],
                    'children': [],
                }
                
                segment_caption_node(
                    merged_node, model, tokenizer, data, 
                    video_features, audio_features, asr_features, question
                )
                
                new_children.append(merged_node)
            parent_node['children'] = new_children   
    
def caption_similarity(sim_model, caption1, caption2):
    with torch.inference_mode():
        embeddings = sim_model.encode(
            [caption1, caption2],
            convert_to_tensor=True,
            show_progress_bar=False,
        )
    cosine_similarity = cos_sim(embeddings[0], embeddings[1])
    return cosine_similarity.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_path", type=str, default=f"./outputs/tree.json")
    parser.add_argument("--prompt_path", type=str, default=f"./data/caption_prompt.json")
    parser.add_argument("--save_path", type=str, default=f"./outputs/tree_caption.json")
    parser.add_argument("--video_feat_folder", type=str, default="./data/features_eval/video_features_1171")
    parser.add_argument("--audio_feat_folder", type=str, default="./data/features_eval/audio_features_1171")
    parser.add_argument("--asr_feat_folder", type=str, default="./data/features_eval/speech_features_1171") 
    parser.add_argument("--model_base", type=str, default="./checkpoints/vicuna-7b-v1.5")
    parser.add_argument("--stage2", type=str, default="./checkpoints/longvale-vicuna-v1-5-7b-stage2-bp")
    parser.add_argument("--stage3", type=str, default="./checkpoints/longvale-vicuna-v1-5-7b-stage3-it")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="./checkpoints/vtimellm_stage1_mm_projector.bin")
    parser.add_argument("--pretrain_audio_mlp_adapter", type=str, default=None)
    parser.add_argument("--pretrain_asr_mlp_adapter", type=str, default=None)
    parser.add_argument("--similarity_threshold", type=float, default=0.9)
    args = parser.parse_args()
    
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.to('cuda')
    model.to(torch.float16)
    
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    sim_model = sim_model.to('cuda')
    sim_model.to(torch.float16)
    
    tree_data = json.load(open(args.tree_path))
    prompt_data = json.load(open(args.prompt_path))
    
    question = prompt_data["question"]
    
    for video_id, data in tqdm(tree_data.items()):       
        video_features, audio_features, asr_features = None, None, None
        
        if args.video_feat_folder is not None:
            video_feat_path = os.path.join(args.video_feat_folder, f"{video_id}.npy")
            if os.path.isfile(video_feat_path):
                video_features = torch.from_numpy(np.load(video_feat_path)).cuda()
        
        if args.audio_feat_folder is not None:
            audio_feat_path = os.path.join(args.audio_feat_folder, f"{video_id}.npy")
            if os.path.isfile(audio_feat_path):
                audio_features = torch.from_numpy(np.load(audio_feat_path)).cuda()
        
        if args.asr_feat_folder is not None:
            asr_feat_path = os.path.join(args.asr_feat_folder, f"{video_id}.npy")
            if os.path.isfile(asr_feat_path):
                asr_features = torch.from_numpy(np.load(asr_feat_path)).cuda()

        if video_features is None:
            print(f'Can not find video {video_id}')
            continue
        
        root_node = tree_data[video_id]
        
        caption_leaf_nodes(
            root_node, model, tokenizer, data,  
            video_features, audio_features, asr_features, question
        )
        
        refine_leaf_nodes(
            root_node, model, tokenizer, sim_model, data, 
            video_features, audio_features, asr_features, 
            question, similarity_threshold=args.similarity_threshold
        )

    with open(args.save_path, 'w') as f:
        json.dump(tree_data, f, indent=4)
