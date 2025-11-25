import os 
import json 
import argparse
import numpy as np 
import utils
import node 

def main(args): 
    result = {}
    js = json.load(open(args.data_path))
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    for video_id, data in js.items():
        duration = data['duration']
        
        video_path = os.path.join(args.video_feat_folder, f"{video_id}.npy")
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
    
        features = utils.load_avs_features(
            video_id=video_id,
            video_path=args.video_feat_folder,
            audio_path=args.audio_feat_folder,
            speech_path=args.speech_feat_folder
        )
        
        avs_features = np.concatenate([
            features['video_features'],
            features['audio_features'],
            features['speech_features']
        ], axis=1)
        
        root_node = node.EventNode(
            video_id=video_id,
            start_time=0.0,
            end_time=duration,
            level=0,
        )

        if args.split_method == 'kmeans':
            node.build_tree_top_down(
                features=avs_features,
                parent_node=root_node,
                args=args
            )
        elif args.split_method == 'twfinch':
            node.build_tree_bottom_up(
                features=avs_features,
                root_node=root_node,
                args=args
            )
    
        def node_to_dict(node):
            start_time = node.start_time - 0.5 if node.start_time > 0.0 else node.start_time
            end_time = node.end_time + 0.5 if node.end_time < duration else node.end_time

            return {
                'level': node.level,
                'start_time': start_time,
                'end_time': end_time,
                'children': [node_to_dict(child) for child in node.children]
            }
            
        root_video_id = root_node.video_id
        root_content = {
            'level': root_node.level,
            'start_time': root_node.start_time,
            'end_time': root_node.end_time,
            'children': [node_to_dict(child) for child in root_node.children]
        }

        result[root_video_id] = root_content

    with open(args.save_path, 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the JSON data file')
    parser.add_argument('--video_feat_folder', type=str, default='./data/features_tree/video_features')
    parser.add_argument('--audio_feat_folder', type=str, default='./data/features_tree/audio_features')
    parser.add_argument('--speech_feat_folder', type=str, default='./data/features_tree/speech_features')
    parser.add_argument('--save_path', type=str, default='./outputs/tree.json')
    parser.add_argument('--split_method', type=str, choices=['kmeans', 'twfinch'], default='twfinch', help='Method to split segments')
    parser.add_argument('--split_n_clusters', type=int, default=3, help='Number of clusters for kmeans splitting')
    parser.add_argument('--max_depth', type=lambda x: None if str(x).lower() == 'none' else int(x), default=None, help='Maximum depth of the video tree ')
    parser.add_argument('--min_segment_length', type=float, default=3.0, help='Minimum length of a segment to consider splitting')
    args = parser.parse_args()
    
    main(args)