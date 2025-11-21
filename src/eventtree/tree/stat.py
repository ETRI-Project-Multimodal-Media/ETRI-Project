import json 
import argparse
import numpy as np 

def analyze_tree_statistics(tree_path, save_path):
    with open(tree_path, 'r') as f:
        tree_data = json.load(f)

    def get_nodes_and_max_depth(node):
        nodes = [node]
        max_depth = node.get('level', 0)
        for child in node.get('children', []):
            child_nodes, child_max_depth = get_nodes_and_max_depth(child)
            nodes.extend(child_nodes)
            max_depth = max(max_depth, child_max_depth)
        return nodes, max_depth

    videos_by_max_depth = {}
    for video_id, root_node in tree_data.items():
        all_nodes, max_depth = get_nodes_and_max_depth(root_node)
        video_length = root_node['end_time'] - root_node['start_time']
        
        if max_depth not in videos_by_max_depth:
            videos_by_max_depth[max_depth] = []
        
        videos_by_max_depth[max_depth].append({
            'video_id': video_id,
            'length': video_length,
            'nodes': all_nodes
        })

    stats_results = {}
    for max_depth, videos in sorted(videos_by_max_depth.items()):
        num_videos = len(videos)
        avg_video_length = np.mean([v['length'] for v in videos])
        
        nodes_by_level = {}
        for video in videos:
            for node in video['nodes']:
                level = node['level']
                duration = node['end_time'] - node['start_time']
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(duration)
        
        level_wise_stats = {}
        for level, durations in sorted(nodes_by_level.items()):
            total_nodes_in_level = len(durations)
            avg_nodes_per_video = total_nodes_in_level / num_videos
            std_duration = np.std(durations)
            avg_duration = np.mean(durations)
            level_wise_stats[level] = {
                'mean_nodes_per_video': round(avg_nodes_per_video, 2),
                'mean_duration': round(avg_duration, 2),
                'std_duration': round(std_duration, 2)
                
            }
                
        stats_results[max_depth] = {
            'num_videos': num_videos,
            'mean_video_length': round(avg_video_length, 2),
            'level_wise_statistics': level_wise_stats
        }

    with open(save_path, 'w') as f:
        json.dump(stats_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_path", type=str, required=True, help="Path to the JSON file containing the tree structures.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the analysis results.")
    args = parser.parse_args()

    analyze_tree_statistics(
        tree_path=args.tree_path,
        save_path=args.save_path
    )
    