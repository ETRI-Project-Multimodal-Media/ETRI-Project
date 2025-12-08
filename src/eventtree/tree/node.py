import split 
import numpy as np 

class EventNode: 
    def __init__(
        self, 
        start_time: float, 
        end_time: float,
        level: int = 0, 
        video_id: str = None
    ): 
        self.video_id = video_id
        self.start_time = start_time
        self.end_time = end_time
        self.level = level 
        self.children = []      
        
    def get_duration(self):
        return self.end_time - self.start_time

    def add_child(self, node):
        self.children.append(node)
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def __repr__(self) -> str:
        return f"Node (video_id={self.video_id}, \
        timestamps=({self.start_time:.2f}, {self.end_time:.2f}), \
        level={self.level}, n_child={len(self.children)})"

# K-means Clustering 
def build_tree_top_down(
    features: np.ndarray, 
    parent_node: EventNode, 
    args
    ): 
    
    if args.max_depth is not None and parent_node.level >= args.max_depth:
        return 

    if parent_node.get_duration() < args.min_segment_length * 2: 
        return 

    start_idx = int(parent_node.start_time)
    end_idx = int(parent_node.end_time)

    segment_features = features[start_idx:end_idx+1]
    if segment_features.shape[0] < 2:
        return
    
    new_segments = split.find_split_points(
        segment_features,
        parent_node.start_time,
        parent_node.end_time,
        args
    )
    
    if new_segments:
        for (start, end) in new_segments:
            child_node = EventNode(
                start_time=start,
                end_time=end,
                level=parent_node.level+ 1,
                # video_id=current_node.video_id
            )
            parent_node.add_child(child_node)
            build_tree_top_down(features, child_node, args)

# TW-FINCH Clustering
def build_tree_bottom_up(
    features: np.ndarray, 
    root_node: EventNode,  
    args
): 
    hier_segments = split.find_split_points(
        features,
        root_node.start_time,
        root_node.end_time,
        args
    )
    
    def build_subtree(parent_node: EventNode): 
        if parent_node.level >= len(hier_segments):
            return
        
        for start, end in hier_segments[parent_node.level]:
            if start >= parent_node.start_time and end <= parent_node.end_time:
                if start == parent_node.start_time and end == parent_node.end_time:
                    continue
                
                child_node = EventNode(
                    start_time=start,
                    end_time=end,
                    level=parent_node.level+1,
                )
                parent_node.add_child(child_node)
                build_subtree(child_node)
                
    build_subtree(root_node)