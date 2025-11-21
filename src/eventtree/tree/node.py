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

def build_tree(
    features: np.ndarray, 
    root_node: EventNode,  
): 
    hier_segments = split.find_split_points(
        features,
        root_node.start_time,
        root_node.end_time,
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