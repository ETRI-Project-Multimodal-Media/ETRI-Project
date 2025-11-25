import numpy as np
from sklearn.cluster import KMeans
from twfinch.python.twfinch import FINCH 


def find_split_points(
    features: np.ndarray, 
    start_time: float, 
    end_time: float, 
    args
): 
    if args.split_method == 'kmeans':
        split_indices = split_by_kmeans(features, args)
        return split_flat_segments(split_indices, start_time, end_time, args)
    
    elif args.split_method == 'twfinch':
        hier_split_indices = split_by_twfinch(features)
        return split_hier_segments(hier_split_indices, start_time, end_time)
    
def split_by_kmeans(
    features: np.ndarray, 
    args
):
    if (features.shape[0] < args.split_n_clusters) or (np.unique(features, axis=0).shape[0] < args.split_n_clusters):
        return []
    
    kmeans = KMeans(n_clusters=args.split_n_clusters, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(features)
    return np.where(labels[1:] != labels[:-1])[0]

def split_flat_segments(
    split_indices: np.ndarray, 
    start_time: float, 
    end_time: float, 
    args
):
    if len(split_indices) == 0:
        return [] 
    
    segments = [] 
    current_start = start_time
    
    for split_idx in split_indices:
        split_time = start_time + float(split_idx) 
        if (end_time - split_time) < args.min_segment_length:
            break 
        
        if (split_time - current_start) >= args.min_segment_length:
            segments.append([current_start, split_time])
            current_start = split_time + 1
            
    if (end_time - current_start) > 0:
        segments.append([current_start, end_time])
        
    return segments if len(segments) > 1 else []

def split_by_twfinch(features: np.ndarray):
    c_, num_clust, _ = FINCH(features, tw_finch=True, verbose=False)
    if c_.shape[1] == 0 or num_clust[0] <= 1: 
        return []
    
    hier_split_indices = []
    num_hier_levels = c_.shape[1]
    
    for i in range(num_hier_levels-1, -1, -1):  
        labels = c_[:, i]
        split_indices = np.where(labels[1:] != labels[:-1])[0]
        hier_split_indices.append(sorted(split_indices))
    return hier_split_indices

def split_hier_segments(
    hier_split_indices: np.ndarray, 
    start_time: float, 
    end_time: float
):
    if not hier_split_indices:
        return []
    hier_segments = []

    for split_indices in hier_split_indices:
        segments = []
        current_start = start_time
        
        for split_idx in split_indices:   
            split_time = start_time + float(split_idx)
            segments.append([current_start, split_time])
            current_start = split_time + 1
            
        if (end_time - current_start) > 0:
            segments.append([current_start, end_time])
        hier_segments.append(segments)     
    return hier_segments