import json
import pandas as pd
import altair as alt

def load_and_flatten_data(files_to_load):
    all_data = []
    
    for file_name, result_type in files_to_load.items():
        data = json.load(open(file_name))
        for video_id, root_node in data.items():
            all_data.extend(flatten_tree_recursive(root_node, video_id, result_type))
            
    return pd.DataFrame(all_data)

def flatten_tree_recursive(node, video_id, result_type):
    results = []
    start_time = node.get('start_time', 0.0)
    end_time = node.get('end_time', node.get('duration')) 
    
    current_node_data = {
        'video_id': video_id,
        'result': result_type,
        'level': node['level'],
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time
    }
    results.append(current_node_data)
    for child_node in node.get('children', []):
        results.extend(flatten_tree_recursive(child_node, video_id, result_type))
            
    return results
    
def visualize_tree_structure(full_df, save_path='visualization.html'):
    video_ids = full_df['video_id'].unique().tolist()
    video_select = alt.selection_point(
        name='Video',
        fields=['video_id'],
        bind=alt.binding_select(options=video_ids, name="Video ID: ")
    )
    
    stacked_icicle = alt.Chart(full_df).mark_rect(
        stroke='black',
        strokeWidth=0.5
    ).encode(
        x=alt.X('start_time', 
                title='Time (s)',  
                axis=alt.Axis(
                    format='~s',
                    titleFontSize=20,       
                    titleFontWeight='bold', 
                    labelFontSize=14        
        )),
        x2=alt.X2('end_time'),
        y=alt.Y('level:O', title='Level (Depth)', axis=alt.Axis(labels=True, titleFontSize=20, titleFontWeight='bold', labelFontSize=14), scale=alt.Scale(reverse=False,)),
        color=alt.Color('level:N'),
        tooltip=[
            alt.Tooltip('video_id', title='Video ID'),
            alt.Tooltip('result', title='Result'),
            alt.Tooltip('level', title='Level'),
            alt.Tooltip('start_time', title='Start', format='.2f'),
            alt.Tooltip('end_time', title='End', format='.2f'),
            alt.Tooltip('duration', title='Duration', format='.2f')
        ],
        row=alt.Row('result:N', 
                    title='', 
                    header=alt.Header(
                        titleOrient="top", 
                        labelOrient="top",
                        labelFontSize=18,      
                        labelFontWeight='bold'  
        ))
    ).properties(
    ).properties(
    ).properties(
        width=1500,
        height=300 
    ).add_params(
        video_select
    ).transform_filter(
        video_select
    ).interactive()
    
    stacked_icicle.save(save_path)

if __name__ == "__main__":
    
    tree_results = {
        # Add tree results
        "./outputs/tree_1.json": 'Result 1',
        "./outputs/tree_2.json": 'Result 2',     
    }

    save_path = './outputs/visualization.html'
    
    tree_data = load_and_flatten_data(tree_results)
    visualize_tree_structure(tree_data, save_path)