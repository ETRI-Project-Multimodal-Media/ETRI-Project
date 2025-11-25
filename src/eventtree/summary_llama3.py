import sys
sys.path.append('./')
import json
import argparse
import torch
from tqdm import tqdm
from transformers import pipeline

def caption_internal_nodes(node, model, prompts):
    if not node['children']:
        return 

    for child in node['children']:
        caption_internal_nodes(child, model, prompts)
    
    child_captions = "\n- ".join([child['caption'] for child in node['children']])
    node['caption'] = summary_captions(model, prompts, child_captions)

def summary_captions(model, prompts, captions):      
    user_part = f"Captions:\n- {captions}\n\nSummary:"
    messages = prompts + [{"role": "user", "content": user_part}]
    
    inputs = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    with torch.inference_mode():
        output = model(
            inputs,
            max_new_tokens=120, 
            min_length=10,      
            do_sample=False,   
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.eos_token_id 
        )
    
    summary = output[0]['generated_text']
    summary = summary[len(inputs):].strip()
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_path", type=str, default="./outputs/tree_caption.json")
    parser.add_argument("--prompt_path", type=str, default="./data/caption_prompt.json")
    parser.add_argument("--save_path", type=str, default="./outputs/tree_structured_data.json")
    args = parser.parse_args()
    
    model = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16}, 
        device='cuda'
    )
    
    tree_data = json.load(open(args.tree_path))
    prompt_data = json.load(open(args.prompt_path))
    
    prompt = prompt_data["summary"]
    examples = prompt_data["summary_shots"]
    
    prompts = [{"role": "system", "content": prompt.strip()}]
    bos_token = model.tokenizer.bos_token
    eos_token = model.tokenizer.eos_token
    
    for example in examples:
        example_captions_str = "- " + "\n- ".join([c.strip() for c in example['input']])
        user_part = f"Captions:\n{example_captions_str}\n\nSummary:"
        assistant_part = f"{example['output']}"
        prompts.append({"role": "user", "content": user_part})
        prompts.append({"role": "assistant", "content": assistant_part})   
    
    for video_id, data in tqdm(tree_data.items()):      
        root_node = tree_data[video_id]
        caption_internal_nodes(root_node, model, prompts)
        
    with open(args.save_path, 'w') as f:
        json.dump(tree_data, f, indent=4)
