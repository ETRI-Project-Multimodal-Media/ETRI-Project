import torch
import torch.nn as nn
from longvalellm.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from abc import ABC, abstractmethod
import os
import torch.nn.functional as F
import numpy as np

# DyCoke 
def dycole_ttm(image_feature, num_tokens_per_frame = 100, merging_ratio = 0.7):
    # Split frames into tokens
    num_frames = image_feature.shape[0] // num_tokens_per_frame
    merging_ratio = 1 - merging_ratio
    # Calculate similarities between adjacent even frames
    similarities = []
    for i in range(0, num_frames - 1, 2):
        # Get tokens for adjacent frames
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]

        # Calculate cosine similarity between normalized tokens
        frame1_norm = torch.nn.functional.normalize(frame1_tokens, p=2, dim=1)
        frame2_norm = torch.nn.functional.normalize(frame2_tokens, p=2, dim=1)
        similarity = torch.nn.functional.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity)

    similarities = torch.stack([torch.tensor(similarity) for similarity in similarities])

    # Process even frames
    modified_image_feature = []
    for i in range(0, num_frames - 1, 2):
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]
        
        avg_similarity = similarities[i // 2]
        num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices
        
        modified_image_feature.append(frame1_tokens)
        modified_image_feature.append(frame2_tokens[tokens_to_keep])

    # Process odd frames
    odd_similarities = []
    for i in range(0, num_frames - 4, 4):
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 2) * num_tokens_per_frame: (i + 3) * num_tokens_per_frame]
        
        similarity = torch.nn.functional.cosine_similarity(frame1_tokens, frame2_tokens, dim=1)
        odd_similarities.append(similarity)

    odd_similarities = torch.stack([torch.tensor(similarity) for similarity in odd_similarities])

    for i in range(0, num_frames - 4, 4):
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 2) * num_tokens_per_frame: (i + 3) * num_tokens_per_frame]
        
        avg_similarity = odd_similarities[i // 4]
        num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices
        
        modified_image_feature[i] = frame1_tokens
        modified_image_feature[i + 2] = frame2_tokens[tokens_to_keep]

    # Combine all tokens
    combined_tokens = torch.cat(modified_image_feature, dim=0)
    return combined_tokens



class DyLongVALELLMMetaModel:

    def initialize_vision_modules(self, model_args):
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(768, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print("load visual mlp:", pretrain_mm_mlp_adapter)


    def initialize_audio_modules(self, model_args):
        pretrain_audio_mlp_adapter = model_args.pretrain_audio_mlp_adapter
        pretrain_asr_mlp_adapter = model_args.pretrain_asr_mlp_adapter

        if not hasattr(self, 'audio_mm_projector'):
            self.audio_mm_projector = nn.Linear(768, self.config.hidden_size)
        if not hasattr(self, 'asr_mm_projector'):
            self.asr_mm_projector = nn.Linear(1280, self.config.hidden_size)

        if pretrain_audio_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_audio_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.audio_mm_projector.load_state_dict(get_w(mm_projector_weights, 'audio_mm_projector'))
            print("load audio mlp:", pretrain_audio_mlp_adapter)

        if pretrain_asr_mlp_adapter is not None: # if path exists, load weights for adapter
            mm_projector_weights = torch.load(pretrain_asr_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.asr_mm_projector.load_state_dict(get_w(mm_projector_weights, 'asr_mm_projector'))
            print("load mlp:", pretrain_asr_mlp_adapter)

class DyLongVALELLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, audio=None, asr=None
    ):
        
        ## image or audio None  / one token step  -> return text input
        ##  original llama
        if (images is None and audio is None) or input_ids.shape[1] == 1:
            if past_key_values is not None and (images is not None or audio is not None) and input_ids.shape[1] == 1:
                if self.get_model().config.model_type == 'chatglm':
                    target_shape = past_key_values[-1][-1].shape[0] + 1
                # DynamicCache(ver 4.57)
                elif hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
                    last_kv = past_key_values.key_cache[-1]
                    target_shape = last_kv.shape[-2] + 1
                # 구버전 (tuple of tuples)
                else:
                    target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if audio is not None:
            if type(audio) is list:
                # WARN all samples must have audio input
                audio_features = [self.get_model().audio_mm_projector(a) if a is not None else None for a in audio]
            else:
                audio_features = self.get_model().audio_mm_projector(audio.to(torch.float16)) ##改过

        if asr is not None:
            if type(asr) is list:
                asr_features = [self.get_model().asr_mm_projector(a) if a is not None else None for a in asr]
            else:
                asr_features = self.get_model().asr_mm_projector(asr.to(torch.float16)) ##改过

        if type(images) is list: # if input is batch
            concat_images = torch.cat([image for image in images], dim=0) # concat
            image_features = self.get_model().mm_projector(concat_images) # project
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0) # split again
            # image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.get_model().mm_projector(images)
        # print([image.shape for image in image_features])

        concated_features = [] # feature concatenation
        # concat for same frame
        for (audio_feat, image_feat, asr_feat) in zip(audio_features, image_features, asr_features):    
            assert not (audio_feat == None and image_feat == None and asr_features == None) 
            # self.dycoke == True
            # self.dycoke_num_tokens_per_frame , self.dycoke_p 사용
            if getattr(self, "dycoke", False):
                image_feat = dycole_ttm(
                    image_feat, 
                    num_tokens_per_frame = self.dycoke_num_tokens_per_frame,
                    keep_ratio = self.dycoke_p
                )
            # save image token length (for KV Pruning)
            if hasattr(self, "model") and hasattr(self.model, "DycokeConfig"):
                self.model.DycokeConfig.image_token_length = image_feat.size(0) # check
            concat_feat = []
            if image_feat is not None:
                concat_feat.append(image_feat) 
            if audio_feat is not None:
                concat_feat.append(audio_feat)
            if asr_feat is not None:
                concat_feat.append(asr_feat)

            concat_feat = torch.cat(concat_feat, dim=-2) # [133,4096] every modality is projected to 4096 channels
            concated_features.append(concat_feat)

        image_features = concated_features # replace image_features to concated_features
        
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        ## output new input embeddings (replace image token to real image embedding)
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() # num of [IMAGE] in a sample
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]]) # only text prompt index
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) # only text embeddings
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i]) # append text embeddings
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features) # append image embeddings
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds) # concat text + image embeddings
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        ## cut embeddings/labels to max_length
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        ## padding to make same size for all batch samples 
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                # concat zero to left 
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                # concat zero to right
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None


        if self.get_model().config.model_type == 'chatglm':
            fake_input_ids = torch.full((new_input_embeds.shape[0], new_input_embeds.shape[1]), -10000, 
                                        dtype=new_input_embeds.dtype, device=new_input_embeds.device)
            attention_mask = attention_mask.to(torch.int8)
            new_input_embeds = new_input_embeds.transpose(0, 1).contiguous()
        else:
            fake_input_ids = None
        # print(position_ids, attention_mask)
        return fake_input_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
