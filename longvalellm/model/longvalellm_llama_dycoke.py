# import torch
# import torch.nn as nn
# from typing import List, Optional, Tuple, Union
# from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
# from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
# # from .longvalellm_arch import LongVALELLMMetaModel, LongVALELLMMetaForCausalLM
# from .longvalellm_arch_dycoke import DyLongVALELLMMetaModel, DyLongVALELLMMetaForCausalLM

# # longvalellm_llama.py (상단 import에 추가)
# from transformers.cache_utils import DynamicCache, Cache, StaticCache
# import torch.nn.functional as F
# from typing import List, Optional, Tuple, Union
# from transformers.utils import (
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     is_flash_attn_2_available,
#     is_flash_attn_greater_or_equal_2_10,
#     logging,
#     replace_return_docstrings,
# )

# logger = logging.get_logger(__name__)

# # ---- DyCoke 설정 컨테이너 ----
# class DycokeConfigs:
#     def __init__(self):
#         self.dycoke_layer_idx: int = 3         # DyCoke를 적용할 layer index (0-based)
#         self.dycoke_radio: float = 0.8         # keep ratio (이미지 토큰 유지 비율)
#         self.image_token_start_index: int = 0  # 이미지 시작 위치(프리필에서 확정)
#         self.image_token_length: Optional[int] = None  # 이미지 token 개수 (arch에서 채움)
#         self.similarity: Optional[torch.Tensor] = None
#         self.attention_score: Optional[torch.Tensor] = None
#         self.seq_length_with_past: Optional[int] = None

# class DycokeSubConfig:
#     def __init__(self, base_config, start_idx, img_len):
#         self.image_token_start_index = start_idx
#         self.image_token_length = img_len
#         self.seq_length_with_past = base_config.seq_length_with_past
#         self.dycoke_radio = base_config.dycoke_radio

# # ---- KV 캐시를 프루닝할 수 있게 확장 ----
# class PrunableDynamicCache(DynamicCache):
#     """
#     - layer별 key/value 캐시 유지
#     - self.kv_cache: 유지할 token 인덱스(list[int]); None이면 전체 유지
#     """
#     def __init__(self, config=None) -> None:
#         super().__init__()
#         self.key_cache: List[torch.Tensor] = []
#         self.value_cache: List[torch.Tensor] = []
#         self._seen_tokens = 0
#         self.kv_cache = None  # list[int]
#     # 모든 layer에서 attention을 수행할때 계속 update
#     # return 값 = attention에서 사용되는 key, value
#     def update(
#         self,
#         key_states: torch.Tensor,    # (B, H_kv, T_new, D)
#         value_states: torch.Tensor,  # (B, H_kv, T_new, D)
#         layer_idx: int,
#         cache_kwargs=None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if layer_idx == 0:
#             # 디코딩 시 생성한 token의 개수
#             self._seen_tokens += key_states.shape[-2]

#         # Update the cache
#         ## 첫 decoding step에서는 모두 append
#         if len(self.key_cache) <= layer_idx:
#             self.key_cache.append(key_states)
#             self.value_cache.append(value_states)
#         else:
#             if key_states.shape[-2] > 1: # decoding 할때 keystates가 1이 아니라서 맨 마지막만 가져와서 cache에 붙여줌 
#                 key_states = key_states[..., -1:, :]
#                 value_states = value_states[..., -1:, :]
#             self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
#             self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

#         if self.kv_cache is None:
#             return self.key_cache[layer_idx], self.value_cache[layer_idx]

#         K_index = torch.tensor(self.kv_cache, device=self.key_cache[layer_idx].device).view(1, 1, -1, 1).expand(
#             self.key_cache[layer_idx].size(0),
#             self.key_cache[layer_idx].size(1),
#             -1,
#             self.key_cache[layer_idx].size(3),
#         )
#         K_used = torch.gather(self.key_cache[layer_idx], dim=2, index=K_index)

#         V_index = torch.tensor(self.kv_cache, device=self.value_cache[layer_idx].device).view(1, 1, -1, 1).expand(
#             self.value_cache[layer_idx].size(0),
#             self.value_cache[layer_idx].size(1),
#             -1,
#             self.value_cache[layer_idx].size(3),
#         )
#         V_used = torch.gather(self.value_cache[layer_idx], dim=2, index=V_index)
#         return K_used, V_used

#     # kv_cache에 attention 값이 높은 index를 return
#     def update_cache(self, image_attention, config):
#         # Pre-calculate values to avoid repeated computation
#         start_idx = config.image_token_start_index
#         img_len = config.image_token_length
#         num_keep = int(img_len * (1 - config.dycoke_radio)) # dycoke_ratio = 삭제할 비율 # num_keep : 유지할 token 수 
#         if num_keep <= 0:
#             num_keep = 1

#         # Get top indices in one operation
#         top_indices = torch.topk(image_attention, num_keep, sorted=False)[1] + start_idx
        
#         # Create ranges efficiently using single arange call
#         full_range = torch.arange(config.seq_length_with_past, device=image_attention.device) # prefill + 생성된 길이
#         keep_indices = torch.cat([
#             full_range[:start_idx],
#             top_indices,
#             full_range[start_idx + img_len:]
#         ]) # text token + 이미지 pruning index + text token

#         return keep_indices
    
#     # dycoke pruning 조건
#     # 이전 iteration에서 L번째 layer의 attention과의 유사도 높으면 =>  update_cache => kv cache update
#     def dycoke_pruning(self, attn, layer_idx, config):
#         # 구버전, decoder output[1] = attention 이었을때의 code 
#         # attention_avg : [B, H, L_q, L_k]
#         attention_avg = attn[1].mean(1)[0, -1] # 마지막 query token = 지금 막 생성된 token에 높은 중요도를 주는 key token을 추출

#         device = attention_avg.device
#         all_keep_indices = []        
#         # revised 
#         for start_idx, img_len in zip(config.image_token_start_index, config.image_token_length):
#             image_attention = attention_avg[start_idx:start_idx + img_len]
#             if config.attention_score is not None:
#                 config.similarity = F.cosine_similarity(
#                     image_attention, 
#                     config.attention_score[start_idx:start_idx + img_len],
#                     dim=0
#                 )
#             else:
#                 config.similarity = 0
            
#             # update stored attention
#             if config.attention_score is None:
#                 config.attention_score = torch.zeros_like(attention_avg)
#             config.attention_score[start_idx:start_idx + img_len] = image_attention
        
                
#             # 조건 만족 시 pruning
#             if config.similarity < 0.9:
#                 keep_indices = self.update_cache(image_attention, 
#                                                 config=DycokeSubConfig(config, start_idx, img_len))
#                 all_keep_indices.append(keep_indices)

#         # 여러 블록에서 얻은 keep_indices 병합
#         if len(all_keep_indices) > 0:
#             merged = torch.unique(torch.cat(all_keep_indices)).sort()[0]
#         else:
#             merged = torch.arange(config.seq_length_with_past, device=device)

#         # 리스트로 저장
#         self.kv_cache = merged.tolist()
#         return merged
    
# class LongVALELLMConfig(LlamaConfig): # inherit llama and register  "LongVALE-LLM" model # used for AutoConfig.from_pretrained()
#     model_type = "LongVALE-LLM"

    
# class DyLongVALELLMConfig(LongVALELLMConfig):
#     model_type = "LongVALE-LLM-DyCoke" # check
#     # --- DyCoke defaults ---
#     dycoke: bool = True            # 활성화 여부
#     dycoke_l: int = 3               # 적용 레이어 index (0-based)
#     dycoke_p: float = 0.8           # deletion ratio
#     dycoke_num_tokens_per_frame: int = 1
#     image_token_start_index: int = 0
#     output_attentions : bool = True
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.dycoke = kwargs.get("dycoke", True)
#         self.dycoke_l = kwargs.get("dycoke_l", 3)
#         self.dycoke_p = kwargs.get("dycoke_p", 0.8)
#         self.dycoke_num_tokens_per_frame = kwargs.get("dycoke_num_tokens_per_frame", 1)
#         self.image_token_start_index = kwargs.get("image_token_start_index", 0)

#         # 이 부분 반드시 __init__ 안에 넣어야 인스턴스 필드로 인식됨
#         self.output_attentions = True
    
# # revised by modelling_llama dycoke.
# class DyLongVALELLMLlamaModel(LlamaModel, DyLongVALELLMMetaModel): # inherit LlamaModel + LongVALELLMMetaModel(Custom Model)
#     config_class = DyLongVALELLMConfig # check

#     def __init__(self, config: DyLongVALELLMConfig):
#         super(DyLongVALELLMLlamaModel, self).__init__(config)
#         # DyCoke flags / configs
#         self.dycoke = getattr(config, "dycoke", True) # config from DyLongVALELLMConfig
#         self.dycoke_l = getattr(config, "dycoke_l", 3)
#         self.dycoke_p = getattr(config, "dycoke_p", 0.8)
#         self.dycoke_num_tokens_per_frame = getattr(config, "dycoke_num_tokens_per_frame", 1)
#         self.image_token_start_index = getattr(config, "image_token_start_index", 0)
#         # insert DyLongVALELLMConfig into config variable
#         self.DycokeConfig = DycokeConfigs() 
#         self.DycokeConfig.dycoke_layer_idx = self.dycoke_l
#         self.DycokeConfig.dycoke_radio = self.dycoke_p
#         self.DycokeConfig.image_token_start_index = self.image_token_start_index
        
#         self.output_attentions = True

#     def _apply_dycoke_mask(self, causal_mask, keep_indices):
#         if causal_mask is None or not torch.is_tensor(causal_mask):
#             return causal_mask
#         keep_tensor = keep_indices if torch.is_tensor(keep_indices) else torch.as_tensor(keep_indices, device=causal_mask.device)
#         keep_tensor = keep_tensor.to(device=causal_mask.device, dtype=torch.long)
#         if keep_tensor.numel() == causal_mask.shape[-1]:
#             return causal_mask
#         return causal_mask.index_select(-1, keep_tensor)

#     def _apply_dycoke_attention_mask(self, attention_mask, keep_indices):
#         keep_tensor = keep_indices if torch.is_tensor(keep_indices) else torch.as_tensor(keep_indices, device=attention_mask.device)
#         keep_tensor = keep_tensor.to(device=attention_mask.device, dtype=torch.long)
#         if keep_tensor.numel() == attention_mask.shape[1]:
#             return attention_mask
#         return attention_mask.index_select(1, keep_tensor)

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         cache_position: Optional[torch.LongTensor] = None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         **kwargs,
#     ):
#         output_attentions = output_attentions if output_attentions is not None and output_attentions is not False else self.config.output_attentions # changed
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError(
#                 "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
#             )
#         elif input_ids is not None:
#             batch_size, seq_length = input_ids.shape
#         elif inputs_embeds is not None:
#             batch_size, seq_length, _ = inputs_embeds.shape
#         else:
#             raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
#         if self.gradient_checkpointing and self.training and use_cache:
#             logger.warning_once(
#                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
#             )
#             use_cache = False

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         # past_key_values_length = 0 # decode output tokens length
#         past_seen_tokens= 0
#         if use_cache:  # kept for PrunableDynamicCache (cache positions)
#             use_legacy_cache = not isinstance(past_key_values, Cache)
#             if use_legacy_cache and past_key_values is None:
#                 past_key_values = PrunableDynamicCache.from_legacy_cache(past_key_values)
#             elif use_legacy_cache:
#                 past_key_values = PrunableDynamicCache.from_legacy_cache(past_key_values)
#             # before ver.
#             # past_key_values_length = past_key_values.get_usable_length(seq_length)
#             past_seen_tokens = past_key_values.get_seq_length()

#         # check when decoding 
#         if cache_position is None:
#             if isinstance(past_key_values, StaticCache):
#                 raise ValueError("cache_position is a required argument when using StaticCache.")
#             cache_position = torch.arange(
#                 past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#             )
#             # cache_position = torch.arange(
#             #     past_key_values_length, past_key_values_length + inputs_embeds.shape[1], device=inputs_embeds.device
#             # )

#         if position_ids is None:
#             position_ids = cache_position.unsqueeze(0)
#         # change position_ids for changed pruning
#         # if position_ids is None:
#         #     device = input_ids.device if input_ids is not None else inputs_embeds.device
#         #     position_ids = torch.arange(
#         #         past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
#         #     )
#         #     position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#         # else:
#         #     position_ids = position_ids.view(-1, seq_length).long()

#         causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

#         # embed positions
#         hidden_states = inputs_embeds

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = None

#         for layer_idx, decoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     decoder_layer.__call__,
#                     hidden_states,
#                     causal_mask,
#                     position_ids,
#                     past_key_values,
#                     output_attentions,
#                     use_cache,
#                     cache_position,
#                 )
#             else:
#                 if self.dycoke:
#                     self.DycokeConfig.seq_length_with_past = seq_length + past_seen_tokens
#                     if layer_idx < self.dycoke_l:
#                         past_key_values.kv_cache = None
#                     elif (
#                         self.dycoke
#                         and past_key_values.kv_cache is None
#                         and layer_idx == self.dycoke_l
#                         and position_ids.shape[1] == 1
#                     ):
#                         keep_indices = past_key_values.dycoke_pruning(layer_outputs, layer_idx, self.DycokeConfig)
#                         # if keep_indices is not None:
#                         #     causal_mask = self._apply_dycoke_mask(causal_mask, keep_indices)
#                         #     if attention_mask is not None and attention_mask.ndim == 2:
#                         #         causal_mask = self._apply_dycoke_attention_mask(attention_mask, keep_indices)
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=causal_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_values,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                     cache_position=cache_position,
#                 )
#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache = layer_outputs[2 if output_attentions else 1]

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#         hidden_states = self.norm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = None
#         if use_cache:
#             next_cache = (
#                 next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
#             )
#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )

    
# class DyLongVALELLMLlamaForCausalLM(LlamaForCausalLM, DyLongVALELLMMetaForCausalLM):  # inherit LlamaForCausalLM(Llama + LM head) + LongVALELLMMetaForCausalLM(Custom, prepare_inputs_labels_for_multimodal)
#     # LlamaForCausalLM used for generating text (forward, prepare_inputs_for_generation)
#     # LongVALELLMMetaForCausalLM used for embedding multimodal input (prepare_inputs_labels_for_multimodal)
#     config_class = DyLongVALELLMConfig 

#     def __init__(self, config):
#         super(LlamaForCausalLM, self).__init__(config) # if use super().__init__(config) then LlamaForCausalLM.__init__ (parent) / we want to set model (LongVALELLMLlamaModel/custom head) not (llamamodel/head)
#         self.model = DyLongVALELLMLlamaModel(config) # backbone = llama + multimodal 
#         self.pretraining_tp = config.pretraining_tp
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # lm head

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_model(self):
#         return self.model

#         # super().forward used by  LlamaForCausalLM  / well constructed 
#             ## inside of super().forward there is self.model which is LongVALELLMLlamaModel
#             ## so use LlamaForCausalLM's forward but change only model to LongVALELLMLlamaModel
#         # use LLM forward but make input_embeds(concat multimodal embedding)  and use it for inputs
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         images: Optional[torch.FloatTensor] = None,
#         audio = None,
#         asr = None,
#         cache_position: Optional[torch.LongTensor] = None, # added for transformers 4.43
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
         
#         if inputs_embeds is None:
#             (
#                 input_ids,
#                 position_ids,
#                 attention_mask,
#                 past_key_values,
#                 inputs_embeds,
#                 labels
#             ) = self.prepare_inputs_labels_for_multimodal(
#                 input_ids,
#                 position_ids,
#                 attention_mask,
#                 past_key_values,
#                 labels,
#                 images,
#                 audio=audio,
#                 asr=asr
#             ) # multimodal embedding - dycoke ttm

#         return super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             cache_position=cache_position, # added for transformers 4.43
#             return_dict=return_dict
#         ) # LlamaForCausalLM.forward() -> model = DyLongVALELLMLlamaModel

#     # prepare_inputs_for_generation -> prepare_inputs_labels_for_multimodal -> super().forward
#     def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
#         images = kwargs.pop("images", None)
#         audio = kwargs.pop("audio", None)
#         asr = kwargs.pop("asr", None)
#         _inputs = super().prepare_inputs_for_generation(
#             input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
#         )
#         if images is not None:
#             _inputs['images'] = images
#         if audio is not None:
#             _inputs["audio"] = audio
#         if asr is not None:
#             _inputs["asr"] = asr
#         return _inputs

# AutoConfig.register("LongVALE-LLM-DyCoke", DyLongVALELLMConfig) # add config 
# AutoModelForCausalLM.register(DyLongVALELLMConfig, DyLongVALELLMLlamaForCausalLM) # add model to config
