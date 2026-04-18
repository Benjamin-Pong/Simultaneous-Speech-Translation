from transformers.models.seamless_m4t.modeling_seamless_m4t import (
    SeamlessM4TConformerEncoder,
    SeamlessM4TConformerSelfAttention
)
from transformers import TrainingArguments, AutoProcessor, SeamlessM4TModel , SeamlessM4TConfig, SeamlessM4TFeatureExtractor, SeamlessM4TTokenizer, SeamlessM4TProcessor
from transformers.modeling_utils import PreTrainedModel

from transformers.integrations import is_deepspeed_zero3_enabled, is_fsdp_managed_module
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput


import os
from datasets import Dataset, DatasetDict
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn

class SchedulerPerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scheduler_FNN = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size//8),
                                           nn.GELU(),
                                           nn.Linear(config.hidden_size//8, 1)
                                          )
        self.K_max_lookahead = 16
        self.epsilon = 1e-6

    def forward(self, hidden_states):

        o_score = self.scheduler_FNN(hidden_states)  #[batch_size, seq_len, 1]
        o_score = torch.sigmoid(o_score) #[batch_size, seq_len, 1]
        o_score = o_score * (self.K_max_lookahead + self.epsilon)  #scale to [0, K_max_lookahead]

        return o_score #[batch_size, seq_len, 1]



class SeamlessM4TConformerEncoderDynamicMasking(SeamlessM4TConformerEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.schedulers = nn.ModuleList([
            SchedulerPerLayer(config)
            for _ in range(config.speech_encoder_layers)
        ])
        self.tau = 0.5
        self.K_max_lookahead = 16

    def compute_s_score(self, idx, hidden_states):
        B, T, _ = hidden_states.shape
        device = hidden_states.device

        o_score = self.schedulers[idx](hidden_states)  # [B, T, 1]

        i_idx = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1)
        j_idx = torch.arange(T, device=device, dtype=torch.float32).view(1, 1, T)

        s = 1 - torch.sigmoid(((j_idx - i_idx) - o_score) / self.tau)
        s = torch.log(s.clamp(min=1e-9))              # needed for additive masking only

        #apply masking after computing s score to ensure that the model can compute attention weights for past tokens and some future tokens, but any tokens past max_lookahead will be masked out with -inf to ensure 0 attention weights after softmax.
        s = s.masked_fill(j_idx <= i_idx, 0.0) # Allow attending to current and past tokens
        s = s.masked_fill(j_idx > i_idx + self.K_max_lookahead, float('-inf'))

        num_heads = self.config.speech_encoder_attention_heads
        s = s.unsqueeze(1).expand(-1, num_heads, -1, -1)  # [B, H, T, T]
        return s

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # Keep original 1D bool mask for conv sublayer
        conv_attention_mask = attention_mask

        # Build 4D additive padding mask — mirrors original exactly
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(
                ~attention_mask.bool().unsqueeze(-1), 0.0
            )
            padding_mask_4d = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            padding_mask_4d = padding_mask_4d * torch.finfo(hidden_states.dtype).min
            padding_mask_4d = padding_mask_4d.expand(
                padding_mask_4d.shape[0], 1, padding_mask_4d.shape[-1], padding_mask_4d.shape[-1]
            )
        else:
            padding_mask_4d = None

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        prev_hidden = hidden_states.clone()

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            dropout_probability = torch.rand([])
            skip_the_layer = (
                self.training and dropout_probability < self.config.speech_encoder_layerdrop
            )

            if not skip_the_layer or synced_gpus:
                dynamic_mask = self.compute_s_score(i, prev_hidden)  # [B, H, T, T]

                combined_mask = (
                    dynamic_mask + padding_mask_4d
                    if padding_mask_4d is not None
                    else dynamic_mask
                )

                layer_outputs = layer(
                    hidden_states,
                    attention_mask=combined_mask,
                    relative_position_embeddings=relative_position_embeddings,  # ✅
                    output_attentions=output_attentions,
                    conv_attention_mask=conv_attention_mask,
                )

                prev_hidden = hidden_states
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class DynamicSeamlessM4T(SeamlessM4TModel):
    def __init__(self, config, current_modality="speech"):
        super().__init__(config, current_modality=current_modality) #this signals the model to use the speech encoder
        # Replace the encoder with custom encoder
        self.speech_encoder.encoder = SeamlessM4TConformerEncoderDynamicMasking(config)
