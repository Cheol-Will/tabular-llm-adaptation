from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from transformers import AutoModelForCausalLM
from ..layer import (
    FeatureTokenizer,
    OutputProj,
)


class LLMSlot(nn.Module):

    def __init__(
        self,
        num_num_features: int,
        cardinalities: list[int] = [],
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_embedding_type: str = 'plr',
        token_dim: int = 16,
        num_classes: int = 1,
        mlp_ratio: float = 1.0,
        use_bidir_attn: bool = False,
        prediction_method: str = 'next_token_pred',
    ):
        super().__init__()
        assert prediction_method in ['next_token_pred', 'token_pooling']

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16,
        )

        self.llm_dim = self.backbone.config.hidden_size
        self.feature_tokenizer = FeatureTokenizer(
            num_num_features,
            cardinalities,
            token_dim,
            num_embedding_type,
        )
        self.mlp_adapter = nn.Sequential(*[
            nn.Linear(token_dim, self.llm_dim // 4),
            nn.ReLU(),
            nn.Linear(self.llm_dim // 4, self.llm_dim),
            nn.LayerNorm(self.llm_dim)
        ])
        self.output_proj = OutputProj(self.llm_dim, num_classes, mlp_ratio)
        self.reset_parameters()

        self.use_bidir_attn = use_bidir_attn
        self.num_features = num_num_features + len(cardinalities)
        self.prediction_method = prediction_method

    def create_prompt(self, column_ids, column_ids_lengths):
        # column_ids: list of N+1 token-ID lists (N feature segs + 1 target seg)
        # column_ids_lengths: list of N+1 int lengths
        num_feature_cols = len(column_ids) - 1  # last segment is target, no feat slot after it

        dev = next(self.backbone.parameters()).device
        flat_ids = torch.tensor(
            [id_ for ids in column_ids for id_ in ids],
            dtype=torch.long,
            device=dev,
        )
        embed_layer = self.backbone.get_input_embeddings()
        with torch.no_grad():
            text_embeds = embed_layer(flat_ids).float()  # (total_text_tokens, d_llm)

        # Build boolean mask over the interleaved sequence:
        #   [seg0_tokens] [FEAT_0] [seg1_tokens] [FEAT_1] ... [segN-1_tokens] [FEAT_N-1] [segN_tokens]
        total_len = sum(column_ids_lengths) + num_feature_cols
        prompt_mask = torch.zeros(total_len, dtype=torch.bool, device=dev)
        pos = 0
        for i, length in enumerate(column_ids_lengths):
            pos += length
            if i < num_feature_cols:
                prompt_mask[pos] = True
                pos += 1

        self.prompt = nn.Parameter(text_embeds)
        self.register_buffer('prompt_mask', prompt_mask)

    def reset_parameters(self):
        pass

    def get_bidir_attn_mask(self, seq_len: int):
        return torch.full((1, 1, seq_len, seq_len), 0, dtype=torch.bool)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        """
        if self.use_bidir_attn, attention mask is filled with zeros (full attention).
        """
        B = x_num.shape[0]

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.feature_tokenizer(x_num, x_cat)  # (B, N, d_token)
            x = self.mlp_adapter(x)  # (B, N, d_llm)

            # Build interleaved sequence: text embeddings + MLP adapter outputs.
            # Must use torch.scatter (out-of-place, differentiable) rather than
            # in-place [:, mask] = ... assignment, which breaks the autograd graph:
            # x.new_zeros() creates a leaf with requires_grad=False, so in-place
            # writes from self.prompt / x are not tracked and gradients stop there.
            total_len = self.prompt_mask.shape[0]
            text_indices = (~self.prompt_mask).nonzero(as_tuple=True)[0]  # (T_text,)
            feat_indices = self.prompt_mask.nonzero(as_tuple=True)[0]    # (N,)

            text_idx_exp = text_indices.view(1, -1, 1).expand(B, -1, self.llm_dim)
            feat_idx_exp = feat_indices.view(1, -1, 1).expand(B, -1, self.llm_dim)
            prompt_exp = self.prompt.to(x.dtype).unsqueeze(0).expand(B, -1, -1)  # (B, T_text, d_llm)

            inputs_embeds = torch.scatter(
                x.new_zeros(B, total_len, self.llm_dim), 1, text_idx_exp, prompt_exp
            )
            inputs_embeds = torch.scatter(inputs_embeds, 1, feat_idx_exp, x)

            attention_mask = None
            if self.use_bidir_attn:
                attention_mask = self.get_bidir_attn_mask(total_len)
                attention_mask = attention_mask.expand(B, -1, -1, -1).to(x.device)

            outputs = self.backbone.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            
            if self.prediction_method == 'next_token_pred':
                pred_hidden = outputs.last_hidden_state[:,-1,:]  # (B, D)
            elif self.prediction_method == 'token_pooling':
                pred_hidden = outputs.last_hidden_state.mean(dim=1)  # (B, D)
            else:
                raise ValueError(f"Unknown prediction method: {self.prediction_method}")

            logits = self.output_proj(pred_hidden)

        return logits
