from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from transformers import AutoModelForCausalLM
from ..layer import (
    FeatureTokenizer, 
    OutputProj,
    LLMAttentionNoRoPE
)

LLM_DIM_MAPPING = {
    "Qwen/Qwen2.5-0.5B": 896, # 24 layers
    "meta-llama/Llama-3.2-1B": 2048, # 16 layers
}


class LLMAdapter(nn.Module):

    def __init__(
        self,
        num_num_features: int,
        cardinalities: list[int] = [],
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_embedding_type: str = 'plr',
        token_dim: int = 16,
        num_classes: int = 1,
        mlp_ratio: float = 1.0,
        use_cls: bool = False, # TODO: use cls or mean pooling
    ):  
        super().__init__()
        self.num_features = num_num_features + len(cardinalities) # num columns
        self.llm_dim = LLM_DIM_MAPPING[model_name]
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
        self.pos_embedding = nn.Parameter(torch.empty(1, self.num_features, self.llm_dim))
        self.output_proj = OutputProj(self.llm_dim, num_classes, mlp_ratio)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.pos_embedding)
  
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.feature_tokenizer(x_num, x_cat) # (B, N, d_token)
            x = self.mlp_adapter(x) # (B, N, d_llm)
            outputs = self.backbone.model(
                inputs_embeds=x,
                # attention_mask=None,
            )
            pred_hidden = outputs.last_hidden_state.mean(dim=1) # (B, D)
            logits = self.output_proj(pred_hidden)

        return logits