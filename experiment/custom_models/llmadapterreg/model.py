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


class LLMAdapterReg(nn.Module):

    def __init__(
        self,
        num_num_features: int,
        cardinalities: list[int] = [],
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_embedding_type: str = 'plr',
        token_dim: int = 16,
        num_classes: int = 1,
        use_cls: bool = False, # TODO: use cls or mean pooling
        mlp_ratio: float = 1.0,
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
        self._build_backbone(model_name)
        self.reset_parameters()
        
    def _build_backbone(self, model_name):
        # load pretrained llm
        pretrained_llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            # dtype="auto",
            # device_map="auto",
        )

        # remove rope
        for layer in pretrained_llm.model.layers:
            attn_rope = layer.self_attn
            attn_no_rope = LLMAttentionNoRoPE(attn_rope.config, attn_rope.layer_idx)
            attn_no_rope.load_state_dict(attn_rope.state_dict())
            layer.self_attn = attn_no_rope

        self.backbone = nn.ModuleList(pretrained_llm.model.layers)

        del pretrained_llm

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.pos_embedding)
  
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.feature_tokenizer(x_num, x_cat) # (B, N, d_token)
            x = self.mlp_adapter(x) # (B, N, d_llm)
            x = x + self.pos_embedding

            for block in self.backbone:
                x = block(
                    x,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=(None, None),
                )
            x = x.mean(dim=1) # (B, N)
            x = self.output_proj(x)

        return x
