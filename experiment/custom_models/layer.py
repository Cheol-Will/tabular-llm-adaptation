from typing import Callable, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    eager_attention_forward,
)
from pytabkit.models.nn_models.rtdl_num_embeddings import (
    LinearEmbeddings,
    LinearReLUEmbeddings,
    PeriodicEmbeddings, 
    PiecewiseLinearEmbeddings,
)

class LLMAttentionNoRoPE(Qwen2Attention):
    """
    Attention operation of LLMs without RoPE.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        cache_position = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class NumEmbeddingLayer(nn.Module):
    """
    Numerical column embedding: (B, C) -> (B, C, D)
    """
    def __init__(
        self,
        num_num_features: int,
        num_embedding_type: str, # [linear, linear-relu, plr, ple]
        token_dim: int,
        bins = None,
        # *,
        # num_emb_sigma: float = 0.01,
        # n_frequencies: int = None,
        # num_emb_lite: bool = False
    ):
        super().__init__()
        assert num_num_features > 0

        if num_embedding_type == 'linear':
            self.num_embedding = LinearEmbeddings(num_num_features, token_dim)
        elif num_embedding_type == 'linear-relu':
            self.num_embedding = LinearReLUEmbeddings(num_num_features, token_dim)
            pass
        elif num_embedding_type == 'plr':
            self.num_embedding = PeriodicEmbeddings(
                n_features=num_num_features, 
                d_embedding=token_dim,
                lite=False,
                activation=True,
                # n_frequencies=n_frequencies,
                # frequency_init_scale=num_emb_sigma,
                # activation=False,
                # lite=num_emb_lite
            )
        elif num_embedding_type == 'ple': 
            # need to pass pre-computed bins
            assert bins is not None
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown numerical embedding type: {num_embedding_type}")

    def forward(self, x):
        return self.num_embedding(x)


class FeatureTokenizer(nn.Module):
    """Per-feature embedding for each data type: numerical and categorical"""
    def __init__(
        self,
        num_num_features: int,
        cardinalities: list[int],
        token_dim: int,
        num_embedding_type: str = 'plr',
    ):  
        super().__init__()
        self.num_num_features = num_num_features
        self.num_cat_features = len(cardinalities)
        
        if self.num_num_features > 0:
            self.num_embedding = NumEmbeddingLayer(
                num_num_features,
                num_embedding_type, # [linear, linear-relu, plr, ple]
                token_dim,
            )

        if self.num_cat_features > 0:
            self.cat_embedding = CategoricalEmbedding(cardinalities, token_dim)
    
    def forward(self, x_num, x_cat):
        tokens = []
        if self.num_num_features > 0:
            tokens.append(self.num_embedding(x_num))
        if self.num_cat_features > 0:
            tokens.append(self.cat_embedding(x_cat))

        x = torch.cat(tokens, dim=1) # (B, N, D)
        return x


class CategoricalEmbedding(nn.Module):
    """
    Categorical embedding using one table with offsets and zeros for missing or unknown values.
    """

    def __init__(
        self, 
        cardinalities: list[int],
        token_dim: int,
    ):
        super().__init__()
        cardinalities_ = [c + 1 for c in cardinalities] # add 1 for unknown or missing
        category_offsets = torch.tensor(
            np.concatenate([[0],
                            np.array(cardinalities_[:-1], dtype=np.int64)
                            ])
        ).cumsum(0)
        self.register_buffer("category_offsets", category_offsets)
        self.category_embeddings = nn.Embedding(int(sum(cardinalities_)), token_dim)
        nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        # set the embedding of the last category of each feature to zero
        for i, c in enumerate(cardinalities_):
            self.category_embeddings.weight.data[
                category_offsets[i] + c - 1
                ].zero_()
            
    def forward(self, x_cat):
        x_cat = self.category_embeddings(x_cat + self.category_offsets[None])
        return x_cat
    

class OutputProj(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(hidden_dim) 
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.lin2(self.relu(self.lin1(x)))
        if self.num_classes == 1:
            x = x.squeeze(-1)
        return x