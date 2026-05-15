from typing import Optional
import os
import torch
import torch.nn as nn
from torch import autocast

from transformers import AutoModelForCausalLM
from ..layer import FeatureTokenizer, OutputProj


def _build_bidir_mask(seq_length: int) -> torch.Tensor:
    """Full attention (all zeros). Shape: (1, 1, N, N)"""
    return torch.zeros(1, 1, seq_length, seq_length)


def _build_structured_mask(
    seq_length: int,
    feat_indices: torch.Tensor,
    column_ids_lengths: list[int],
) -> torch.Tensor:
    """
    Read attention mask. Shape: (1, 1, N, N)
    """
    num_feature_cols = len(column_ids_lengths) - 1
    mask = torch.full((seq_length, seq_length), float("-inf"))

    prev, cur = 0, 0
    for i, length in enumerate(column_ids_lengths):
        if i < num_feature_cols:
            cur += length + 1  # +1 for feat slot token
            size = cur - prev
            mask[prev:cur, prev:cur] = torch.full((size, size), float("-inf")).triu(diagonal=1)
            prev = cur
        else:
            # target segment
            mask[-length:, :-length] = 0.0
            mask[-length:, -length:] = torch.full((length, length), float("-inf")).triu(diagonal=1)

    # feat slot tokens attend to each other freely
    mask[feat_indices[:, None], feat_indices[None, :]] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

def _build_structured_v2_mask(
    seq_length: int,
    feat_indices: torch.Tensor,
    column_ids_lengths: list[int],
) -> torch.Tensor:
    """
    Like _build_structured_mask, but target segment tokens attend only to
    feat slot tokens (feat_indices), not to column name tokens.
    Shape: (1, 1, N, N)
    """
    num_feature_cols = len(column_ids_lengths) - 1
    mask = torch.full((seq_length, seq_length), float("-inf"))

    prev, cur = 0, 0
    for i, length in enumerate(column_ids_lengths):
        if i < num_feature_cols:
            cur += length + 1
            size = cur - prev
            mask[prev:cur, prev:cur] = torch.full((size, size), float("-inf")).triu(diagonal=1)
            prev = cur
        else:
            # target segment: attend only to feat_indices (block to all others stays -inf)
            mask[-length:, feat_indices] = 0.0
            mask[-length:, -length:] = torch.full((length, length), float("-inf")).triu(diagonal=1)

    # feat slot tokens attend to each other freely
    mask[feat_indices[:, None], feat_indices[None, :]] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)

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
        attn_type: str = 'causal', # 'causal' | 'bidir' | 'structured' | 'structured_v2'
        prediction_method: str = 'next_token_pred',
        bins: list[torch.Tensor] = None,
    ):
        super().__init__()
        assert attn_type in ('causal', 'bidir', 'structured', 'structured_v2')
        assert prediction_method in ('next_token_pred', 'token_pooling')

        self.attn_type = attn_type
        self.prediction_method = prediction_method
        self.num_features = num_num_features + len(cardinalities)

        local_model_path = f"./pretrained_llm/{model_name}"
        self.backbone = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            # model_name, 
            dtype=torch.bfloat16,
            token=os.getenv("HUGGING_FACE_TOKEN"),
        )
        self.llm_dim = self.backbone.config.hidden_size

        self.feature_tokenizer = FeatureTokenizer(
            num_num_features, cardinalities, token_dim, num_embedding_type, bins
        )
        self.mlp_adapter = nn.Sequential(
            nn.Linear(token_dim, self.llm_dim // 4),
            nn.ReLU(),
            nn.Linear(self.llm_dim // 4, self.llm_dim),
            nn.LayerNorm(self.llm_dim),
        )
        self.output_proj = OutputProj(self.llm_dim, num_classes, mlp_ratio)

        # filled by create_prompt()
        self.prompt: Optional[nn.Parameter] = None
        self.register_buffer('prompt_mask', None)
        self.register_buffer('attn_mask', None)

    def create_prompt(
        self,
        column_ids: list[list[int]],
        column_ids_lengths: list[int],
    ) -> None:
        num_feature_cols = len(column_ids) - 1
        device = next(self.backbone.parameters()).device

        flat_ids = torch.tensor(
            [tok for seg in column_ids for tok in seg],
            dtype=torch.long, device=device,
        )
        embed_layer = self.backbone.get_input_embeddings()
        with torch.no_grad():
            text_embeds = embed_layer(flat_ids).float()

        total_len = sum(column_ids_lengths) + num_feature_cols
        prompt_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
        pos = 0
        for i, length in enumerate(column_ids_lengths):
            pos += length
            if i < num_feature_cols:
                prompt_mask[pos] = True
                pos += 1
        self.prompt = nn.Parameter(text_embeds)
        self.register_buffer('prompt_mask', prompt_mask)
        self.register_buffer('text_indices', (~prompt_mask).nonzero(as_tuple=True)[0])
        self.register_buffer('feat_indices', prompt_mask.nonzero(as_tuple=True)[0])

        if self.attn_type == 'causal':
            attn_mask = None
        elif self.attn_type == 'bidir':
            attn_mask = _build_bidir_mask(total_len).to(device)
        elif self.attn_type == 'structured':
            attn_mask = _build_structured_mask(
                total_len, self.feat_indices, column_ids_lengths
            ).to(device)
        elif self.attn_type == 'structured_v2':
            attn_mask = _build_structured_v2_mask(
                total_len, self.feat_indices, column_ids_lengths
            ).to(device)

        self.register_buffer('attn_mask', attn_mask)
        # print(f"Total Sequence Length: {total_len}")


    def _build_inputs(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        total_len = self.prompt_mask.shape[0]

        full = x.new_zeros(B, total_len, self.llm_dim)
        full[:, self.text_indices] = self.prompt.to(x.dtype)
        full[:, self.feat_indices] = x
        return full


    def _get_attention_mask(self, B: int, dtype: torch.dtype):
        if self.attn_mask is None:
            return None
        mask = self.attn_mask.to(dtype=dtype).expand(B, -1, -1, -1)
        if "llama" in self.backbone.config.model_type.lower():
            return mask
        return {"full_attention": mask}

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> torch.Tensor:
        B = x_num.shape[0]

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.feature_tokenizer(x_num, x_cat)   # (B, N, d_token)
            x = self.mlp_adapter(x)                     # (B, N, d_llm)

            inputs_embeds = self._build_inputs(x)
            attention_mask = self._get_attention_mask(B, x.dtype)

            outputs = self.backbone.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            hidden_state = outputs.last_hidden_state
            if self.prediction_method == 'next_token_pred':
                pred_hidden = hidden_state[:, -1, :]          # (B, D)
            else:  # token_pooling
                pred_hidden = hidden_state.mean(dim=1)        # (B, D)

            return self.output_proj(pred_hidden)

    def forward_with_attn(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> tuple:
        B = x_num.shape[0]

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.feature_tokenizer(x_num, x_cat)
            x = self.mlp_adapter(x)

            inputs_embeds = self._build_inputs(x)
            attention_mask = self._get_attention_mask(B, x.dtype)

            outputs = self.backbone.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
            )

            hs = outputs.last_hidden_state
            if self.prediction_method == 'next_token_pred':
                pred_hidden = hs[:, -1, :]
            else:
                pred_hidden = hs.mean(dim=1)

            logits = self.output_proj(pred_hidden)

        return logits, outputs.attentions, {"x_num": x_num, "x_cat": x_cat}