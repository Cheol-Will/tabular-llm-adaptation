import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from transformers import AutoModelForCausalLM
from ..layer import (
    FeatureTokenizer,
    OutputProj,
)


class LLMSlotv2(nn.Module):

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
    ):
        super().__init__()

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
        self.mlp_adapter = nn.Sequential(
            nn.Linear(token_dim, self.llm_dim // 4),
            nn.ReLU(),
            nn.Linear(self.llm_dim // 4, self.llm_dim),
            nn.LayerNorm(self.llm_dim),
        )
        self.output_proj = OutputProj(self.llm_dim, num_classes, mlp_ratio)
        self.alpha_weights = nn.ParameterList()
        
        self.use_bidir_attn = use_bidir_attn
        self.num_features = num_num_features + len(cardinalities)

    def create_prompt(self, column_ids, column_ids_lengths):
        # column_ids: list of token-id lists, one per column (features + target, in order)
        # column_ids_lengths: list of ints (len(column_ids[i]) == column_ids_lengths[i])
        dev = next(self.backbone.parameters()).device
        embed_layer = self.backbone.get_input_embeddings()

        self.alpha_weights = nn.ParameterList()
        with torch.no_grad():
            for i, ids in enumerate(column_ids):
                ids_tensor = torch.tensor(ids, dtype=torch.long, device=dev)
                embeds = embed_layer(ids_tensor).float()  # (T_i, d_llm), frozen
                self.register_buffer(f'col_text_embeds_{i}', embeds)
                n = len(ids)
                self.alpha_weights.append(nn.Parameter(torch.full((n,), 1.0 / n, device=dev)))

    def reset_parameters(self):
        for param in self.alpha_weights:
            n = param.shape[0]
            nn.init.constant_(param, 1.0 / n)

    def _get_col_embed(self, col_idx: int) -> torch.Tensor:
        text_embeds = getattr(self, f'col_text_embeds_{col_idx}')  # (T, d_llm)
        weights = F.softmax(self.alpha_weights[col_idx], dim=0)    # (T,)
        return (weights.unsqueeze(-1) * text_embeds).sum(dim=0)    # (d_llm,)

    def get_bidir_attn_mask(self, seq_len: int):
        return torch.full((1, 1, seq_len, seq_len), 0, dtype=torch.bool)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        """
        Sequence: [COL_1][VAL_1][COL_2][VAL_2]...[COL_N][VAL_N][target]
        Even positions (0, 2, ..., 2N): column name tokens (learned alpha combination).
        Odd positions (1, 3, ..., 2N-1): feature value tokens (FeatureTokenizer + MLP adapter).
        Prediction from last hidden state at position 2N ([COL_target]).
        """
        B = x_num.shape[0]
        N = self.num_features

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.feature_tokenizer(x_num, x_cat)  # (B, N, token_dim)
            x = self.mlp_adapter(x)                   # (B, N, d_llm)

            # (N+1, d_llm): N feature cols + 1 target col
            col_embeds = torch.stack(
                [self._get_col_embed(i) for i in range(N + 1)], dim=0
            ).to(x.dtype)

            # Build interleaved sequence of length 2N + 1
            seq = torch.empty(B, 2 * N + 1, self.llm_dim, dtype=x.dtype, device=x.device)
            seq[:, 0::2, :] = col_embeds.unsqueeze(0).expand(B, -1, -1)  # COL tokens
            seq[:, 1::2, :] = x                                           # VAL tokens

            attention_mask = None
            if self.use_bidir_attn:
                attention_mask = self.get_bidir_attn_mask(seq.shape[1])
                attention_mask = attention_mask.expand(B, -1, -1, -1).to(seq.device)

            outputs = self.backbone.model(
                inputs_embeds=seq,
                attention_mask=attention_mask,
            )
            pred_hidden = outputs.last_hidden_state[:, -1, :]  # (B, d_llm)
            logits = self.output_proj(pred_hidden)

        return logits
