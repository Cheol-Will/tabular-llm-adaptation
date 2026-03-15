import torch
import torch.nn as nn
from typing import List, Optional

class FeatureTokenizer(nn.Module):
    def __init__(self, n_num: int, cat_cards: List[int], d_token: int):
        super().__init__()
        """
        N linear layers applied in parallel to N columns.
        
        Bias serves as a positional embedding for each column.
        """
        self.num_proj = nn.Parameter(torch.empty(n_num, d_token)) if n_num > 0 else None
        self.num_bias = nn.Parameter(torch.empty(n_num, d_token)) if n_num > 0 else None
        self.cat_embs = nn.ModuleList([nn.Embedding(c + 1, d_token) for c in cat_cards])
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_proj is not None:
            nn.init.xavier_uniform_(self.num_proj)
            nn.init.zeros_(self.num_bias)
        for emb in self.cat_embs:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.cls_token)

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]):
        tokens = []
        if x_num is not None and self.num_proj is not None:
            # Parallel linear projection for numerical features
            tokens.append(x_num.unsqueeze(-1) * self.num_proj + self.num_bias)
        if x_cat is not None and len(self.cat_embs) > 0:
            # Parallel embedding lookup for categorical features
            tokens.append(torch.stack([self.cat_embs[i](x_cat[:, i]) for i in range(len(self.cat_embs))], dim=1))
        
        x = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1)

class FTTransformer(nn.Module):
    def __init__(self, n_num: int, cat_cards: List[int], n_out: int,
                 n_blocks: int, d_token: int, n_heads: int, d_ffn_factor: float, dropout: float):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cards, d_token)
        d_ffn = int(d_token * d_ffn_factor)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True),
                'norm0': nn.LayerNorm(d_token),
                'ffn': nn.Sequential(
                    nn.Linear(d_token, d_ffn),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ffn, d_token)
                ),
                'norm1': nn.LayerNorm(d_token),
                'drop': nn.Dropout(dropout)
            }) for _ in range(n_blocks)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, n_out)
        )

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]):
        x = self.tokenizer(x_num, x_cat)
        for b in self.blocks:
            z = b['norm0'](x)
            z, _ = b['attn'](z, z, z)
            x = x + b['drop'](z)
            x = x + b['drop'](b['ffn'](b['norm1'](x)))
        return self.head(x[:, 0])