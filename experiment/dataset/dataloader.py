import typing as tp
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def serialize_data(X: pd.DataFrame, target_name: str) -> list[str]:
    """
    format: "col1: val1, col2: val2, ..., {target_name} is"
    """
    return [
        ", ".join([f"{col}: {val}" for col, val in row.items()])
        + f", {target_name} is"
        for _, row in X.iterrows()
    ]


def get_column_mask(tokenizer, row, max_length: int | None = None):
    """
    Build a column-index mask aligned to the joint tokenization of the full row.

    Tokenizes the full serialized row once (no special tokens except BOS at the
    very start), then tokenizes each "col: val, " segment without special tokens
    to measure its token span.  This avoids the per-column BOS tokens that caused
    length mismatches in the previous approach.

    Returns a 1-D LongTensor of length ≤ max_length (if given) where each entry
    is the 0-based column index that owns that token position.  Tokens belonging
    to the trailing ", {target} is" part get index (num_columns).
    """
    items = list(row.items())
    segments = []
    for col, val in items:
        segments.append(f"{col}: {val}, ")

    # Tokenize each segment without special tokens to get exact span lengths.
    span_lengths = []
    for seg in segments:
        ids = tokenizer.encode(seg, add_special_tokens=False)
        span_lengths.append(len(ids))

    # The full tokenization (with a leading BOS) is 1 + sum(span_lengths) + tail.
    # Build a column index per token position (0-based, after the BOS).
    total = sum(span_lengths)
    col_indices = torch.zeros(total, dtype=torch.long)
    pos = 0
    for col_idx, length in enumerate(span_lengths):
        col_indices[pos:pos + length] = col_idx
        pos += length

    # Prepend a slot for the BOS token (assign it to column 0 arbitrarily).
    bos_slot = torch.zeros(1, dtype=torch.long)
    column_mask_ = torch.cat([bos_slot, col_indices], dim=0)

    if max_length is not None:
        column_mask_ = column_mask_[:max_length]

    return column_mask_


class TextLabelDataset(Dataset):
    """
    input_ids, attention_mask, (optional) label
    - classification: label = class index (int)
    - regression: label = normalized float
    - inference: label (has_label=False)
    """
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series, 
        # texts: list[str],
        tokenizer,
        task_type: str,                         # "binclass" | "multiclass" | "regression"
        labels: list | None = None,            
        label_token_ids: list[int] | None = None,  # classification only 
        max_length: int = 256,
        y_mean: float = 0.0,
        y_std: float = 1.0,
    ):
        self.task_type = task_type
        self.has_label = labels is not None
        self.num_columns = X.shape[1]
        target_name = y.name
        X_texts = serialize_data(X, target_name)

        encodings = tokenizer(
            X_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

        if self.has_label:
            if task_type == "regression":
                labels_norm = [(l - y_mean) / y_std for l in labels]
                self.labels = torch.tensor(labels_norm, dtype=torch.float32)
            else:
                # class index, not token_id
                self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.has_label:
            item["label"] = self.labels[idx]
        return item
    
class TextLabelColumnTokenDataset(Dataset):
    """
    Serialized dataset with attention mask designed for Column-aware attention.
    TODO: our tokenizer uses padding=left!

    """    
    def __init__(
        self,
        tokenizer,
        X: pd.DataFrame,
        y: pd.Series, 
        task_type: str,                         # "binclass" | "multiclass" | "regression"
        labels: list | None = None,            
        label_token_ids: list[int] | None = None,  # classification only 
        max_length: int = 256,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        target_name: str = None,
    ):
        self.task_type = task_type
        self.has_label = labels is not None
        self.num_columns = X.shape[1]
        X_texts = serialize_data(X, target_name)

        encodings = tokenizer(
            X_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.column_mask = self._make_column_mask(X, encodings, tokenizer)  # (N, S)
        self.attention_mask_4d = self._make_attention_mask(self.input_ids, self.column_mask)  # (N, 1, S+C, S+C)

        if self.has_label:
            if task_type == "regression":
                labels_norm = [(l - y_mean) / y_std for l in labels]
                self.labels = torch.tensor(labels_norm, dtype=torch.float32)
            else:
                # class index, not token_id
                self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask_4d[idx],
        }
        if self.has_label:
            item["label"] = self.labels[idx]
        
        return item

    def _make_column_mask(self, X, encodings, tokenizer):
        """
        NOTE: we use left padding for tokenizer.
        """
        column_mask_list = []
        for tensor_idx, (_, row) in enumerate(X.iterrows()):
            attn_mask = encodings['attention_mask'][tensor_idx]
            pad_mask = (attn_mask == 0)
            num_pad = int(pad_mask.sum().item())
            available = attn_mask.shape[0] - num_pad  # non-padding slots

            column_mask_ = get_column_mask(tokenizer, row, max_length=available)

            column_mask = attn_mask.clone().detach()
            column_mask[pad_mask] = -1  # left padding
            column_mask[num_pad:num_pad + column_mask_.shape[0]] = column_mask_
            # Remaining tokens (truncated tail / ", target is") get index = num_columns
            column_mask[num_pad + column_mask_.shape[0]:] = self.num_columns

            column_mask_list.append(column_mask)

        column_mask = torch.stack(column_mask_list, dim=0) # (N, max_token_length)
        
        return column_mask
    
    def _make_attention_mask(
        self,
        input_ids: torch.Tensor,    # (N, S)
        column_mask: torch.Tensor,  # (N, S)
    ) -> torch.Tensor:              # (N, 1, S+C, S+C)
        N, S = input_ids.shape
        C = self.num_columns
        S_ = S + C

        attention_mask = torch.full((N, 1, S_, S_), float('-inf'), dtype=torch.bfloat16)

        # Text block: causal mask 
        causal = torch.triu(torch.full((S, S), float('-inf'), dtype=torch.bfloat16), diagonal=1)
        attention_mask[:, :, :S, :S] = causal  

        # handle padding tokens
        pad_mask = (column_mask == -1)  # (N, S)
        attention_mask[:, :, :S_, :S] = attention_mask[:, :, :S_, :S].masked_fill(
            pad_mask[:, None, None, :], float('-inf')
        )

        # Column tokens: each attends only to its own text tokens 
        for c in range(C):
            col_text_mask = (column_mask == c)  # (N, S)
            scores = torch.where(col_text_mask, 0.0, float('-inf')).to(torch.bfloat16)
            attention_mask[:, :, S + c, :S] = scores.unsqueeze(1)

        # Full attention in column tokens
        attention_mask[:, :, S:, S:] = 0.0

        return attention_mask  # (N, 1, S+C, S+C)