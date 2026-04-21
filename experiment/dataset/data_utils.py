import typing as tp
import torch
import numpy as np
import pandas as pd


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