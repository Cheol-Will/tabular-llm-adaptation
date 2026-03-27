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


class TextLabelDataset(Dataset):
    """
    input_ids, attention_mask, (optional) label
    - classification: label = class index (int)
    - regression: label = normalized float
    - inference: label (has_label=False)
    """
    def __init__(
        self,
        texts: list[str],
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

        encodings = tokenizer(
            texts,
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