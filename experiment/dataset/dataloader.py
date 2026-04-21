import typing as tp
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataset.data_utils import serialize_data, get_column_mask


class TabularDataset(Dataset):

    def __init__(
        self,
        X_num,
        X_cat,
        label = None,
    ):
        self.input_num = X_num
        self.input_cat = X_cat
        self.label = label
        
        if self.label is not None: 
            self.has_labels = True
        else:
            self.has_labels = False

    def __len__(self):
        return self.input_num.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_num": self.input_num[idx],
            "input_cat": self.input_cat[idx],
        }
        if self.has_labels:
            item["label"] = self.label[idx]

        return item

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
        use_pred_token: bool = False,
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
        use_pred_token: bool = False,
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
        
        column_mask = self._make_column_mask(X, encodings, tokenizer)  # (N, S)
        self.attention_mask_4d = self._make_attention_mask(self.input_ids, column_mask, use_pred_token)  # (N, 1, S+C, S+C)

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
        Mask to represent 
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
            # Remaining tokens (", target is") get index = num_columns
            column_mask[num_pad + column_mask_.shape[0]:] = self.num_columns

            column_mask_list.append(column_mask)

        column_mask = torch.stack(column_mask_list, dim=0) # (N, max_token_length)
        
        return column_mask
    
    def _make_attention_mask(
        self,
        input_ids: torch.Tensor,    # (N, S)
        column_mask: torch.Tensor,  # (N, S)
        use_pred_token: bool = False,  # add pred token (self attention)
    ) -> torch.Tensor:
        N, S = input_ids.shape
        C = self.num_columns
        S_ = S + C 
        if use_pred_token:
            S_ += 1 # for pred token

        attention_mask = torch.full((N, 1, S_, S_), float('-inf'), dtype=torch.bfloat16)

        # Text block: causal mask 
        causal = torch.triu(torch.full((S, S), float('-inf'), dtype=torch.bfloat16), diagonal=1)
        attention_mask[:, :, :S, :S] = causal  

        # handle padding tokens
        pad_mask = (column_mask == -1)  # (N, S)
        attention_mask[:, :, :, :S] = attention_mask[:, :, :, :S].masked_fill(
            pad_mask[:, None, None, :], float('-inf')
        )

        # Column tokens: each attends only to its own text tokens 
        for c in range(C):
            col_text_mask = (column_mask == c)  # (N, S)
            scores = torch.where(col_text_mask, 0.0, float('-inf')).to(torch.bfloat16)
            attention_mask[:, :, S + c, :S] = scores.unsqueeze(1)

        # Full attention in column tokens (with pred token)
        attention_mask[:, :, S:, S:] = 0.0

        return attention_mask  # (N, 1, S_, S_)