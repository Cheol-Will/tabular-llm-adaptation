import torch
import torch.nn as nn
from torch import autocast
from transformers import AutoModelForCausalLM
from ..layer import OutputProj


class LLMBaseline(nn.Module):
    """
    Text Serialization LLM Baseline.
    For regression, apply linear head to last hidden states.
    For classification, use cross entropy with token ids.
    Interaction: caual attention.
    Prediction: next token prediction.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_classes: int = 2,
        task_type: str = "binclass",
        label_token_ids: list[int] | None = None,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes

        if label_token_ids is not None:
            self.register_buffer(
                "label_token_ids",
                torch.tensor(label_token_ids, dtype=torch.long)
            )
        else:
            self.label_token_ids = None

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        if task_type == "regression":
            llm_dim = self.backbone.config.hidden_size
            self.regression_head = nn.Linear(llm_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor, # (B, seq_len)
        attention_mask: torch.Tensor, # (B, seq_len)
    ) -> torch.Tensor:
      
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.task_type == "regression":
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                last_hidden = outputs.hidden_states[-1][:, -1, :]  # (B, llm_dim)
                return self.regression_head(last_hidden.float()).squeeze(-1).float()  # (B,)
            else:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                last_logits = outputs.logits[:, -1, :].float()  # (B, vocab_size)
                return last_logits[:, self.label_token_ids]      # (B, n_classes)
            

class LLMBaselineBidirectional(nn.Module):
    """
    Interaction: bidirectional attention.
    Prediction: next token prediction.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_classes: int = 2,
        task_type: str = "binclass",
        label_token_ids: list[int] | None = None,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes

        if label_token_ids is not None:
            self.register_buffer(
                "label_token_ids",
                torch.tensor(label_token_ids, dtype=torch.long)
            )
        else:
            self.label_token_ids = None

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        if task_type == "regression":
            llm_dim = self.backbone.config.hidden_size
            self.regression_head = nn.Linear(llm_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor, # (B, seq_len)
        attention_mask: torch.Tensor, # (B, seq_len)
    ) -> torch.Tensor:
        B, S = input_ids.shape
        bidir_attention_mask = torch.zeros((B, 1, S, S), dtype=torch.bfloat16, device=input_ids.device)
        pad_mask = (attention_mask == 0)
        bidir_attention_mask = bidir_attention_mask.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.task_type == "regression":
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=bidir_attention_mask, # pass zero mask
                    output_hidden_states=True,
                )
                last_hidden = outputs.hidden_states[-1][:, -1, :] # (B, llm_dim)
                return self.regression_head(last_hidden.float()).squeeze(-1).float()
            else:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=bidir_attention_mask,
                )
                last_logits = outputs.logits[:, -1, :].float() # (B, vocab_size)
                return last_logits[:, self.label_token_ids] # (B, n_classes)


class LLMBaselinePooling(nn.Module):
    """
    Interaction: caual attention.
    Prediction: mean pooling of tokens -> MLP.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_classes: int = 2,
        task_type: str = "binclass",
        label_token_ids: list[int] | None = None,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        llm_dim = self.backbone.config.hidden_size
        if task_type == "regression":
            self.output_proj = OutputProj(llm_dim, 1)
        else:
            self.output_proj = OutputProj(llm_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor, # (B, seq_len)
        attention_mask: torch.Tensor, # (B, seq_len)
    ) -> torch.Tensor:
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1] # (B, S, D)
            pooling_mask = attention_mask.unsqueeze(-1).float() # (B, S, 1)
            masked_hidden = last_hidden * pooling_mask
            pooled_hidden = masked_hidden.sum(dim=1) / pooling_mask.sum(dim=1) # (B, D)
            logits = self.output_proj(pooled_hidden) # (B, num_classes)

        return logits
    
class LLMBaselineBidirectionalPooling(nn.Module):
    """
    Interaction: bidirectional attention.
    Prediction: mean pooling of tokens.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_classes: int = 2,
        task_type: str = "binclass",
        label_token_ids: list[int] | None = None,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        llm_dim = self.backbone.config.hidden_size
        if task_type == "regression":
            self.output_proj = OutputProj(llm_dim, 1)
        else:
            self.output_proj = OutputProj(llm_dim, num_classes)
    
    def forward(
        self,
        input_ids: torch.Tensor, # (B, seq_len)
        attention_mask: torch.Tensor, # (B, seq_len)
    ) -> torch.Tensor:
        B, S = input_ids.shape
        bidir_attention_mask = torch.zeros((B, 1, S, S), dtype=torch.bfloat16, device=input_ids.device)
        pad_mask = (attention_mask == 0)
        bidir_attention_mask = bidir_attention_mask.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=bidir_attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1] # (B, S, D)
            pooling_mask = attention_mask.unsqueeze(-1).float() # (B, S, 1)
            masked_hidden = last_hidden * pooling_mask
            pooled_hidden = masked_hidden.sum(dim=1) / pooling_mask.sum(dim=1) # (B, D)
            logits = self.output_proj(pooled_hidden) # (B, num_classes)

        return logits