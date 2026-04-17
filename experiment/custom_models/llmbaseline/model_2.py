import torch
import torch.nn as nn
from torch import autocast
from transformers import AutoModelForCausalLM
from ..layer import OutputProj

class LLMColumnSpecificToken(nn.Module):
    """
    Column-specific tokens aggregate information from their own text tokens and attent to each other.
    
    NOTE: Currently, something's gone wrong since the loss is nan.
    """

    def __init__(
        self,
        model_name: str,
        num_columns: int,
        num_classes: int, 
        task_type: str,
        label_token_ids: list[int] | None = None,
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        self.num_columns = num_columns
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        
        llm_dim = self.backbone.config.hidden_size
        self.column_tokens = nn.Parameter(torch.empty(1, num_columns, llm_dim))
        self.reset_parameters()

        if task_type == "regression":
            num_classes = 1
        else:
            assert num_classes is not None
        self.output_proj = OutputProj(llm_dim, num_classes, mlp_ratio)

    def reset_parameters(self, ):
        nn.init.normal_(self.column_tokens, std=0.02) 

    def forward(
        self,
        input_ids: torch.Tensor, # (B, seq_len)
        attention_mask: torch.Tensor, # pre-computed mask
    ) -> torch.Tensor:
        B, S = input_ids.shape
        C = self.num_columns
        
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            column_tokens = self.column_tokens.expand(B, -1, -1) # (B, C, D)
            inputs_embeds_ = self.backbone.model.embed_tokens(input_ids) # (B, S, D)
            inputs_embeds = torch.cat([inputs_embeds_, column_tokens], dim=1) # (B, S+C, D)
            outputs = self.backbone.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            last_hidden = outputs.last_hidden_state[:, -C:, :] # (B, S+C, D)
            mean_pooling = last_hidden.mean(dim=1) # (B, D)
            logits = self.output_proj(mean_pooling)

        return logits # (B, num_classes)