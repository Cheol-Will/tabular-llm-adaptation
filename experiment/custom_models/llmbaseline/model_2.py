import torch
import torch.nn as nn
from torch import autocast
from transformers import AutoModelForCausalLM
from ..layer import OutputProj

class LLMRead(nn.Module):
    """
    [READ] tokens read information from their corresponding text tokens.
    Mean pooling of [READ] tokens makes a prediction.
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
        self.read_tokens = nn.Parameter(torch.empty(1, num_columns, llm_dim))
        self.reset_parameters()

        if task_type == "regression":
            num_classes = 1
        else:
            assert num_classes is not None
        self.output_proj = OutputProj(llm_dim, num_classes, mlp_ratio)

    def reset_parameters(self, ):
        nn.init.normal_(self.read_tokens, std=0.02) 

    def forward(
        self,
        input_ids: torch.Tensor, # (B, seq_len)
        attention_mask: torch.Tensor, # pre-computed mask
    ) -> torch.Tensor:
        B, S = input_ids.shape
        C = self.num_columns
        
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            read_tokens = self.read_tokens.expand(B, -1, -1) # (B, C, D)
            inputs_embeds_ = self.backbone.model.embed_tokens(input_ids) # (B, S, D)
            inputs_embeds = torch.cat([inputs_embeds_, read_tokens], dim=1) # (B, S+C, D)
            outputs = self.backbone.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            last_hidden = outputs.last_hidden_state[:, -C:, :] # (B, S+C, D)
            mean_pooling = last_hidden.mean(dim=1) # (B, D)
            logits = self.output_proj(mean_pooling)

        return logits # (B, num_classes)
    
class LLMReadPred(LLMRead):
    """
    [READ] tokens read information from their corresponding text tokens.
    [PRED] tokens aggregate [READ] tokens and make a prediction.
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
        super().__init__(
            model_name,
            num_columns,
            num_classes,
            task_type,
            label_token_ids,
            mlp_ratio,
        )
        llm_dim = self.backbone.config.hidden_size
        self.pred_token = nn.Parameter(torch.empty(1, 1, llm_dim))
        self.reset_parameters() # init pred token

    def reset_parameters(self):
        super().reset_parameters() # init read tokens
        if hasattr(self, 'pred_token'):
            nn.init.normal_(self.pred_token, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        B, S = input_ids.shape
        C = self.num_columns

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            read_tokens = self.read_tokens.expand(B, -1, -1) # (B, C, D)
            pred_tokens = self.pred_token.expand(B, -1, -1) # (B, 1, D)
            inputs_embeds_ = self.backbone.model.embed_tokens(input_ids) # (B, S, D)
            inputs_embeds = torch.cat([inputs_embeds_, read_tokens, pred_tokens], dim=1) # (B, S+C+1, D)
            outputs = self.backbone.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            pred_hidden = outputs.last_hidden_state[:, -1, :] # (B, D)
            logits = self.output_proj(pred_hidden)

        return logits