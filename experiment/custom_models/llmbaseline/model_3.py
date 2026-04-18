import torch
import torch.nn as nn
from torch import autocast
from transformers import AutoModelForCausalLM
from ..layer import OutputProj

class LLMDummy(nn.Module):
    """
    
    """
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        ipnut_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        return

class LLMREADTokenLastOnly(nn.Module):
    """
    Docstings for LLMREADTokenLastOnly.
    """

    def __init__(
        self,
        model_name: str,
        num_columns: int,
        num_classes: int,
        task_type: str,
        label_token_ids: list[int] | None = None,
        mlp_ratio: float = 1.0,
        use_mlp_adapter: bool = False,
        adapter_dim: int = None,
    ):
        super().__init__()
        llm_dim = self.backbone.config.hidden_size
        if use_mlp_adapter: 
            assert adapter_dim is not None
        else:
            adapter_dim = llm_dim
    
        self.task_type = task_type
        self.num_classes = num_classes
        self.num_columns = num_columns
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        self.read_tokens = nn.Parameter(torch.empty(1, num_columns, adapter_dim))
        
        if use_mlp_adapter:
            self.mlp_adapter = nn.Sequential(*[
                nn.Linear(adapter_dim, llm_dim // 4),
                nn.ReLU(),
                nn.Linear(llm_dim // 4, llm_dim),
                nn.LayerNorm(self.llm_dim),
            ]) 
        else:
            self.mlp_adapter = None

        if task_type == "regression":
            num_classes = 1
        else:
            assert num_classes is not None
        self.output_proj = OutputProj(llm_dim, num_classes, mlp_ratio)

    def reset_parameters(self):
        nn.init.normal_(self.read_tokens, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor, # pre-computed mask
    ) -> torch.Tensor:
        
        B, S = input_ids.shape
        C = self.num_columns

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            read_tokens = self.read_tokens.expand(B, -1, -1) # (B, C, D)
            if self.mlp_adapter is not None:
                read_tokens = self.mlp_adapter(read_tokens) # (B, C, D)

            outputs = self.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            last_hidden = outputs.last_hidden_state # (B, S, D)

            # TODO: enabling last layer attention with READ tokens.
            # current last hidden is just hidden representations of text tokens.

            
        return

