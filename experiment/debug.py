import torch
from dataset.dataloader import TextLabelColumnTokenDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.dataloader import serialize_data, get_column_mask
from run_analysis import load_openml_data

import argparse

    
def get_response(task_id: int = 363621, idx: int = 0):
    X, y, label = load_openml_data(task_id)
    target_name = y.name if y is not None else "target"
    
    for i in range(idx):
        x_dict = X.iloc[i].to_dict() 
        y_val = y.iloc[i] 
        
        print(f"[Sample {i}]")
        print(f"X: {x_dict}\n")
        print(f"Target ({target_name}): {y_val}")

def debug_dataset_attn_mask():
    X, y, label = load_openml_data(363621)

    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    
    dataset = TextLabelColumnTokenDataset(
        tokenizer, X, y, "binclass", max_length=64
    )
    attn_mask = dataset[0]['attention_mask']
    print(attn_mask[0, 2:5, ])
    print(attn_mask[0, -5:, ])

    return
    


def debug_llmadapter_frozen():
    from custom_models.llmadapter.model import LLMAdapter
    from peft import LoraConfig, get_peft_model

    model = LLMAdapter(
        num_num_features=10,
        cardinalities=[5, 3],
        model_name="Qwen/Qwen2.5-0.5B",
        num_embedding_type="plr",
        token_dim=16,
        num_classes=2,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    print("\n=== Trainable modules after get_peft_model ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  [TRAIN] {name} {list(param.shape)}")

    print("\n=== Frozen modules (adapter layers only) ===")
    for name, param in model.named_parameters():
        if not param.requires_grad and any(k in name for k in ("feature_tokenizer", "mlp_adapter", "output_proj")):
            print(f"  [FROZEN] {name} {list(param.shape)}")


def _build_read_mask(
    seq_length: int,
    feat_indices: torch.Tensor,
    column_ids_lengths: list[int],
) -> torch.Tensor:
    """
    Read attention mask. Shape: (1, 1, N, N)
    """
    num_feature_cols = len(column_ids_lengths) - 1
    mask = torch.full((seq_length, seq_length), float("-inf"))

    prev, cur = 0, 0
    for i, length in enumerate(column_ids_lengths):
        if i < num_feature_cols:
            cur += length + 1  # +1 for feat slot token
            size = cur - prev
            mask[prev:cur, prev:cur] = torch.full((size, size), float("-inf")).triu(diagonal=1)
            prev = cur
        else:
            # target segment
            mask[-length:, :-length] = 0.0
            mask[-length:, -length:] = torch.full((length, length), float("-inf")).triu(diagonal=1)

    # feat slot tokens attend to each other freely
    mask[feat_indices[:, None], feat_indices[None, :]] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    args = parser.parse_args()
    # print(args.task_ids)

    # debug_dataset_attn_mask()
    # get_response(363621, idx=9)
    # get_response(363612)
    # get_response(363629)
    # get_response(363671)
    # get_response(363615)
    # debug_llmadapter_frozen()

    # idx = [0, 1, 3, 4 ,5, 7, 8, 9, 10]
    idx = [2, 6, 11]
    
    seq = torch.randn(2, 14, 16)
    
    prompt_mask = torch.zeros(seq.shape[1], dtype=torch.bool)
    prompt_mask[idx] = True

    text_indices = (~prompt_mask).nonzero(as_tuple=True)[0]
    feat_indices = prompt_mask.nonzero(as_tuple=True)[0]
    column_ids_lengths = [2, 3, 4, 2]
    mask = _build_read_mask(12, feat_indices, column_ids_lengths)
    print(mask)
    
if __name__ == "__main__":
    main()