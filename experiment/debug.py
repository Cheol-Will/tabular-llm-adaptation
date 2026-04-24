import torch
from dataset.dataloader import TextLabelColumnTokenDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.dataloader import serialize_data, get_column_mask
from run_analysis import load_openml_data

import argparse

    
def get_response(task_id: int = 363621):
    X, y, label = load_openml_data(task_id)
    target_name = y.name if y is not None else "target"
    print(f"X[0]: {X.iloc[0, :]}, {target_name}: {y[0]}")

    # target_name = y.name if y is not None else "target"
    # texts = serialize_data(X, target_name)
    # sampled_row = texts[0]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    args = parser.parse_args()
    # print(args.task_ids)

    # debug_dataset_attn_mask()
    get_response(363621)
    get_response(363612)
    get_response(363629)
    get_response(363671)
    get_response(363615)
    # debug_llmadapter_frozen()

if __name__ == "__main__":
    main()