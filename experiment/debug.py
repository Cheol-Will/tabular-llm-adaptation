import torch
from dataset.dataloader import TextLabelColumnTokenDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.dataloader import serialize_data, get_column_mask
from run_analysis import load_openml_data

# import pandas as pd

# df = pd.read_csv("/home/cheolseok/tabular-prediction/tabarena/tabarena/tabarena/nips2025_utils/metadata/curated_tabarena_dataset_metadata.csv")
# print(df.columns)
# df.sort_values(["problem_type","num_instances"], inplace=True)
# print(df.iloc[:10, :10])

# df.to_csv("/home/cheolseok/tabular-prediction/tabarena/tabarena/tabarena/nips2025_utils/metadata/curated_tabarena_dataset_metadata_sorted.csv")

# import pickle

# # file_path = "experiment/results/260318/TFMLLM/data/TFMLLM_r10_BAG_L1/363612/0_0/results.pkl"
# file_path = "experiment/results/260320-num_emb/TFMLLM/data/TFMLLM_c1_BAG_L1/363675/0_0/results.pkl"
# with open(file_path, 'rb') as file:
#     # Load the data from the file into a Python object
#     loaded_object = pickle.load(file)
# print(loaded_object["simulation_artifacts"].keys())
# # print(loaded_object["simulation_artifacts"]["pred_proba_dict_test"]) # prediction
# print(loaded_object["simulation_artifacts"]["y_test"][:10]) # prediction
# print(loaded_object["simulation_artifacts"]["y_val"][:10]) # prediction


# for k, v in loaded_object.items():
#     print(k)
#     print(v)

# can we check this during experiment and key the best hp config and validation result?

# print(loaded_object["metric"])
# print(loaded_object["metric_error_val"]) # validation
# print(loaded_object["metric_error"]) # test
# print(loaded_object["method_metadata"]["hyperparameters"]) # test


# print(loaded_object["time_train_s"])
# print(loaded_object.keys())


# import pickle
# with open("experiment/results/260320-num_emb/TFMLLM/data/TFMLLM_c1_BAG_L1/363621/0_0/results.pkl", "rb") as f:
#     obj = pickle.load(f)

# print(obj["framework"])
# print(obj["method_metadata"])
# print(type(obj))

# from main import load_tid
# task_ids = load_tid()
# print(task_ids)

    
def get_response():
    X, y, label = load_openml_data(363621)
    # print(y.name)
    target_name = y.name if y is not None else "target"
    texts = serialize_data(X, target_name)
    sampled_row = texts[0]

    model_name = "Qwen/Qwen2.5-0.5B"
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     dtype="auto",
    #     device_map="cuda:0",
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    text = f"{target_name} is"
    encoding = tokenizer(
        texts[:2],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )   
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left" # don't know why but this does not work. must specify padding side when init tokenizer.

    column_mask_list = []
    for row_idx, row in X.iloc[:2,].iterrows():
        column_mask_ = get_column_mask(tokenizer, row)
        attn_mask = encoding['attention_mask'][row_idx]
        
        # can we implement this efficiently?
        pad_mask = (attn_mask == 0) 
        num_pad = pad_mask.sum().item() 

        column_mask = attn_mask.clone().detach()
        column_mask[pad_mask] = -1 # left padding
        column_mask[num_pad: num_pad+column_mask_.shape[0]] = column_mask_ # column mask
        column_mask[num_pad+column_mask_.shape[0]:] = column_mask_.max().item() + 1

        column_mask_list.append(column_mask)

    column_mask = torch.stack(column_mask_list, dim=0) # (N, C)

    print(encoding['input_ids'].shape)
    print(encoding['input_ids'][:2])

    print(encoding['attention_mask'].shape)
    print(encoding['attention_mask'][:2])

    print(column_mask.shape)
    print(column_mask[:2])
    
    # print(attn_mask[:column_mask.shape[0]])
    # column_mask = attn_mask



    # TODO: Masking and padding
    # NOTE: Padding should be considered (left padding was used for LLM Baselines)

    # input_texts = [
    #     sampled_row,
    #     "MonthsSinceLastDonation",
    # ]
    # for input_text in input_texts:
    #     input_token = tokenizer(input_text, return_tensors="pt").to(model.device)
    #     generated_ids = model.generate(
    #         **input_token,
    #         max_new_tokens=1,
    #         do_sample=False,       # greedy decoding
    #         pad_token_id=tokenizer.eos_token_id,  # suppress warning
    #         repetition_penalty=1.3,
    #     )

    #     input_len = input_token.input_ids.shape[1]
    #     output_ids = generated_ids[0][input_len:]
    #     # response = tokenizer.decode(output_ids, skip_special_tokens=True)
    #     print(input_text)
    #     print(input_token)
    #     # print(response)

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
    

def main():
    debug_dataset_attn_mask()
    # get_response()

if __name__ == "__main__":
    main()