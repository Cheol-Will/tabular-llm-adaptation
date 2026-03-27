from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.dataloader import serialize_data
from run_analysis import load_openml_data
    
def get_response():
    X, y, label = load_openml_data(363621)
    # print(y.name)
    target_name = y.name if y is not None else "target"
    texts = serialize_data(X, target_name)
    sampled_row = texts[0]

    model_name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="cuda:7",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_texts = [
        sampled_row,
        "MonthsSinceLastDonation",
    ]
    for input_text in input_texts:
        input_token = tokenizer(input_text, return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **input_token,
            max_new_tokens=1,
            do_sample=False,       # greedy decoding
            pad_token_id=tokenizer.eos_token_id,  # suppress warning
            repetition_penalty=1.3,
        )

        input_len = input_token.input_ids.shape[1]
        output_ids = generated_ids[0][input_len:]
        # response = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(input_text)
        print(input_token)
        # print(response)

    

def main():
    get_response()

if __name__ == "__main__":
    main()