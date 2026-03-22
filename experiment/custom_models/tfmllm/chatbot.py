from transformers import AutoModelForCausalLM, AutoTokenizer

def get_response():
    model_name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="cuda:2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "What is a large langue model?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,       # greedy decoding
        pad_token_id=tokenizer.eos_token_id,  # suppress warning
        repetition_penalty=1.3,
    )

    input_len = inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_len:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(response)

def main():
    get_response()

if __name__ == "__main__":
    main()