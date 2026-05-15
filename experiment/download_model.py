from transformers import AutoModelForCausalLM, AutoTokenizer

model_id="Qwen/Qwen2.5-0.5B"

save_path = f"./{model_id}" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Save model and tokenizer to {save_path}")