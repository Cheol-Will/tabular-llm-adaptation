exp_list=(
    260514-LS-qwen-structured_v2
    260514-LS-qwen-structured_v2-num_emb-pwl
)

model_list=(
    LLMSlot
    LLMSlot
)

for i in "${!exp_list[@]}"; do
    exp_name="${exp_list[$i]}"
    model="${model_list[$i]}"

    python experiment/evaluate.py \
        --exp_name "$exp_name" \
        --model "$model" \
        --generate_cache
done

# attn_type="bidir"
exp_name="260514-LS-qwen-structured_v2"
model="LLMSlot"

python experiment/evaluate.py \
    --exp_name "$exp_name" \
    --model "$model$exp_name" \
    # --generate_cache