# attn_type="causal"
# attn_type="bidir"
attn_type="structured_v2"
# num_embedding_type="pwl"
# exp_name="260508-LS-qwen-$attn_type-num_emb-$num_embedding_type"
# exp_name="260514-LS-qwen-$attn_type-num_emb-$num_embedding_type"
# exp_name="260514-LS-qwen-$attn_type"
exp_name="260506-LS-qwen-$attn_type"

# attn_type="causal"
# num_boosting=8
# boosting_token="eos"
# boosting_token="trainable"
# exp_name="260510-LSB-qwen-$attn_type-$boosting_token$num_boosting"

# model_name="meta-llama/Llama-3.2-1B"
model_name="Qwen/Qwen2.5-0.5B"
model="LLMSlot"
# model="LLMSlotBoost"

num_random_configs=10 # for fast experiment
python experiment/main.py \
    --model $model \
    --exp_name $exp_name \
    --num_random_configs $num_random_configs \
    --attn_type $attn_type \
    --model_name $model_name  \
    --task_subset small-features \
    --use_tail_task_ids \
    # --num_embedding_type $num_embedding_type \

    # --task_subset small-features \
    # --task_ids 363675 363615 363629 \
    # --num_boosting $num_boosting \
    # --boosting_token  $boosting_token \