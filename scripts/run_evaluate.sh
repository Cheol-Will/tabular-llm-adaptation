
# generate cache
# exp_name="260323"
# new_model="FTTransformer"

# exp_name="260320-num_emb"
# new_model="TFMLLM"
# exp_name="260326"

# exp_name="260420-LLMRead-GradClip"
# new_model="LLMRead"
# exp_name="260401-2-engineering"
# new_model="LLMAdapterEngineered"
# python experiment/evaluate.py \
#     --model $new_model\
#     --exp_name $exp_name \
#     --generate_cache

# new_model="LLMAdapterReg"
# python experiment/evaluate.py \
#     --model $new_model\
#     --exp_name $exp_name \
#     --generate_cache


# exp_name="260402-mlp_ratio-tune_mlp"
# new_model="LLMAdapterReg"
# python experiment/evaluate.py \
#     --model $new_model\
#     --exp_name $exp_name \
#     --generate_cache


# LLMAdapterReg_
# exp_name="260401-2-engineering"
# exp_name="260402-mlp_ratio-tune_mlp"
# new_model="LLMAdapterReg"
# # new_model="LLMAdapterEngineered"
# python experiment/evaluate.py \
#     --model $new_model \
#     --exp_name $exp_name \
#     --generate_cache



# summary
# exp_name="260331-engineering"
# exp_name="260331-engineering"
# exp_name="260423-bidir"
# exp_name="260424-next_token_pred"
# model="LLMAdapter"
# model="LLMSlot"

# exp_name="260429-LB-Bidir"
# model="LLMBaseline"

# exp_name="260421-3"
# exp_name="260423-bidir"
# exp_name="260429-LA-Bidir"
attn_type="bidir"
attn_type="causal"
attn_type="structured"
exp_name="260506-LS-qwen-$attn_type"
model="LLMSlot"
# model="LLMAdapter"
# exp_name="260429-LB-Bidir"
# model="LLMBaseline"

python experiment/evaluate.py \
    --exp_name $exp_name \
    --model "$model" \
    --generate_cache \
    # --model "$model$exp_name" \