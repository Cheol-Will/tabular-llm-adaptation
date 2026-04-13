
# generate cache
# exp_name="260323"
# new_model="FTTransformer"

# exp_name="260320-num_emb"
# new_model="TFMLLM"
# exp_name="260326"

exp_name="260401-2-engineering"
new_model="LLMAdapterEngineered"
python experiment/evaluate.py \
    --model $new_model\
    --exp_name $exp_name \
    --generate_cache

new_model="LLMAdapterReg"
python experiment/evaluate.py \
    --model $new_model\
    --exp_name $exp_name \
    --generate_cache


exp_name="260402-mlp_ratio-tune_mlp"
new_model="LLMAdapterReg"
python experiment/evaluate.py \
    --model $new_model\
    --exp_name $exp_name \
    --generate_cache


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

# model="260401-2-engineeringLLMAdapterEngineered"
# exp_name="260401-engineering"
# model="FTTransformer"
# python experiment/evaluate.py \
#     --model $model \
#     --exp_name $exp_name