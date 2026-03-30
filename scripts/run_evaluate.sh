
# generate cache
# exp_name="260323"
# new_model="FTTransformer"

# exp_name="260320-num_emb"
# new_model="TFMLLM"

# exp_name="260326"
# # new_model="LLMBaselineBidirectional"
# new_model="LLMBaselineBidirectionalPooling"

# python experiment/evaluate.py \
#     --model $new_model\
#     --exp_name $exp_name \
#     --generate_cache


# summary
exp_name="260323"
model="LLMBaseline"
python experiment/evaluate.py \
    --model $model \
    --exp_name "260323"