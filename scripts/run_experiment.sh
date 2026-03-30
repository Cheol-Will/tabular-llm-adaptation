# model="FTTransformer"
# model="LLMBaseline"
model="TFMLLM"

# model_cls_name="LLMBaseline"
# model_cls_name="LLMBaselineBidirectional"
# model_cls_name="LLMBaselinePooling"
# model_cls_name="LLMBaselineBidirectionalPooling"
# exp_name="260326"
exp_name="260320-num_emb"
num_random_configs=10 # for fast experiment

python experiment/main.py \
    --model $model \
    --exp_name $exp_name \
    --subset "small" \
    --num_random_configs $num_random_configs

    # --subset "small-large-features"
    # --model_cls_name $model_cls_name \
