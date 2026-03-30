# model="FTTransformer"
# model="TFMLLM"

model="LLMBaseline"
# model_cls_name="LLMBaseline"
# model_cls_name="LLMBaselineBidirectional"
# model_cls_name="LLMBaselinePooling"
# model_cls_name="LLMBaselineBidirectionalPooling"

model="FTTransformer"
exp_name="260323"
num_random_configs=200 # for fast experiment

python experiment/main.py \
    --model $model \
    --exp_name $exp_name \
    --subset "small"
    # --model_cls_name $model_cls_name \
    # --subset "small-large-features"