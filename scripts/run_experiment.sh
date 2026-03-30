# model="FTTransformer"
# model="LLMBaseline"
# model="TFMLLM"

model="LLMBaseline"
# model_cls_name="LLMBaseline"
# model_cls_name="LLMBaselinePooling"
# model_cls_name="LLMBaselineBidirectional"
exp_name="260326"
num_random_configs=10 # for fast experiment

model_cls_name="LLMBaselineBidirectionalPooling"
python experiment/main.py \
    --model $model \
    --exp_name $exp_name \
    --model_cls_name $model_cls_name \
    --subset "small"

# model_cls_name="LLMBaselineBidirectional"
# python experiment/main.py \
#     --model $model \
#     --exp_name $exp_name \
#     --model_cls_name $model_cls_name \
#     --subset "small-large-features"
#     # --subset "small"
