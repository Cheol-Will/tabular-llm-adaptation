# model="LLMBaseline"
model="FTTransformer"
python experiment/evaluate.py \
    --model $model\
    --exp_name "260323"