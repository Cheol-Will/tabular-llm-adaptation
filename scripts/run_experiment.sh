model="FTTransformer"
# model="TFMLLM"
exp_name="260323"
# num_random_configs=10 # for fast experiment
num_random_configs=200 # for fast experiment
# num_data=10

python experiment/main.py \
    --model $model \
    --exp_name $exp_name \
    --num_random_configs $num_random_configs \
    --subset "tail"
