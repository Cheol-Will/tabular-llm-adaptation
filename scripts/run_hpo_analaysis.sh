model="TFMLLM"
exp_name="260320-num_emb"

python experiment/get_hpo_results.py \
    --model $model \
    --exp_name $exp_name