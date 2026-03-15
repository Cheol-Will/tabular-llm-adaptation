model="FTTransformer"
name="FTTransformer"
python examples/benchmarking/run_experiment.py \
    --model $model\
    --name $name \
    --scale small middle \
    --num_gpus 1