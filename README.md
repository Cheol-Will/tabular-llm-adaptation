TabArena currently consists of:

- 51 manually curated tabular datasets representing real-world tabular data tasks.
- 9 to 30 evaluated splits per dataset.
- 16 tabular machine learning methods, including 3 tabular foundation models.
- 25,000,000 trained models across the benchmark, with all validation and test predictions cached to enable tuning and post-hoc ensembling analysis.
- A [live TabArena leaderboard](https://huggingface.co/spaces/TabArena/leaderboard) showcasing the results.

### Run Experiment
```
python experiment/main.py \
    --model "TFMLLM" \
    --exp_name "results" \
    --num_random_configs 200
```
or
```
bash scripts/run_experiment.sh
```

### Experiment Results
```
python examples/plots/run_generate_result_ours.py
```

### Datasets 
Please refer to our [dataset curation repository](https://github.com/TabArena/tabarena_dataset_curation) to learn more about or contributed data! 
