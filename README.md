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

## 🕹️ Quickstart Use Cases

We share more details on various use cases of TabArena in our [examples](examples):

* 📊 **Benchmarking Predictive Machine Learning Models**: please refer to [examples/benchmarking](examples/benchmarking).
* 🚀 **Using SOTA Tabular Models Benchmarked by TabArena**: please refer to [examples/running_tabarena_models](examples/running_tabarena_models).
* 🗃️ **Analysing Metadata and Meta-Learning**: please refer to [examples/meta](examples/meta).
* 📈 **Generating Plots and Leaderboards**: please refer to [examples/plots_and_leaderboards](examples/plots_and_leaderboards).
* 🔁 **Reproducibility**: we share instructions for reproducibility in [examples](examples).

### Datasets 
Please refer to our [dataset curation repository](https://github.com/TabArena/tabarena_dataset_curation) to learn more about or contributed data! 


### Benchmark (Fitting Models)

If you intend to fit models, this is required.

```
uv pip install --prerelease=allow -e ./tabarena[benchmark]

# use GIT_LFS_SKIP_SMUDGE=1 in front of the command if installing TabDPT fails due to a broken LFS/pip setup
# GIT_LFS_SKIP_SMUDGE=1 uv pip install --prerelease=allow -e ./tabarena/[benchmark]
```

### Example Steps

Creating a project:
```
python examples/benchmarking/run_quickstart_tabarena.py 
```


# Downloading and using TabArena Artifacts

Artifacts will by default be downloaded into `~/.cache/tabarena/`. You can change this by specifying the environment variable `TABARENA_CACHE`.

The types of artifacts are:

1. Raw data -> The original results that are used to derive all other artifacts. Contains per-child test predictions from the bagged models, along with detailed metadata and system information absent from the processed results. Very large, often 100 GB per method type.
2. Processed data -> The minimal information needed for simulating HPO, portfolios, and generating the leaderboard. Often 10 GB per method type.
3. Results -> Pandas DataFrames of the results for each config and HPO setting on each task. Contains information such as test error, validation error, train time, and inference time. Generated from processed data. Used to generate leaderboards. Very small, often under 1 MB per method type.
4. Leaderboards -> Aggregated metrics comparing methods. Contains information such as ELO, win-rate, average rank, and improvability. Generated from a list of results files. Under 1 MB for all methods.
5. Figures & Plots -> Generated from results and leaderboards.

Examples of artifacts include:
* **Raw data**: [examples/meta/inspect_raw_data.py](examples/meta/inspect_raw_data.py)
* **Processed data**: [examples/meta/inspect_processed_data.py](examples/meta/inspect_processed_data.py)
* **Results**: [examples/plots/run_generate_main_leaderboard.py](examples/plots/run_generate_main_leaderboard.py)
