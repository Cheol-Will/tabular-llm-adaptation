from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabarena.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.website.website_format import format_leaderboard
from tabarena.benchmark.models.ag import RealMLPModel

def main():
    
    return

if __name__ == '__main__':
    expname = str(Path(__file__).parent / "experiments" / "quickstart")  # folder location to save all experiment artifacts
    eval_dir = Path(__file__).parent / "eval" / "quickstart"
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata # dataframe that contains meta data such as data ID and num_features.

    datasets = ["anneal"]  
    folds = [0]

    methods = [
        AGModelBagExperiment(
            name='RealMLPDebug',
            model_cls=RealMLPModel,
            model_hyperparameters={
                "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"},  
            },
            num_bag_folds=8,
            time_limit=3600,
        ),
    ]
    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)
    result_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    print(result_lst)