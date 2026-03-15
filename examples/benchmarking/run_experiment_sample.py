from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import argparse

from tabarena.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.website.website_format import format_leaderboard

dataset_list = [
    "blood-transfusion-service-center",
    "diabetes",
    "anneal",
    "QSAR_fish_toxicity",
    "credit-g",
    "maternal_health_risk",
    "concrete_compressive_strength",
    "qsar-biodeg",
    "healthcare_insurance_expenses",
    "website_phishing",
    "Fitness_Club",
    "airfoil_self_noise",
    "Another-Dataset-on-used-Fiat-500",
    "MIC",
    "Is-this-a-good-customer",
    "Marketing_Campaign",
    "hazelnut-spread-contaminant-detection",
    "seismic-bumps",
    "splice",
    "Bioresponse",
    "hiva_agnostic",
    "students_dropout_and_academic_success",
    "churn",
    "QSAR-TID-11",
    "polish_companies_bankruptcy",
    "wine_quality",
    "taiwanese_bankruptcy_prediction",
    "NATICUSdroid",
    "coil2000_insurance_policies",
    "Bank_Customer_Churn",
    # ~10k
    "heloc",
    "jm1",
    "E-CommereShippingData",
    "online_shoppers_intention",
    "in_vehicle_coupon_recommendation",
    "miami_housing",
    "HR_Analytics_Job_Change_of_Data_Scientists",
    "houses",
    "superconductivity",
    "credit_card_clients_default",
    "Amazon_employee_access",
    "bank-marketing",
    "Food_Delivery_Time",
    "physiochemical_protein",
    "kddcup09_appetency",
    # ~50k
    "diamonds",
    "Diabetes130US",
    "APSFailure",
    "SDSS17",
    "customer_satisfaction_in_airline",
    "GiveMeSomeCredit"
]

def main(args):
    expname = str(Path(__file__).parent / "results" / "all_results")  # folder location to save all experiment artifacts
    eval_dir = Path(__file__).parent / "results" / "summary"
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    # Sample for a quick demo
    datasets = ["anneal", "credit-g", "diabetes"]  
    # datasets = list(task_metadata["name"]) # for all data
    folds = [0]

    # import your model classes
    from tabarena.benchmark.models.ag import RealMLPModel
    from autogluon.tabular.models import LGBModel

    # This list of methods will be fit sequentially on each task (dataset x fold)
    methods = [
        # This will be a `config` in EvaluationRepository, because it computes out-of-fold predictions and thus can be used for post-hoc ensemble.
        AGModelBagExperiment(  # Wrapper for fitting a single bagged model via AutoGluon
            # The name you want the config to have
            name="LightGBM_c1_BAG_L1_Reproduced",

            # The class of the model. Can also be a string if AutoGluon recognizes it, such as `"GBM"`
            # Supports any model that inherits from `autogluon.core.models.AbstractModel`
            model_cls=LGBModel,
            model_hyperparameters={
                # "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"},  # uncomment to fit folds sequentially, allowing for use of a debugger
            },  # The non-default model hyperparameters.
            num_bag_folds=8,  # num_bag_folds=8 was used in the TabArena 2025 paper
            time_limit=3600,  # time_limit=3600 was used in the TabArena 2025 paper
        ),
        AGModelBagExperiment(
            name="TA-RealMLP_c1_BAG_L1_Reproduced",
            model_cls=RealMLPModel,
            model_hyperparameters={},
            num_bag_folds=8,
            time_limit=3600,
        ),
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    # Get the run artifacts.
    # Fits each method on each task (datasets * folds)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    # compute results
    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    print(f"New Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=eval_dir,
        only_valid_tasks=True,  # True: only compare on tasks ran in `results_lst`
        use_model_results=True,  # If False: Will instead use the ensemble/HPO results
        new_result_prefix="Demo_",
    )
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print(leaderboard_website.to_markdown(index=False))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # expname = str(Path(__file__).parent / "experiments" / "quickstart")  # folder location to save all experiment artifacts
    # eval_dir = Path(__file__).parent / "eval" / "quickstart"
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_trials', type=int, default=20,
                        help='Number of Optuna-based hyper-parameter tuning.')
    parser.add_argument(
        '--num_repeats', type=int, default=5,
        help='Number of repeated training and eval on the best config.')
    parser.add_argument(
        '--model_type', type=str, default='TabNet', choices=[
            'TabNet', 'FTTransformer', 'ResNet', 'MLP', 'TabTransformer', 'Trompt',
            'ExcelFormer', 'FTTransformerBucket', 'XGBoost', 'CatBoost', 'LightGBM'
        ])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='')
    args = parser.parse_args()
    
    main(args)
