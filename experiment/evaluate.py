from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.compare import get_subset_results
from tabarena.website.website_format import format_leaderboard
from utils_analysis import pivot_main_table, save_latex


def generate_cache(args):
    # From the specified path, generate cache.
    path = Path(__file__).parent / "results" / args.exp_name / args.model  
    end_to_end = EndToEnd.from_path_raw(
        path_raw=path,
        cache=True,
        cache_raw=True,
        artifact_name=args.exp_name,
        name_suffix=args.exp_name,
        # name_prefix=args.exp_name,
    )


def save_main_table(args, end_to_end_results, eval_dir):
    csv_dir = eval_dir / "csv"
    latex_dir = eval_dir / "latex"
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)

    results = end_to_end_results.get_results()
    table = get_subset_results(
        output_dir=eval_dir,
        new_results=results,
        folds=[0],
        subset=None,
    )

    dataset_metric_map: pd.Series = (
        table[["dataset", "metric"]]
        .drop_duplicates()
        .set_index("dataset")["metric"]
    )

    method_category_list = ["(default)", "(tuned)", "(tuned + ensemble)"]
    
    for method_category in method_category_list:
        tag = method_category.strip("()").replace(" + ", "_").replace(" ", "_")

        for use_baseline_subset in [True, False]:
            subset_tag = "subset" if use_baseline_subset else "full"
            csv_path   = csv_dir / f"main_table_{tag}_{subset_tag}.csv"
            latex_path = latex_dir / f"main_table_{tag}_{subset_tag}_latex.tex"

            pivot, abbrev_metric_map = pivot_main_table(
                table=table,
                method_category=method_category,
                dataset_metric_map=dataset_metric_map,
                model=args.model,
                use_baseline_subset=use_baseline_subset,
            )

            pivot.to_csv(csv_path)
            save_latex(
                pivot=pivot,
                abbrev_metric_map=abbrev_metric_map,
                path=latex_path,
                method_category=method_category,
            )

            print(f"Saved: {csv_path}")
            print(f"Saved: {latex_path}\n")


def plot_elo(args, end_to_end_results, eval_dir):
    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=eval_dir,
        only_valid_tasks=[f"{args.model} (default)"],  # filter dataset 
        use_model_results=False,  # If False: Will instead use the ensemble/HPO results
        # new_result_prefix="260330",
    )
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print(leaderboard_website.to_markdown(index=False))
    print(f"Plot is saved into {eval_dir}")


def summary_evaluate(args):
    plot_dir = Path(__file__).parent / "evals" / args.exp_name / "plot"
    eval_dir = Path(__file__).parent / "evals" / args.exp_name
    
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    methods = [
        "FTTransformer",
        "LLMBaseline",
        "LLMBaselineBidirectional",
        "LLMBaselineBidirectionalPooling",
        "TFMLLM",
        # ("LLMAdapterEngineered260331-engineering", "260331-engineering"),
        # ("LLMAdapterEngineered260401-engineering", "260401-engineering"),
        # ("LLMAdapterEngineered260401-2-engineering", "260401-2-engineering"),
        # ("LLMAdapterReg260401-2-engineering", "260401-2-engineering"),
        # ("LLMAdapterReg260402-mlp_ratio-tune_mlp", "260402-mlp_ratio-tune_mlp"),
        ("LLMRead260420-LLMRead-GradClip", "260420-LLMRead-GradClip"),
    ]

    end_to_end = EndToEnd.from_cache(
        methods=methods,
    )
    end_to_end_results = end_to_end.to_results()
    
    plot_elo(args, end_to_end_results, plot_dir)
    save_main_table(args, end_to_end_results, eval_dir)

def main(args):
    if args.generate_cache:
        generate_cache(args)
        return
    summary_evaluate(args)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TFMLLM")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--baseline_subset", action='store_true')
    parser.add_argument("--generate_cache", action='store_true')
    args = parser.parse_args()

    main(args)