from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.compare import get_subset_results
from utils_analysis import pivot_main_table, save_latex

def main(args):
    eval_dir = Path(__file__).parent / "evals" / args.exp_name
    os.makedirs(eval_dir, exist_ok=True)

    # methods = [
    #     "FTTransformer", 
    #     "LLMBaseline",
    #     "LLMBaselineBidirectionalPooling",
    #     "TFMLLM"
    # ]

    # TODO: refactoring

    new_methos = [
        "LLMBaselineBidirectionalPooling",
    ]
    # when new experiment results are added, run below code to make cache.
    base_dir = Path(__file__).parent / "results" / "260326"
    path_raw_lst = [base_dir / method for method in new_methos]
    end_to_end = EndToEnd.from_path_raw(
        path_raw=path_raw_lst,
        cache=True,
        cache_raw=True,
    )

    # end_to_end = EndToEnd.from_cache(
    #     methods=methods,
    # )
    # end_to_end_results = end_to_end.to_results()
    # results = end_to_end_results.get_results()
    # table = get_subset_results(
    #     output_dir=eval_dir,
    #     new_results=results,
    #     folds=[0],
    #     subset=None,
    # )

    # dataset_metric_map: pd.Series = (
    #     table[["dataset", "metric"]]
    #     .drop_duplicates()
    #     .set_index("dataset")["metric"]
    # )

    # method_category_list = ["(default)", "(tuned)", "(tuned + ensemble)"]
    # for method_category in method_category_list:
    #     tag = method_category.strip("()").replace(" + ", "_").replace(" ", "_")

    #     pivot, abbrev_metric_map = pivot_main_table(
    #         table=table,
    #         method_category=method_category,
    #         dataset_metric_map=dataset_metric_map,
    #         model=args.model,
    #     )
    #     print(f"\n=== {method_category} ===")
    #     print(pivot.head())

    #     csv_path   = eval_dir / f"main_table_{tag}.csv"
    #     latex_path = eval_dir / f"main_table_{tag}_latex.tex"

    #     pivot.to_csv(csv_path)
    #     save_latex(
    #         pivot=pivot,
    #         abbrev_metric_map=abbrev_metric_map,
    #         path=latex_path,
    #         method_category=method_category,
    #     )

    #     print(f"Saved: {csv_path}")
        # print(f"Saved: {latex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TFMLLM")
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    main(args)