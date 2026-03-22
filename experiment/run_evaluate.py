from __future__ import annotations

import argparse
import os
from pathlib import Path

from tabarena.nips2025_utils.end_to_end import EndToEnd, EndToEndResults
from tabarena.nips2025_utils.compare import get_subset_results
from tabarena.website.website_format import format_leaderboard

def main():
    methods = [
        'FTTransformer', 
        'TFMLLM'
    ]

    base_dir = Path(__file__).parent / "results" / "260320-num_emb"
    eval_dir = Path(__file__).parent / "evals" / "260320-num_emb"
    os.makedirs(eval_dir, exist_ok=True)

    path_raw_lst = [base_dir / method for method in methods]
    end_to_end = EndToEnd.from_path_raw(
        path_raw=path_raw_lst, 
        cache=True,
        cache_raw=True,
    )

    end_to_end_results = end_to_end.to_results()  # returns EndToEndResults
    results = end_to_end_results.get_results() # ours result
    table = get_subset_results(
        output_dir=eval_dir, 
        new_results=results, 
        folds=[0],
        subset=None,
    )
    print(table.columns)

    def pivot_main_table(table, method_category: str = 'default'):
        table_filtered = table[table['method'].str.contains(method_category, regex=False)]
        pivot = table_filtered.pivot(index="method", columns="dataset", values="metric_error_val") # test score
        
        # add problem_type column

        return pivot
    
    method_category_list = ['(default)', '(tuned)', '(tuned + ensemble)']
    for method_category in method_category_list:
        pivot = pivot_main_table(table, method_category)
        print(pivot.head())
        pivot.to_csv(f"{eval_dir}/main_table_{method_category}.csv")
    # if args.plot:
    #     leaderboard = end_to_end_results.compare_on_tabarena(
    #         only_valid_tasks=True,
    #         output_dir=fig_output_dir,
    #     )
    # leaderboard_website = format_leaderboard(leaderboard)
    # print(leaderboard_website.to_markdown(index=False))


if __name__ == "__main__":
    main()