from __future__ import annotations

import argparse
from pathlib import Path

from tabarena.nips2025_utils.end_to_end import EndToEnd, EndToEndResults
from tabarena.website.website_format import format_leaderboard

def main(args):
    methods = [
        # 'FTtransformer', 
        'TFMLLM'
    ]

    # base_dir = Path(__file__).parent / "results"
    base_dir = str(Path(__file__).parent / "results" / args.exp_name / args.model)

    # cache files
    model = args.model
    exp_name = args.exp_name
    f"{Path(__file__).parent}/results/260318/TFMLLM/data/TFMLLM_r35_BAG_L1/363612/0_0/results.pkl"
    f"{Path(__file__).parent}/results/260318/TFMLLM/data/TFMLLM_r35_BAG_L1/363612/0_0/results.pkl"
    
    
    fig_output_dir = Path(__file__).parent / "evals" 

    path_raw_lst = [base_dir / method for method in methods]
    end_to_end = EndToEnd.from_path_raw(
        path_raw=path_raw_lst,   # Pass both paths as a list
        cache=True,
        cache_raw=True,
    )

    results = end_to_end.get_results()
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
