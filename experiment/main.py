from __future__ import annotations

import argparse
from pathlib import Path

import openml
from tabarena.benchmark.experiment import run_experiments_new
from utils import get_model_experiments

def filter_data(args):
    task_ids = openml.study.get_suite("tabarena-v0.1").tasks
    if args.subset == 'tail':
        task_ids = task_ids[len(task_ids)//2:]
    if args.num_data is not None:
        task_ids = task_ids[:args.num_data]

    if args.subset == 'small':
        task_ids =[
            363621,
            363629,
            363614,
            363698,
            363626,
            363685,
            363625,
            363696,
            363675,
            363707,
            363671,
            363612,
            363615
        ]

    return task_ids
    



def main(args):
    output_dir = str(Path(__file__).parent / "results" / args.exp_name / args.model)
    task_ids = filter_data(args)
    model_experiments = get_model_experiments(model=args.model, num_random_configs=args.num_random_configs) 

    run_experiments_new(
        output_dir=output_dir,
        model_experiments=model_experiments,
        tasks=task_ids,
        repetitions_mode="TabArena-Lite",
        exclude_task_ids = [
            363620, # num_cols > 1000
            363677,
            363697,
        ] 
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--num_random_configs", type=int, default=10) # 200 in paper
    parser.add_argument("--num_data", type=int, default=None) 
    parser.add_argument("--subset", type=str, default='all', help="") 
    args = parser.parse_args()
    main(args)