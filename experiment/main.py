from __future__ import annotations

from pathlib import Path

from utils import (
    get_parser,
    get_model_experiments,
    filter_data,
)
from tabarena.benchmark.experiment import run_experiments_new

def main(args):
    output_dir = str(Path(__file__).parent / "results" / args.exp_name / args.model)
    task_ids = filter_data(args)
    model_experiments = get_model_experiments(
        model=args.model, 
        exp_name=args.exp_name,
        num_random_configs=args.num_random_configs,
        model_cls_name=args.model_cls_name, # for variants mapping
    )
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
    parser = get_parser()
    args = parser.parse_args()
    main(args)