import argparse
import logging
import yaml
from pathlib import Path
from tabarena.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
import tabarena.benchmark.models.ag as models

logger = logging.getLogger(__name__)

def load_datasets(yaml_path: str, scales: list = None):
    if not scales:
        scales = ["small", "middle", "large"]
    
    path = Path(yaml_path)
    with open(path, 'r') as f:
        groups = yaml.safe_load(f)
    
    data_list = []
    for scale in scales:
        if scale in groups:
            data_list.extend(groups[scale])
    return data_list

def main(args):
    exp_path = str(Path(__file__).parent / "results" / args.name)
    exp_path = str(Path(__file__).parent / "results")
    context = TabArenaContext()
    datasets = load_datasets("data/dataset_list.yaml", scales=args.scales)

    try:
        model_cls_name = f"{args.model}Model"
        model_cls = getattr(models, model_cls_name)
    except AttributeError:
        logger.error(f"Class {model_cls_name} not found in tabarena.benchmark.models.ag")
        return

    methods = [
        AGModelBagExperiment(
            name=args.name,
            model_cls=model_cls, 
            model_hyperparameters={
                'ag_args_fit': {
                    'num_gpus': args.num_gpus, 
                },
                'ag_args_ensemble': {
                    'fold_fitting_strategy': 'sequential_local', 
                }
            },
            num_bag_folds=8,
            time_limit=3600,
        )
    ]

    runner = ExperimentBatchRunner(expname=exp_path, task_metadata=context.task_metadata)
    runner.run(datasets=datasets, folds=[0], methods=methods)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model name prefix")
    parser.add_argument('--name', type=str, required=True, help="Result folder name")
    parser.add_argument('--scales', type=str, nargs='+', default=["small"], help="Dataset scales")
    parser.add_argument('--num_gpus', type=int, required=True)
    
    args = parser.parse_args()
    main(args)