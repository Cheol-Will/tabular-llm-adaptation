import os
import yaml
import importlib
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_cls_name", 
                    type=str, 
                    default=None, 
                    help='Model class name used for variant mapping.')
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--num_random_configs", type=int, default=10) # 200 in paper
    parser.add_argument("--num_data", type=int, default=None) 
    parser.add_argument("--subset", type=str, default='all', help="") 
    parser.add_argument("--problem_type", type=str, default=None, help="") 
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--use_tail_task_ids", action="store_true")

    # common model hyperparameters
    parser.add_argument("--mlp_fine_tune", action="store_true")
    parser.add_argument("--use_bidir_attn", action="store_true")
    parser.add_argument("--prediction_method", type=str, default="next_token_pred")
    return parser

def load_tid(name: str = 'tid'):
    file_path = 'data/dataset_list.yaml'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        tid_list = config.get(name, [])
    
    return tid_list

def filter_data(args):
    # task_ids = openml.study.get_suite("tabarena-v0.1").tasks
    task_ids = load_tid() # sorted by num_instances

    if args.num_data is not None:
        task_ids = task_ids[:args.num_data]

    if args.subset == 'small':
        task_ids =[
            # binary
            363621,
            363626,
            363629,
            # reg
            363698,
            363612,
            363625,
            363675,
            # multi-class
            363707,
            # 363614, # OOM
        ]       

    if args.subset == 'small-large-features':
        task_ids = [
        ]
    if args.problem_type is not None:
        if args.problem_type == 'reg':
            task_ids = load_tid(name='tid_reg')
        elif args.problem_type == 'multi':
            task_ids = load_tid(name='tid_multi')
        elif args.problem_type == 'binary':
            task_ids = load_tid(name='tid_binary')
        else:
            raise ValueError(f"Unknown problem_type: {args.problem_type}")

    if args.task_ids is not None:
        task_ids = args.task_ids

    if args.use_tail_task_ids:
        task_ids = task_ids[len(task_ids)//2:]

    return task_ids


def get_model_experiments(
        args,
        model: str, 
        exp_name: str,    
        num_random_configs: int = 200,
        model_cls_name: str = None,
    ):
    try:
        module_path = f"custom_models.{model.lower()}.config_generator"
        config_module = importlib.import_module(module_path)
        get_configs_func = getattr(config_module, "get_experiment_configs")
        if model_cls_name is not None:
            return get_configs_func(args, num_random_configs, exp_name, model_cls_name)
        else:
            return get_configs_func(args, num_random_configs, exp_name)

    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find configuration generator for model '{model}': {e}")