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
    
    return parser

def load_tid():
    file_path = 'data/dataset_list.yaml'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        tid_list = config.get('tid', [])
    
    return tid_list

def filter_data(args):
    # task_ids = openml.study.get_suite("tabarena-v0.1").tasks
    task_ids = load_tid() # sorted by num_instances
    if args.subset == 'tail':
        task_ids = task_ids[len(task_ids)//2:]
    if args.num_data is not None:
        task_ids = task_ids[:args.num_data]

    if args.subset == 'small': # small datasets
        task_ids =[
            363621,
            363629,
            # 363614, # num_features=39 OOM on 2 GPUs
            363698,
            363626,
            363685,
            363625,
            # 363696, # num_featrues=42 OOM on 2 GPUS
            363675,
            363707,
        ]

    return task_ids


def get_model_experiments(
        model: str, 
        exp_name: str,    
        num_random_configs: int = 200,
        model_cls_name: str = None,
    ):
    """
    Dynamically loads the experiment configuration generator for a given model.
    """
    try:
        module_path = f"custom_models.{model.lower()}.config_generator"
        config_module = importlib.import_module(module_path)
        get_configs_func = getattr(config_module, "get_experiment_configs")
        if model_cls_name is not None:
            return get_configs_func(num_random_configs, exp_name, model_cls_name)
        else:
            return get_configs_func(num_random_configs, exp_name)


    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find configuration generator for model '{model}': {e}")