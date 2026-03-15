from __future__ import annotations

import numpy as np
import torch
from tabarena.benchmark.experiment import YamlExperimentSerializer
from tabarena.benchmark.models.ag.fttransformer.ftt_model import FTTransformerModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator, generate_bag_experiments

def generate_single_config_ftt(rng):
    # Search space for FT-Transformer
    n_heads = rng.choice([4, 8])
    # d_token must be divisible by n_heads
    d_token = n_heads * rng.choice([16, 24, 32]) 
    
    params = {
        "n_blocks": int(rng.integers(2, 6, endpoint=True)),
        "d_token": int(d_token),
        "n_heads": int(n_heads),
        "dropout": float(rng.uniform(0.0, 0.3)),
        "lr": float(np.exp(rng.uniform(np.log(1e-5), np.log(5e-4)))),
        "weight_decay": float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-4)))),
        "epochs": int(rng.choice([64, 128, 256, 512])),
        "batch_size": int(rng.choice([128, 256, 512])),
    }
    return convert_numpy_dtypes(params)

def generate_configs_ftt(num_random_configs=100, seed=1234):
    rng = np.random.default_rng(seed)
    return [generate_single_config_ftt(rng) for _ in range(num_random_configs)]

gen_ftt = CustomAGConfigGenerator(
    model_cls=FTTransformerModel,
    search_space_func=generate_configs_ftt,
    manual_configs=[{}],
)

if __name__ == "__main__":
    config_defaults = [{}]
    random_configs = generate_configs_ftt(100, seed=1234)

    # Generate experiments for registry
    experiments_default = generate_bag_experiments(
        model_cls=FTTransformerModel,
        configs=config_defaults,
        time_limit=3600,
        name_id_prefix="default",
    )
    
    experiments_random = generate_bag_experiments(
        model_cls=FTTransformerModel, 
        configs=random_configs, 
        time_limit=3600,
        name_id_prefix="random"
    )
    
    experiments = experiments_default + experiments_random
    
    YamlExperimentSerializer.to_yaml(
        experiments=experiments, 
        path="configs_ftt.yaml"
    )