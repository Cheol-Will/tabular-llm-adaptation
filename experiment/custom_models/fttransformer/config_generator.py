from autogluon.common.space import Int, Real, Categorical
from tabarena.utils.config_utils import ConfigGenerator
from .wrapper import FTTransformerModel

def get_experiment_configs(num_random_configs: int = 10):
    manual_configs = [
        {
            "num_gpus": 1,
            "num_epochs": 200,
        },
    ]

    search_space = {
        "n_layers": Int(1, 6),
        "d_token": Categorical(64, 128, 192, 256),
        "n_heads": Categorical(4, 8),
        "d_ffn_factor": Categorical(0.5, 1, 2, 4),
        "attention_dropout": Real(0.0, 0.5),
        "ffn_dropout": Real(0.0, 0.5),
        "residual_dropout": Real(0.0, 0.2),
        "lr": Real(1e-5, 1e-3, log=True),
        "weight_decay": Real(1e-6, 1e-3, log=True),
        "batch_size": Categorical(256, 512, 1024),
    }

    gen = ConfigGenerator(
        model_cls=FTTransformerModel,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    
    return gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        fold_fitting_strategy="sequential_local",
    )