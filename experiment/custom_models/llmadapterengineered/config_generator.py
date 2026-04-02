from autogluon.common.space import Int, Real, Categorical
from tabarena.utils.config_utils import ConfigGenerator
from .wrapper import LLMAdapterEngineeredModel


def get_experiment_configs(num_random_configs: int, exp_name: str):
    """Generate the hyperparameter configurations to run for LLMAdapterEgineeredModel."""
    manual_configs = [
        {
            "num_epochs": 200,
            "lr": 1e-3,
            "lora_lr": 1e-4,
            "lora_rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "batch_size": 256,
            "weight_decay": 1e-5,
        },
    ]

    search_space = {
        "token_dim": Categorical(16, 32, 64),
        "lr": Real(1e-4, 5e-2, log=True), 
        "lora_lr": Real(1e-5, 1e-2, log=True), 
        "lora_rank": Categorical(4, 8, 16, 32),
        "lora_alpha": Categorical(16, 32, 64),
        "lora_dropout": Categorical(0.0, 0.05, 0.1, 0.2),
        # "weight_decay": Real(1e-6, 1e-3, log=True),
        "mlp_ratio": Categorical(0.25, 0.5, 1.0),
        "batch_size": Categorical(128, 256, 512),
    }

    gen = ConfigGenerator(
        model_cls=LLMAdapterEngineeredModel,
        manual_configs=manual_configs,
        search_space=search_space,
        name=f"LLMAdapterEngineered_{exp_name}",
    )
    return gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        fold_fitting_strategy="sequential_local",
        method_kwargs=dict(init_kwargs=dict(verbosity=0)),
    )