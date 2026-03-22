from autogluon.common.space import Int, Real, Categorical
from tabarena.utils.config_utils import ConfigGenerator
from .wrapper import MLPModel


def get_experiment_configs(num_random_configs: int = 10):
    """Generate the hyperparameter configurations to run for MLPModel."""
    manual_configs = [
        {},  # default params
    ]

    search_space = {
        "num_blocks": Int(1, 6),
        "hidden_dim": Categorical(64, 128, 256, 512),
        "dropout": Real(0.0, 0.5),
        "lr": Real(1e-4, 1e-2, log=True),
        "batch_size": Categorical(64, 128, 256, 512),
    }

    gen = ConfigGenerator(
        model_cls=MLPModel,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    return gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        fold_fitting_strategy="sequential_local",
    )