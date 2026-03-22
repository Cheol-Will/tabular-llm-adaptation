from autogluon.common.space import Int
from tabarena.utils.config_utils import ConfigGenerator
from .custom_random_forest_model import CustomRandomForestModel

def get_experiment_configs(*, num_random_configs: int = 1):
    """Generate the hyperparameter configurations to run for our custom model."""
    manual_configs = [
        {},
    ]
    search_space = {
        "n_estimators": Int(4, 50),
    }

    gen_custom_rf = ConfigGenerator(
        model_cls=CustomRandomForestModel,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    return gen_custom_rf.generate_all_bag_experiments(
        num_random_configs=num_random_configs, fold_fitting_strategy="sequential_local"
    )
