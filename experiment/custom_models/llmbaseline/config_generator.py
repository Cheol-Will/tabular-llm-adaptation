from autogluon.common.space import Real, Categorical
from tabarena.utils.config_utils import ConfigGenerator
from .wrapper import (
    LLMBaselineModel,
    LLMBaselineBidirectionalModel,
    LLMBaselinePoolingModel,
    LLMBaselineBidirectionalPoolingModel,
    LLMColumnSpecificTokenModel,
)


_MODEL_CLS_MAP = {
    "LLMBaseline": LLMBaselineModel,
    "LLMBaselineBidirectional": LLMBaselineBidirectionalModel,
    "LLMBaselinePooling": LLMBaselinePoolingModel,
    "LLMBaselineBidirectionalPooling": LLMBaselineBidirectionalPoolingModel,
    "LLMColumnSpecificToken": LLMColumnSpecificTokenModel,
}


def get_experiment_configs(
        num_random_configs: int,
        exp_name: str,
        model_cls_name: str = "LLMBaseline",
    ):
    """Generate the hyperparameter configurations to run for LLM Baseline."""
    if model_cls_name not in _MODEL_CLS_MAP:
        raise ValueError(f"Unknown model class: {model_cls_name}")
    model_cls = _MODEL_CLS_MAP[model_cls_name]
    print(f"Get expeirment configs for {model_cls_name}")

    manual_configs = [
        {
            "model_name": "Qwen/Qwen2.5-0.5B",
            "num_epochs": 100,
            "lora_lr": 1e-4,
            "lora_rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "batch_size": 128,
            "eval_batch_size": 128,
            "max_length": 128,
            "weight_decay": 1e-5,
            "patience": 16,
        },
    ]

    search_space = {
        "lora_lr": Real(1e-5, 1e-3, log=True),
        "lora_rank": Categorical(4, 8, 16, 32),
        "lora_alpha": Categorical(16, 32, 64),
        "lora_dropout": Real(0.0, 0.2),
        "batch_size": Categorical(64, 128),
        "weight_decay": Real(1e-6, 1e-3, log=True),
        "max_length": Categorical(128),
    }


    gen = ConfigGenerator(
        model_cls=model_cls,
        manual_configs=manual_configs,
        search_space=search_space,
        name=f"LLMBaselineModel_{exp_name}",
    )
    return gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        fold_fitting_strategy="sequential_local",
        method_kwargs=dict(init_kwargs=dict(verbosity=0)),
    )