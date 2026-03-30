from autogluon.common.space import Int, Real, Categorical
from tabarena.utils.config_utils import ConfigGenerator
from .wrapper import TFMLLMModel

def get_experiment_configs(num_random_configs: int, exp_name: str):
    """Generate the hyperparameter configurations to run for TFM-LLM."""
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
        "token_dim": Categorical(16, 32),
        "lr": Real(1e-4, 1e-2, log=True), # feature_tokenizer, output_proj
        "lora_lr": Real(1e-5, 1e-3, log=True), # backbone (LoRA)
        "lora_rank": Categorical(4, 8, 16, 32),
        "lora_alpha": Categorical(16, 32, 64),
        "lora_dropout": Real(0.0, 0.2),
        "weight_decay": Real(1e-6, 1e-3, log=True),
        # "batch_size": Categorical(256, 512, 1024),
        "batch_size": Categorical(128, 256, 512),
    }

    gen = ConfigGenerator(
        model_cls=TFMLLMModel,
        manual_configs=manual_configs,
        search_space=search_space,
        name=f"TFMLLM_{exp_name}",
    )
    return gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        fold_fitting_strategy="sequential_local",
    )