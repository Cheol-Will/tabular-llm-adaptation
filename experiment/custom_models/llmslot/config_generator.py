from autogluon.common.space import Int, Real, Categorical
from tabarena.utils.config_utils import ConfigGenerator
from .wrapper import LLMSlotModel

def get_experiment_configs(
        args,
        num_random_configs: int, 
        exp_name: str
    ):
    """
    Generate the hyperparameter configurations to run for LLMSlot.
    We make configurations for hyperparameters and construct specific experiment setting such as
    bidirectional attention or so on.
    """
    manual_configs = [
        {
            "num_epochs": 100,
            "token_dim": 32,
            "lr": 1e-3,
            "lora_lr": 5e-4,
            "lora_rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "mlp_ratio": 1.0,
            "weight_decay": 1e-5,
            "batch_size": 128,
            "mlp_fine_tune": args.mlp_fine_tune,
            "use_bidir_attn": args.use_bidir_attn,
            "prediction_method": args.prediction_method,
            "project_name": f"{args.model}_{args.exp_name}", # for wandb
        },
    ]

    search_space = {
        "num_epochs": 100,
        "token_dim": Categorical(16, 32),
        "lr": Real(1e-4, 5e-3, log=True), # feature_tokenizer, output_proj
        "lora_lr": Real(1e-5, 1e-4, log=True), # backbone (LoRA)
        "lora_rank": Categorical(4, 8, 16, 32),
        "lora_alpha": Categorical(16, 32, 64),
        "lora_dropout": Real(0.0, 0.2),
        "mlp_ratio": Categorical(0.25, 0.5, 1.0),
        "weight_decay": Real(1e-6, 1e-3, log=True),
        "batch_size": Categorical(128, 256),
        "mlp_fine_tune": args.mlp_fine_tune,
        "use_bidir_attn": args.use_bidir_attn,
        "prediction_method": args.prediction_method,
        "project_name": f"{args.model}_{args.exp_name}", # for wandb
    }

    gen = ConfigGenerator(
        model_cls=LLMSlotModel,
        manual_configs=manual_configs,
        search_space=search_space,
        name=f"LLMAdapter_{exp_name}",
    )
    return gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        fold_fitting_strategy="sequential_local",
        method_kwargs=dict(init_kwargs=dict(verbosity=0)),
    )