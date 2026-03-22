import importlib

def get_model_experiments(model: str, num_random_configs: int = 200):
    """
    Dynamically loads the experiment configuration generator for a given model.
    """
    try:
        module_path = f"custom_models.{model.lower()}.config_generator"
        config_module = importlib.import_module(module_path)
        get_configs_func = getattr(config_module, "get_experiment_configs")
        
        return get_configs_func(num_random_configs)
        
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find configuration generator for model '{model}': {e}")