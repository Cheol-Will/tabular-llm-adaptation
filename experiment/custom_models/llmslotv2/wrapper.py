from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import pandas as pd
import torch

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from sklearn.impute import SimpleImputer

# Late import of the engine is handled inside _fit to manage dependencies
logger = logging.getLogger(__name__)

@contextmanager
def set_logger_level(logger_name: str, level: int):
    _logger = logging.getLogger(logger_name)
    old_level = _logger.level
    _logger.setLevel(level)
    try:
        yield
    finally:
        _logger.setLevel(old_level)

class LLMSlotv2Model(AbstractModel):
    ag_key = "LLMSlotv2"
    ag_name = "LLMSlotv2"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._bool_to_cat = None
        self._features_bool = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        time_limit: float | None = None,
        num_cpus: int = 4,
        num_gpus: float = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        start_time = time.time()
        log_level = logging.WARNING if verbosity <= 2 else logging.INFO
        device = 'cuda' if num_gpus > 0 and torch.cuda.is_available() else 'cpu'

        from .engine import LLMSlotv2Implementation

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split
            X, X_val, y, y_val = generate_train_test_split(
                X=X, y=y, problem_type=self.problem_type, test_size=0.2, random_state=0
            )
        
        hyp = self._get_model_params()
        bool_to_cat = hyp.pop("bool_to_cat", True)

        X = self.preprocess(X, is_train=True, bool_to_cat=bool_to_cat)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        cat_cols = X.select_dtypes(include='category').columns.tolist()

        self.model = LLMSlotv2Implementation(
            early_stopping_metric=self.stopping_metric,
            device=device,
            problem_type=self.problem_type,
            n_threads=num_cpus,
            **hyp
        )

        time_to_fit = None
        if time_limit is not None:
            time_to_fit = max(0.0, time_limit - (time.time() - start_time))

        with set_logger_level("custom_mlp", log_level):
            self.model.fit(
                X_train=X,
                y_train=y,
                X_val=X_val,
                y_val=y_val,
                cat_col_names=cat_cols,
                time_to_fit_in_seconds=time_to_fit
            )
        
    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X = self.preprocess(X, **kwargs)
        preds = self.model.predict_raw(X)

        if self.problem_type == 'regression':
            return (
                preds.squeeze(-1).float().cpu().numpy() * self.model.y_std_ + self.model.y_mean_
            )
        probs = torch.softmax(preds.float(), dim=-1).detach().cpu().numpy()
        return self._convert_proba_to_unified_form(probs)
    
    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        bool_to_cat: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        X = X.copy(deep=True)

        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(
                required_special_types=["bool"]
            )
            self._features_to_impute = self._feature_metadata.get_features(
                valid_raw_types=["int", "float"]
            )
            self._features_to_keep = [
                f for f in self._feature_metadata.get_features()
                if f not in self._features_to_impute
            ]

            if self._features_to_impute:
                self._imputer = SimpleImputer(strategy="mean")
                self._imputer.fit(X[self._features_to_impute])
            
        if self._imputer is not None and self._features_to_impute:
            X_impute = self._imputer.transform(X[self._features_to_impute])
            X_impute = pd.DataFrame(
                X_impute,
                index=X.index,
                columns=self._features_to_impute,
            )
            X = pd.concat([X[self._features_to_keep], X_impute], axis=1)
        
        if self._bool_to_cat and self._features_bool:
            X[self._features_bool] = X[self._features_bool].astype("category")
        
        return X
        
    def _set_default_params(self):
        defaults = {
            "num_epochs": 100,
            "lr": 1e-3,
            "lora_lr": 5e-4,
            "lora_rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "batch_size": 256,
            "dropout": 0.0,
            "weight_decay": 1e-5,
        }
        for param, val in defaults.items():
            self._set_default_param_value(param, val)
            
    @classmethod
    def supported_problem_types(cls) -> list:
        return ['binary', 'multiclass', 'regression']

    def _get_default_resources(self) -> tuple[int, int]:
        # Detect available hardware resources automatically
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def _more_tags(self) -> dict:
        return {"can_refit_full": False}