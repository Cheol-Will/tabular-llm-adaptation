from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd


class MLPModel(AbstractModel):
    ag_key = "MLP"
    ag_name = "MLP"

    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        bool_to_cat: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(
                required_special_types=["bool"]
            )
        if self._bool_to_cat and self._features_bool:
            # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
            X = X.copy(deep=True)
            X[self._features_bool] = X[self._features_bool].astype("category")

        return X
    
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        start_time = time.time()

        from .engine import MLPImplementation

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split
            X, X_val, y, y_val = generate_train_test_split(
                X=X, y=y, problem_type=self.problem_type, test_size=0.2, random_state=0
            )

        X = self.preprocess(X, is_train=True)
        X_val = self.preprocess(X_val)
        cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()

        params = self._get_model_params()
        device = "cuda" if num_gpus > 0 and torch.cuda.is_available() else "cpu"

        self.model = MLPImplementation(
            early_stopping_metric=self.stopping_metric,
            device=device,
            problem_type=self.problem_type,
            n_threads=num_cpus,
            **params,
        )

        time_to_fit = None
        if time_limit is not None:
            time_to_fit = max(0.0, time_limit - (time.time() - start_time))

        self.model.fit(
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            cat_col_names=cat_cols,
            time_to_fit_in_seconds=time_to_fit,
        )

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        raw_outputs = self.model.predict_raw(X)

        if self.problem_type == "regression":
            return (
                raw_outputs.squeeze(-1).cpu().numpy() * self.model.y_std_ + self.model.y_mean_
            )

        probs = torch.softmax(raw_outputs, dim=-1).detach().cpu().numpy()
        return self._convert_proba_to_unified_form(probs)

    def _set_default_params(self):
        defaults = {
            "num_blocks": 2,
            "hidden_dim": 256,
            "dropout": 0.1,
            "lr": 1e-3,
            "batch_size": 128,
            "eval_batch_size": 1024,
            "n_epochs": 100,
            "patience": 10,
        }
        for param, val in defaults.items():
            self._set_default_param_value(param, val)

    def _get_default_resources(self):
        from autogluon.common.utils.resource_utils import ResourceManager
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus