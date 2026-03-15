import logging
import time
from contextlib import contextmanager

import pandas as pd
import torch

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from sklearn.impute import SimpleImputer

from ._fttransformer_internal import FTTransformerImplementation

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

class FTTransformerModel(AbstractModel):
    ag_key = "FT-TRANSFORMER"
    ag_name = "FT-Transformer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        start_time = time.time()
        log_level = logging.WARNING if verbosity <= 2 else logging.INFO
        device = 'cuda' if num_gpus > 0 and torch.cuda.is_available() else 'cpu'
        hyp = self._get_model_params()
        
        X = self.preprocess(X, is_train=True, bool_to_cat=True, impute_bool=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)
            
        cat_cols = X.select_dtypes(include='category').columns.tolist()
        cards = [len(X[c].cat.categories) + 1 for c in cat_cols]
        
        for c in cat_cols:
            X[c] = X[c].cat.codes.astype('int64') + 1
            if X_val is not None:
                X_val[c] = X_val[c].cat.codes.astype('int64') + 1

        n_out = self.num_classes if self.problem_type != 'regression' else 1
        
        self.model = FTTransformerImplementation(
            task='reg' if self.problem_type == 'regression' else 'clf',
            cards=cards,
            n_out=n_out,
            device=device,
            n_threads=num_cpus,
            verbosity=verbosity,
            **hyp
        )

        with set_logger_level("fttransformer", log_level):
            self.model.fit(
                X_train=X,
                y_train=y,
                X_val=X_val,
                y_val=y_val,
                cat_col_names=cat_cols,
                time_to_fit_in_seconds=time_limit - (time.time() - start_time) if time_limit else None
            )

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X = self.preprocess(X, **kwargs).copy()
        cat_cols = X.select_dtypes(include='category').columns.tolist()
        
        for c in cat_cols:
            X[c] = X[c].cat.codes.astype('int64') + 1
            
        preds = self.model.predict(X, cat_cols)
        
        if self.problem_type == 'regression':
            return preds.flatten()
        
        probs = torch.softmax(torch.as_tensor(preds), dim=-1).detach().cpu().numpy()
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
                columns=self._features_to_impute
            )
            X = pd.concat([X[self._features_to_keep], X_impute], axis=1)

        if self._bool_to_cat and self._features_bool:
            X[self._features_bool] = X[self._features_bool].astype("category")

        return X


    def _set_default_params(self):
        defaults = {
            'n_blocks': 3,
            'd_token': 64,
            'n_heads': 4,
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'epochs': 100,
            'patience': 10,
            'batch_size': 256,
        }
        for param, val in defaults.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list:
        return ['binary', 'multiclass', 'regression']

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def _more_tags(self) -> dict:
        return {"can_refit_full": False}