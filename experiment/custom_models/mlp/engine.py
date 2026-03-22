from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from ..preprocessor import CustomOrdinalEncoder, CustomQuantileTransformer

from autogluon.core.metrics import compute_metric
from .model import MLP

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

TaskType = Literal["regression", "binclass", "multiclass"]
logger = logging.getLogger(__name__)


class MLPImplementation:
    """MLP engine handling automated preprocessing, training, and inference."""

    def __init__(self, early_stopping_metric: Scorer, **config):
        self.config = config
        self.early_stopping_metric = early_stopping_metric
        self.model: MLP | None = None

        self.ord_enc_: CustomOrdinalEncoder | None = None
        self.num_prep_: Pipeline | None = None
        self.cat_col_names_: list[Any] = []
        self.n_classes_: int | None = None
        self.task_type_: TaskType | None = None
        self.device_: torch.device | None = None
        self.has_num_cols: bool = False
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0

    def _check_is_fitted(self):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")

    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        is_train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Categorical encoding
        if self.cat_col_names_:
            if is_train:
                self.ord_enc_.fit(X[self.cat_col_names_])
            X_cat = torch.as_tensor(
                self.ord_enc_.transform(X[self.cat_col_names_]),
                dtype=torch.float32,
            )
        else:
            X_cat = torch.empty((len(X), 0), dtype=torch.float32)

        # Numerical preprocessing
        if self.has_num_cols:
            num_cols = [c for c in X.columns if c not in self.cat_col_names_]
            x_cont_np = X[num_cols].to_numpy(dtype=np.float32)
            if is_train:
                self.num_prep_.fit(x_cont_np)
            X_cont = torch.as_tensor(
                self.num_prep_.transform(x_cont_np),
                dtype=torch.float32,
            )
        else:
            X_cont = torch.empty((len(X), 0), dtype=torch.float32)

        X_input = torch.cat([X_cont, X_cat], dim=1)

        # Target tensor
        y_tensor: torch.Tensor | None = None
        if y is not None:
            if self.task_type_ == "regression":
                y_tensor = torch.as_tensor(y.to_numpy(np.float32))
            else:
                y_tensor = torch.as_tensor(y.to_numpy(np.int64), dtype=torch.long)

        return X_input, y_tensor

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_col_names: list[Any],
        time_to_fit_in_seconds: float | None = None,
    ):
        start_time = time.time()

        # Seed for reproducibility
        random_state = self.config.get("random_state", None)
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.device_ = torch.device(self.config.get("device", "cpu"))
        self.cat_col_names_ = cat_col_names
        problem_type = self.config["problem_type"]
        self.task_type_ = "binclass" if problem_type == "binary" else problem_type

        # Preprocessors
        self.ord_enc_ = CustomOrdinalEncoder()
        self.num_prep_ = Pipeline(steps=[
            ("qt", CustomQuantileTransformer(random_state=random_state)),
            ("imp", SimpleImputer(add_indicator=True)),
        ])
        self.has_num_cols = bool(set(X_train.columns) - set(cat_col_names))

        # Data preparation (tensors remain on CPU for DataLoader)
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train, is_train=True)
        X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)

        # Regression: normalize target
        if self.task_type_ == "regression":
            self.n_classes_ = 1
            self.y_mean_ = float(y_train_tensor.mean().item())
            self.y_std_ = float(y_train_tensor.std().item())
            if self.y_std_ < 1e-8:
                logger.warning("Target std is near zero. Skipping normalization.")
                self.y_std_ = 1.0
            y_train_tensor = (y_train_tensor - self.y_mean_) / self.y_std_
        else:
            self.n_classes_ = int(y_train_tensor.max().item() + 1)

        # DataLoader setup
        batch_size = self.config.get("batch_size", 128)
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )

        # Model init
        self.model = MLP(
            num_features=X_train_tensor.shape[1],
            num_blocks=self.config.get("num_blocks", 2),
            hidden_dim=self.config.get("hidden_dim", 256),
            dropout=self.config.get("dropout", 0.1),
            num_classes=self.n_classes_,
        ).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("lr", 1e-3),
        )
        loss_fn = nn.MSELoss() if self.task_type_ == "regression" else nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_val_score = -np.inf
        best_state: dict | None = None
        patience = self.config.get("patience", 10)
        remaining_patience = patience

        # Evaluate before any training (epoch 0 baseline)
        val_score = self._evaluate(X_val_tensor, y_val_tensor)
        best_val_score = val_score
        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        for epoch in range(self.config.get("n_epochs", 100)):
            elapsed = time.time() - start_time
            if time_to_fit_in_seconds is not None and elapsed >= time_to_fit_in_seconds:
                logger.info(f"Time limit reached at epoch {epoch}. Stopping.")
                break

            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimizer.zero_grad()
                output = self.model(batch_x)
                if self.task_type_ == "regression":
                    output = output.squeeze(-1)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            val_score = self._evaluate(X_val_tensor, y_val_tensor)
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                remaining_patience = patience
            else:
                remaining_patience -= 1

            if remaining_patience <= 0:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _get_raw_outputs(self, data_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Run forward pass with automatic OOM-safe batch size reduction."""
        try:
            outputs = []
            loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size)
            with torch.no_grad():
                for (batch_x,) in loader:
                    outputs.append(self.model(batch_x.to(self.device_)).cpu())
            return torch.cat(outputs, dim=0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                logger.warning(
                    f"OOM detected, reducing eval batch size: {batch_size} -> {batch_size // 2}"
                )
                torch.cuda.empty_cache()
                return self._get_raw_outputs(data_tensor, batch_size // 2)
            raise

    def _evaluate(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> float:
        self.model.eval()
        initial_batch_size = self.config.get("eval_batch_size", 2048)
        output = self._get_raw_outputs(X_tensor, initial_batch_size)

        if self.task_type_ == "regression":
            y_pred = output.squeeze(-1).numpy() * self.y_std_ + self.y_mean_
            y_true = y_tensor.numpy() * self.y_std_ + self.y_mean_
            y_pred_proba = None
        else:
            y_true = y_tensor.numpy().astype(np.int64)
            y_pred_proba = torch.softmax(output, dim=-1).numpy()
            y_pred = y_pred_proba.argmax(axis=1)
            if self.task_type_ == "binclass":
                y_pred_proba = y_pred_proba[:, 1]

        return compute_metric(
            y=y_true,
            metric=self.early_stopping_metric,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            silent=True,
        )

    def predict_raw(self, X: pd.DataFrame) -> torch.Tensor:
        self._check_is_fitted()
        self.model.eval()
        X_tensor, _ = self._prepare_data(X)
        return self._get_raw_outputs(X_tensor, self.config.get("eval_batch_size", 1024))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.task_type_ == "regression":
            return (raw.squeeze(-1).numpy() * self.y_std_ + self.y_mean_)
        return raw.argmax(dim=-1).numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.task_type_ == "regression":
            return raw.squeeze(-1).numpy() * self.y_std_ + self.y_mean_
        probas = torch.softmax(raw, dim=-1).numpy()
        return probas[:, 1] if self.task_type_ == "binclass" else probas