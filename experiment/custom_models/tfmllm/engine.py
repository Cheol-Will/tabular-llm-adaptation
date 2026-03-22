from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import autocast
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from ..preprocessor import CustomOrdinalEncoder, CustomQuantileTransformer

from autogluon.core.metrics import compute_metric
from .model import TFMLLM
from peft import LoraConfig, get_peft_model
# from .model import FTTransformer

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

TaskType = Literal["regression", "binclass", "multiclass"]
logger = logging.getLogger(__name__)


class TFMLLMImplementation:

    def __init__(self, early_stopping_metric: Scorer, **config):
        self.config = config
        self.early_stopping_metric = early_stopping_metric
        self.model = None

        self.ord_enc_: CustomOrdinalEncoder | None = None
        self.num_prep_: Pipeline | None = None
        self.cat_col_names_: list[Any] = []
        self.num_col_names_: list[Any] = []
        self.n_classes_: int | None = None
        self.task_type_: TaskType | None = None
        self.device_: torch.device | None = None
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Numerical
        if self.num_col_names_:
            x_cont_np = X[self.num_col_names_].to_numpy(dtype=np.float32)
            if is_train:
                self.num_prep_.fit(x_cont_np)
            X_num = torch.as_tensor(
                self.num_prep_.transform(x_cont_np),
                dtype=torch.float32,
            )
        else:
            X_num = torch.empty((len(X), 0), dtype=torch.float32)

        # Categorical
        if self.cat_col_names_:
            if is_train:
                self.ord_enc_.fit(X[self.cat_col_names_])
            X_cat = torch.as_tensor(
                self.ord_enc_.transform(X[self.cat_col_names_]),
                dtype=torch.long,
            )
        else:
            X_cat = torch.empty((len(X), 0), dtype=torch.long)

        # Target
        y_tensor: torch.Tensor | None = None
        if y is not None:
            if self.task_type_ == "regression":
                y_tensor = torch.as_tensor(y.to_numpy(np.float32))
            else:
                y_tensor = torch.as_tensor(y.to_numpy(np.int64), dtype=torch.long)

        return X_num, X_cat, y_tensor

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
        random_state = self.config.get("random_state", None)
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.device_ = torch.device(self.config.get("device", "cpu"))
        gpu_ids = self.config.get("gpu_ids", [0])
        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = [c for c in X_train.columns if c not in cat_col_names]
        problem_type = self.config["problem_type"]
        self.task_type_ = "binclass" if problem_type == "binary" else problem_type

        # Preprocessors
        self.ord_enc_ = CustomOrdinalEncoder()
        self.num_prep_ = Pipeline(steps=[
            ("qt", CustomQuantileTransformer(random_state=random_state)),
            ("imp", SimpleImputer(add_indicator=True)),
        ])

        # Data preparation
        X_train_num, X_train_cat, y_train_tensor = self._prepare_data(X_train, y_train, is_train=True)
        X_val_num, X_val_cat, y_val_tensor = self._prepare_data(X_val, y_val)

        # Regression: normalize target
        if self.task_type_ == "regression":
            self.n_classes_ = 1
            self.y_mean_ = float(y_train_tensor.mean().item())
            self.y_std_ = float(y_train_tensor.std().item())
            y_train_tensor = (y_train_tensor - self.y_mean_) / self.y_std_
        else:
            self.n_classes_ = int(y_train_tensor.max().item() + 1)

        num_num_features = X_train_num.shape[1]
        cardinalities = self.ord_enc_.get_cardinalities() if self.cat_col_names_ else []

        # DataLoader
        batch_size = self.config.get("batch_size", 128)
        train_loader = DataLoader(
            TensorDataset(X_train_num, X_train_cat, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )

        self.model = TFMLLM(
            num_num_features=num_num_features,
            cardinalities=cardinalities,
            model_name=self.config.get("model_name", 'Qwen/Qwen2.5-0.5B'),
            num_embedding_type=self.config.get("num_embedding_type", 'plr'),
            token_dim=self.config.get("token_dim", 16),
            num_classes=self.n_classes_,
        ).to(self.device_)

        # Define LoRA Config
        lora_config = LoraConfig(
            r=self.config.get('lora_rank', 8), 
            lora_alpha=self.config.get('lora_alpha', 32), 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            # target_modules=[
            #     "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"
            # ], 
            lora_dropout=self.config.get('lora_dropout', 0.1),
            bias="none",
        )
        # add LoRA adaptor
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        if len(gpu_ids) > 1 and torch.cuda.device_count() >= len(gpu_ids):
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
            logger.info(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
            
        if isinstance(self.model, nn.DataParallel):
            base = self.model.module.base_model.model
        else:
            base = self.model.base_model.model

        loss_fn = nn.MSELoss() if self.task_type_ == "regression" else nn.CrossEntropyLoss()
        lr, lora_lr = self.config.get('lr', 1e-3), self.config.get('lora_lr', 1e-4)
        optimizer = torch.optim.AdamW([
            {"params": base.feature_tokenizer.parameters(), "lr": lr},  
            {"params": base.output_proj.parameters(), "lr": lr},  
            {"params": base.backbone.parameters(), "lr": lora_lr}, 
        ], weight_decay=self.config.get("weight_decay", 1e-5))

        # Early stopping
        best_val_score = -np.inf
        best_state: dict | None = None
        patience = self.config.get("patience", 16)
        remaining_patience = patience

        # Epoch 0 baseline
        val_score = self._evaluate(X_val_num, X_val_cat, y_val_tensor)
        best_val_score = val_score

        best_state = {
            k: v.cpu().clone() 
            for k, v in (
                self.model.module.state_dict()
                if isinstance(self.model, nn.DataParallel)
                else self.model.state_dict()
            ).items()
        }

        num_epochs = self.config.get("num_epochs", 200)
        epoch_bar = tqdm(range(num_epochs), desc="Total Training")

        for epoch in epoch_bar:
            if time_to_fit_in_seconds is not None and (time.time() - start_time) >= time_to_fit_in_seconds:
                logger.info(f"Time limit reached at epoch {epoch}.")
                break

            self.model.train()
            for batch_num, batch_cat, batch_y in train_loader:
                batch_num = batch_num.to(self.device_)
                batch_cat = batch_cat.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimizer.zero_grad()
                
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = self.model(batch_num, batch_cat)
                    if self.task_type_ == "regression":
                        output = output.squeeze(-1)
                    loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()

            val_score = self._evaluate(X_val_num, X_val_cat, y_val_tensor)

            # logging
            epoch_bar.set_postfix({
                "val": f"{val_score:.4f}", 
                "best": f"{best_val_score:.4f}"
            })
            logger.info(f"Epoch {epoch:03d}: Val Score = {val_score:.4f} (Best: {best_val_score:.4f})")

            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {
                    k: v.cpu().clone()
                    for k, v in (
                        self.model.module.state_dict()
                        if isinstance(self.model, nn.DataParallel)
                        else self.model.state_dict()
                    ).items()
                }
                remaining_patience = patience
            else:
                remaining_patience -= 1
            if remaining_patience <= 0:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        if best_state is not None:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(best_state)
                self.model = self.model.module  # unwrap
            else:
                self.model.load_state_dict(best_state)

    def _get_raw_outputs(
        self,
        num_tensor: torch.Tensor,
        cat_tensor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        try:
            outputs = []
            loader = DataLoader(TensorDataset(num_tensor, cat_tensor), batch_size=batch_size)
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    for batch_num, batch_cat in loader:
                        outputs.append(
                            self.model(batch_num.to(self.device_), batch_cat.to(self.device_)).cpu()
                        )
            return torch.cat(outputs, dim=0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                logger.warning(f"OOM detected, reducing eval batch size: {batch_size} -> {batch_size // 2}")
                torch.cuda.empty_cache()
                return self._get_raw_outputs(num_tensor, cat_tensor, batch_size // 2)
            raise

    def _evaluate(
        self,
        X_num: torch.Tensor,
        X_cat: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> float:
        self.model.eval()
        output = self._get_raw_outputs(X_num, X_cat, self.config.get("eval_batch_size", 4096))

        if self.task_type_ == "regression":
            y_pred = output.squeeze(-1).float().numpy() * self.y_std_ + self.y_mean_
            y_true = y_tensor.numpy() * self.y_std_ + self.y_mean_
            y_pred_proba = None
        else:
            y_true = y_tensor.numpy().astype(np.int64)
            y_pred_proba = torch.softmax(output, dim=-1).float().numpy()
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
        X_num, X_cat, _ = self._prepare_data(X)
        return self._get_raw_outputs(X_num, X_cat, self.config.get("eval_batch_size", 1024))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.task_type_ == "regression":
            return raw.squeeze(-1).float().numpy() * self.y_std_ + self.y_mean_
        return raw.argmax(dim=-1).numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.task_type_ == "regression":
            return raw.squeeze(-1).float().numpy() * self.y_std_ + self.y_mean_
        probas = torch.softmax(raw, dim=-1).float().numpy()
        return probas[:, 1] if self.task_type_ == "binclass" else probas