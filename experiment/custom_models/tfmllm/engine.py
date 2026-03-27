from __future__ import annotations

import gc
import logging
import os
import socket
import tempfile
import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from autogluon.core.metrics import compute_metric
from peft import LoraConfig, get_peft_model

from ..preprocessor import CustomOrdinalEncoder, CustomQuantileTransformer
from .model import TFMLLM

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

TaskType = Literal["regression", "binclass", "multiclass"]
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (module-level so mp.spawn can pickle them)
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _evaluate_worker(
    model: nn.Module,
    X_num: torch.Tensor,
    X_cat: torch.Tensor,
    y_tensor: torch.Tensor,
    task_type: TaskType,
    y_mean: float,
    y_std: float,
    early_stopping_metric,
    device: torch.device,
    eval_batch_size: int = 4096,
) -> float:
    """Run evaluation on rank-0 using the unwrapped model (no DDP sync needed)."""
    model.eval()
    loader = DataLoader(TensorDataset(X_num, X_cat), batch_size=eval_batch_size)
    outputs = []
    with torch.no_grad():
        for batch_num, batch_cat in loader:
            outputs.append(
                model(batch_num.to(device), batch_cat.to(device)).cpu()
            )
    model.train()

    output = torch.cat(outputs, dim=0)

    if task_type == "regression":
        y_pred = output.squeeze(-1).float().numpy() * y_std + y_mean
        y_true = y_tensor.numpy() * y_std + y_mean
        y_pred_proba = None
    else:
        y_true = y_tensor.numpy().astype(np.int64)
        y_pred_proba = torch.softmax(output, dim=-1).float().numpy()
        y_pred = y_pred_proba.argmax(axis=1)
        if task_type == "binclass":
            y_pred_proba = y_pred_proba[:, 1]

    return compute_metric(
        y=y_true,
        metric=early_stopping_metric,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        silent=True,
    )


def _ddp_worker(
    rank: int,
    world_size: int,
    gpu_ids: list[int],
    config: dict,
    task_type: TaskType,
    n_classes: int,
    num_num_features: int,
    cardinalities: list[int],
    y_mean: float,
    y_std: float,
    X_train_num: torch.Tensor,
    X_train_cat: torch.Tensor,
    y_train_tensor: torch.Tensor,
    X_val_num: torch.Tensor,
    X_val_cat: torch.Tensor,
    y_val_tensor: torch.Tensor,
    early_stopping_metric,
    model_save_path: str,
    start_time: float,
    time_to_fit_in_seconds: float | None,
    master_port: int,
):
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    model = TFMLLM(
        num_num_features=num_num_features,
        cardinalities=cardinalities,
        model_name=config.get("model_name", "Qwen/Qwen2.5-0.5B"),
        num_embedding_type=config.get("num_embedding_type", "plr"),
        token_dim=config.get("token_dim", 16),
        num_classes=n_classes,
    ).to(device)

    lora_config = LoraConfig(
        r=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=config.get("lora_dropout", 0.1),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if rank == 0:
        model.print_trainable_parameters()

    model = DDP(model, device_ids=[gpu_ids[rank]])

    base = model.module.base_model.model

    train_dataset = TensorDataset(X_train_num, X_train_cat, y_train_tensor)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 128),
        sampler=train_sampler,
    )

    loss_fn = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()
    lr = config.get("lr", 1e-3)
    lora_lr = config.get("lora_lr", 1e-4)
    optimizer = torch.optim.AdamW([
        {"params": base.feature_tokenizer.parameters(), "lr": lr},
        {"params": base.output_proj.parameters(), "lr": lr},
        {"params": base.backbone.parameters(), "lr": lora_lr},
    ], weight_decay=config.get("weight_decay", 1e-5))

    patience = config.get("patience", 16)
    remaining_patience = patience
    best_val_score = -np.inf
    best_state: dict | None = None

    # Epoch-0 baseline (rank 0 only; result broadcast to sync all ranks)
    if rank == 0:
        best_val_score = _evaluate_worker(
            model.module, X_val_num, X_val_cat, y_val_tensor,
            task_type, y_mean, y_std, early_stopping_metric, device,
            eval_batch_size=config.get("eval_batch_size", 4096),
        )
        best_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
        logger.info(f"Epoch 000 (baseline): Val Score = {best_val_score:.4f}")

    score_buf = torch.tensor([best_val_score], device=device)
    dist.broadcast(score_buf, src=0)
    best_val_score = score_buf.item()

    num_epochs = config.get("num_epochs", 200)
    epoch_iter = tqdm(range(num_epochs), desc="Training") if rank == 0 else range(num_epochs)

    for epoch in epoch_iter:
        if time_to_fit_in_seconds and (time.time() - start_time) >= time_to_fit_in_seconds:
            break

        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        for batch_num, batch_cat, batch_y in train_loader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_num, batch_cat)
            if task_type == "regression":
                output = output.squeeze(-1).float()
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Rank 0 evaluates; rank 1 waits at the broadcast below
        val_score = -np.inf
        if rank == 0:
            val_score = _evaluate_worker(
                model.module, X_val_num, X_val_cat, y_val_tensor,
                task_type, y_mean, y_std, early_stopping_metric, device,
                eval_batch_size=config.get("eval_batch_size", 4096),
            )
            avg_loss = train_loss / len(train_loader)
            if isinstance(epoch_iter, tqdm):
                epoch_iter.set_postfix({
                    "train_loss": f"{avg_loss:.4f}",
                    "val": f"{val_score:.4f}",
                    "best": f"{best_val_score:.4f}",
                })
            logger.info(f"Epoch {epoch:03d}: Val Score = {val_score:.4f} (Best: {best_val_score:.4f})")

        score_buf = torch.tensor([val_score], device=device)
        dist.broadcast(score_buf, src=0)
        val_score = score_buf.item()

        if val_score > best_val_score:
            best_val_score = val_score
            remaining_patience = patience
            if rank == 0:
                best_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
        else:
            remaining_patience -= 1

        # Broadcast early-stopping decision so all ranks agree
        patience_buf = torch.tensor([remaining_patience], device=device)
        dist.broadcast(patience_buf, src=0)
        remaining_patience = int(patience_buf.item())

        if remaining_patience <= 0:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    if rank == 0 and best_state is not None:
        torch.save(best_state, model_save_path)

    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

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
        gc.collect()
        torch.cuda.empty_cache()

        start_time = time.time()
        random_state = self.config.get("random_state", None)
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        gpu_ids = self.config.get("gpu_ids", [0, 1])
        self.device_ = torch.device(f"cuda:{gpu_ids[0]}")
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

        # Spawn DDP workers
        world_size = len(gpu_ids)
        master_port = _find_free_port()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model_save_path = f.name

        mp.spawn(
            _ddp_worker,
            args=(
                world_size, gpu_ids, self.config, self.task_type_, self.n_classes_,
                num_num_features, cardinalities, self.y_mean_, self.y_std_,
                X_train_num, X_train_cat, y_train_tensor,
                X_val_num, X_val_cat, y_val_tensor,
                self.early_stopping_metric, model_save_path,
                start_time, time_to_fit_in_seconds, master_port,
            ),
            nprocs=world_size,
            join=True,
        )

        # Load the best model on the primary GPU for inference
        lora_config = LoraConfig(
            r=self.config.get("lora_rank", 8),
            lora_alpha=self.config.get("lora_alpha", 32),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.config.get("lora_dropout", 0.1),
            bias="none",
        )
        self.model = TFMLLM(
            num_num_features=num_num_features,
            cardinalities=cardinalities,
            model_name=self.config.get("model_name", "Qwen/Qwen2.5-0.5B"),
            num_embedding_type=self.config.get("num_embedding_type", "plr"),
            token_dim=self.config.get("token_dim", 16),
            num_classes=self.n_classes_,
        ).to(self.device_)
        self.model = get_peft_model(self.model, lora_config)
        if os.path.exists(model_save_path):
            state = torch.load(model_save_path, map_location=self.device_)
            self.model.load_state_dict(state)
            os.unlink(model_save_path)

        gc.collect()
        torch.cuda.empty_cache()

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
