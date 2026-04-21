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
from torch.utils.data import Dataset, DataLoader, TensorDataset, DistributedSampler
from dataset.dataloader import TabularDataset
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from sklearn.preprocessing import RobustScaler

import wandb
from dotenv import load_dotenv

from autogluon.core.metrics import compute_metric
from peft import LoraConfig, get_peft_model

from ..preprocessor import CustomOrdinalEncoder, SmoothClipTransformer
from .model import LLMAdapter

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

TaskType = Literal["regression", "binclass", "multiclass"]
logger = logging.getLogger(__name__)



def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def get_optimizer(model, config):
    lr = config.get("lr", 1e-3)
    lora_lr = config.get("lora_lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-5)
    base = model.module.base_model.model

    params = [{"params": base.backbone.parameters(), "lr": lora_lr}]
    if hasattr(base, "feature_tokenizer"):
        params.append({"params": base.feature_tokenizer.parameters(), "lr": lr})
    if hasattr(base, "mlp_adapter"):
        params.append({"params": base.mlp_adapter.parameters(), "lr": lr})
    if hasattr(base, "output_proj"):
        params.append({"params": base.output_proj.parameters(), "lr": lr})
    if hasattr(base, "read_tokens"):
        params.append({"params": [base.read_tokens], "lr": lr})
    if hasattr(base, "pred_token"):
        params.append({"params": [base.pred_token], "lr": lr})

    # PEFT freezes all non-LoRA params; re-enable the tabular adapter layers
    for name, param in model.named_parameters():
        if any(k in name for k in ("feature_tokenizer", "mlp_adapter", "output_proj")):
            param.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        params,
        weight_decay=weight_decay,
    )

    return optimizer

def _evaluate_worker(
    model: nn.Module,
    val_loader: DataLoader,
    task_type: TaskType,
    y_mean: float,
    y_std: float,
    early_stopping_metric,
    device: torch.device,
    eval_metric=None,
) -> float:
    """Run evaluation on rank-0 using the unwrapped model (no DDP sync needed)."""
    model.eval()
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch_num = batch["input_num"].to(device)
            batch_cat = batch["input_cat"].to(device)
            batch_y = batch["label"]

            out = model(batch_num.to(device), batch_cat.to(device)).cpu().float()
            all_outputs.append(out)
            all_labels.append(batch_y)

    model.train()
    outputs = torch.cat(all_outputs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if task_type == "regression":
        y_pred = outputs.numpy() * y_std + y_mean
        y_true = labels.numpy() * y_std + y_mean
        y_pred_proba = None
    else:
        y_true = labels.numpy().astype(np.int64)
        y_pred_proba = torch.softmax(outputs, dim=-1).numpy()
        y_pred = y_pred_proba.argmax(axis=1)
        if task_type == "binclass":
            y_pred_proba = y_pred_proba[:, 1]

    val_score = compute_metric(
        y=y_true,
        metric=early_stopping_metric,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        silent=True,
    )

    if eval_metric is not None:
        # Compute roc_auc for binary classification.
        metric_val = compute_metric(
            y=y_true,
            metric=eval_metric,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            silent=True,
        )
    else:
        metric_val = val_score

    return val_score, metric_val


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
    train_dataset: Dataset,
    val_dataset: Dataset,
    early_stopping_metric,
    model_save_path: str,
    start_time: float,
    time_to_fit_in_seconds: float | None,
    master_port: int,
    use_wandb: bool = False,
    task_id: int = 0,
    project_name: str = 'tfmllm',
):
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        world_size=world_size,
        rank=rank,
        device_id=device,
    )

    if config.get("mlp_fine_tune", False):
        _target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        _target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=_target_modules,
        lora_dropout=config.get("lora_dropout", 0.1),
        bias="none",
    )

    model = LLMAdapter(
        num_num_features=num_num_features,
        cardinalities=cardinalities,
        model_name=config.get("model_name", "Qwen/Qwen2.5-0.5B"),
        num_embedding_type=config.get("num_embedding_type", "plr"),
        token_dim=config.get("token_dim", 16),
        num_classes=n_classes,
        mlp_ratio=config.get("mlp_ratio", 1.0)
    ).to(device)
    model = get_peft_model(model, lora_config)
    
    if rank == 0:
        model.print_trainable_parameters()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"  trainable: {name} {list(param.shape)}")

    model = DDP(model, device_ids=[gpu_ids[rank]])

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 128),
        sampler=train_sampler,
    )

    val_loader = (
        DataLoader(val_dataset, batch_size=config.get("eval_batch_size", 1024), shuffle=False)
        if rank == 0
        else None
    )

    loss_fn = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()
    
    optimizer = get_optimizer(model, config)
    
    patience = config.get("patience", 16)
    remaining_patience = patience
    best_val_score = -np.inf
    best_state: dict | None = None

    ag_args = os.getenv("CURRENT_AG_ARGS", "_c1")

    if rank == 0 and use_wandb:
        wandb.init(
            project=f"{project_name}-tabarena",
            name=f"task_{task_id}_{project_name}",
            group=f"{task_id}{ag_args}",
            config={
                "task_id": task_id,
                "model_name": config.get("model_name", "Qwen/Qwen2.5-0.5B"),
                "task_type": task_type,
                "token_dim": config.get("token_dim", 16),
                "lora_rank": config.get("lora_rank", 8),
                "lora_alpha": config.get("lora_alpha", 32),
                "lr": config.get("lr", 1e-3),
                "lora_lr": config.get("lora_lr", 5e-4),
                "batch_size": config.get("batch_size", 128),
                "weight_decay": config.get("weight_decay", 1e-5),
                "num_epochs": config.get("num_epochs", 100),
                "patience": config.get("patience", 16),
                "ag_args": ag_args,
            },
            reinit="finish_previous",
            settings=wandb.Settings(silent=True),
        )

    from autogluon.core.metrics import get_metric
    eval_metric = get_metric("roc_auc", "binary") if task_type == "binclass" else None

    # Epoch-0 baseline (rank 0 only; result broadcast to sync all ranks)
    if rank == 0:
        best_val_score, _ = _evaluate_worker(
            model.module, val_loader,
            task_type, y_mean, y_std, early_stopping_metric, device,
        )
        best_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
        logger.info(f"Epoch 000 (baseline): Val Score = {best_val_score:.4f}")

    score_buf = torch.tensor([best_val_score], device=device)
    dist.broadcast(score_buf, src=0)
    best_val_score = score_buf.item()

    num_epochs = config.get("num_epochs", 100)
    warmup_epochs = config.get("warmup_epochs", 10)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    epoch_iter = tqdm(range(num_epochs), desc="Training") if rank == 0 else range(num_epochs)

    for epoch in epoch_iter:
        if time_to_fit_in_seconds and (time.time() - start_time) >= time_to_fit_in_seconds:
            break

        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        grad_norm = 0.0

        for batch in train_loader:
            batch_num = batch["input_num"].to(device)
            batch_cat = batch["input_cat"].to(device)
            batch_y = batch["label"].to(device)

            optimizer.zero_grad()
            output = model(batch_num, batch_cat).float()
            if task_type == "regression":
                output = output.squeeze(-1)
            loss = loss_fn(output, batch_y)
            loss.backward()
            grad_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            train_loss += loss.item()

        # Rank 0 evaluates; rank 1 waits at the broadcast below
        val_score = -np.inf
        if rank == 0:
            val_score, metric_val = _evaluate_worker(
                model.module, val_loader,
                task_type, y_mean, y_std, early_stopping_metric, device,
                eval_metric=eval_metric
            )
            avg_loss = train_loss / len(train_loader)
            avg_grad_norm = grad_norm / len(train_loader)
            if isinstance(epoch_iter, tqdm):
                epoch_iter.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "val_score": f"{val_score:.4f}",
                    "best_best_score": f"{best_val_score:.4f}",
                    "metric_val": f"{metric_val:.4f}",
                })
            logger.info(f"Epoch {epoch:03d}: Val Score = {val_score:.4f} (Best: {best_val_score:.4f})")
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "grad_norm": avg_grad_norm,
                    "val_score": val_score,
                    "metric_val": metric_val,
                    "best_val_score": best_val_score,
                })

        score_buf = torch.tensor([val_score], device=device)
        dist.broadcast(score_buf, src=0)
        val_score = score_buf.item()

        if val_score > best_val_score:
            best_val_score = val_score
            remaining_patience = patience
            if rank == 0:
                best_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
                if use_wandb:
                    wandb.run.summary["best_val_score"] = best_val_score
                    wandb.run.summary["best_epoch"] = epoch
        else:
            remaining_patience -= 1

        scheduler.step()

        # Broadcast early-stopping decision so all ranks agree
        patience_buf = torch.tensor([remaining_patience], device=device)
        dist.broadcast(patience_buf, src=0)
        remaining_patience = int(patience_buf.item())

        if remaining_patience <= 0:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    if rank == 0:
        if best_state is not None:
            torch.save(best_state, model_save_path)
        if use_wandb:
            wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


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

        load_dotenv()
        self.use_wandb = bool(os.getenv("WANDB_API_KEY"))
        if self.use_wandb:
            print("✅ WandB API Key loaded from .env")

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

        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        self.device_ = torch.device(f"cuda:{gpu_ids[0]}")
        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = [c for c in X_train.columns if c not in cat_col_names]
        problem_type = self.config["problem_type"]
        self.task_type_ = "binclass" if problem_type == "binary" else problem_type

        # Preprocessors
        self.ord_enc_ = CustomOrdinalEncoder()
        self.num_prep_ = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("robust_scale", RobustScaler()),
            ("smooth_clip", SmoothClipTransformer()),
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
            y_val_tensor = (y_val_tensor - self.y_mean_) / self.y_std_
        else:
            self.n_classes_ = int(y_train_tensor.max().item() + 1)

        num_num_features = X_train_num.shape[1]
        cardinalities = self.ord_enc_.get_cardinalities() if self.cat_col_names_ else []

        num_embedding_type=self.config.get("num_embedding_type", "plr")
        if num_embedding_type == 'ple':
            # TODO: compute bins and pass it to the model
            NotImplementedError

        train_dataset = TabularDataset(X_train_num, X_train_cat, y_train_tensor)
        val_dataset = TabularDataset(X_val_num, X_val_cat, y_val_tensor)

        # Spawn DDP workers
        world_size = len(gpu_ids)
        master_port = _find_free_port()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model_save_path = f.name

        task_id = self.config.get("task_id") or int(os.getenv("CURRENT_TASK_ID", "0"))
        project_name = self.config.get("project_name", "llmadapter")

        mp.spawn(
            _ddp_worker,
            args=(
                world_size, gpu_ids, self.config, self.task_type_, self.n_classes_,
                num_num_features, cardinalities, self.y_mean_, self.y_std_,
                train_dataset, val_dataset,
                self.early_stopping_metric, model_save_path,
                start_time, time_to_fit_in_seconds, master_port,
                self.use_wandb, task_id, project_name,
            ),
            nprocs=world_size,
            join=True,
        )

        # Load the best model on the primary GPU for inference
        if self.config.get("mlp_fine_tune", False):
            _target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            _target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=self.config.get("lora_rank", 8),
            lora_alpha=self.config.get("lora_alpha", 32),
            target_modules=_target_modules,
            lora_dropout=self.config.get("lora_dropout", 0.1),
            bias="none",
        )
        self.model = LLMAdapter(
            num_num_features=num_num_features,
            cardinalities=cardinalities,
            model_name=self.config.get("model_name", "Qwen/Qwen2.5-0.5B"),
            num_embedding_type=self.config.get("num_embedding_type", "plr"),
            token_dim=self.config.get("token_dim", 16),
            num_classes=self.n_classes_,
            mlp_ratio=self.config.get("mlp_ratio", 1.0),
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
            loader = DataLoader(TabularDataset(num_tensor, cat_tensor), batch_size=batch_size)
            with torch.no_grad():
                for batch in loader:
                    batch_num = batch["input_num"]
                    batch_cat = batch["input_cat"]
                    
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