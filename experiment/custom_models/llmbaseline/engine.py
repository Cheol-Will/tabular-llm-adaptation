from __future__ import annotations

import gc
import os
import logging
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
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

import wandb
from dotenv import load_dotenv
from transformers import AutoTokenizer
from autogluon.core.metrics import compute_metric
from peft import LoraConfig, get_peft_model

from .model import (
    LLMBaseline,
    LLMBaselineBidirectional,
    LLMBaselineBidirectionalPooling,
    LLMBaselinePooling
)
from dataset.dataloader import serialize_data, TextLabelDataset

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

TaskType = Literal["regression", "binclass", "multiclass"]
logger = logging.getLogger(__name__)



def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _evaluate_worker(
    model: nn.Module,
    val_loader: DataLoader,
    task_type: TaskType,
    y_mean: float,
    y_std: float,
    early_stopping_metric,
    device: torch.device,
) -> float:
    """Run evaluation on rank-0 using the unwrapped model (no DDP sync needed)."""
    model.eval()
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            out = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            ).cpu().float()
            all_outputs.append(out)
            all_labels.append(batch["label"])
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
    label_token_ids: list[int] | None,
    label_texts: list[str] | None,
    y_mean: float,
    y_std: float,
    train_dataset: TextLabelDataset,
    val_dataset: TextLabelDataset,
    early_stopping_metric,
    model_save_path: str,
    start_time: float,
    time_to_fit_in_seconds: float | None,
    master_port: int,
    task_id: int,
    use_wandb: bool,
    project_name: str = 'llmbaseline',
    model_cls: nn.Module = LLMBaseline,
):
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    model_name = config.get("model_name", "Qwen/Qwen2.5-0.5B")
    is_regression = task_type == "regression"

    # Build model
    model = model_cls(
        model_name=model_name,
        num_classes=n_classes,
        task_type=task_type,
        label_token_ids=label_token_ids,
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

    # DataLoaders
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 64),
        sampler=train_sampler,
    )
    # Only rank 0 evaluates; uses model.module directly (no DDP all-reduce)
    val_loader = (
        DataLoader(val_dataset, batch_size=config.get("eval_batch_size", 128), shuffle=False)
        if rank == 0
        else None
    )

    loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get("lora_lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    patience = config.get("patience", 16)
    remaining_patience = patience
    best_val_score = -np.inf
    best_state: dict | None = None

    if rank == 0 and use_wandb:
        wandb.init(
            project=f"{project_name}-tabarena",
            name=f"task_{task_id}_{project_name}",
            group=str(task_id),
            config={
                "task_id": task_id,
                "model_name": model_name,
                "task_type": task_type,
                "lora_rank": config.get("lora_rank", 8),
                "lora_alpha": config.get("lora_alpha", 32),
                "lora_lr": config.get("lora_lr", 1e-4),
                "batch_size": config.get("batch_size", 64),
                "max_length": config.get("max_length", 256),
                "weight_decay": config.get("weight_decay", 1e-5),
                "num_epochs": config.get("num_epochs", 200),
                "patience": config.get("patience", 16),
            },
            reinit="finish_previous",
            settings=wandb.Settings(silent=True),  
        )

    # Epoch-0 baseline (rank 0 only; result broadcast to sync all ranks)
    if rank == 0:
        best_val_score = _evaluate_worker(
            model.module, val_loader, task_type, y_mean, y_std, early_stopping_metric, device
        )
        best_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
        logger.info(f"Epoch 000 (baseline): Val Score = {best_val_score:.4f}")

    score_buf = torch.tensor([best_val_score], device=device)
    dist.broadcast(score_buf, src=0)
    best_val_score = score_buf.item()

    num_epochs = config.get("num_epochs", 100)
    epoch_iter = tqdm(range(num_epochs), desc="Training") if rank == 0 else range(num_epochs)

    for epoch in epoch_iter:
        if time_to_fit_in_seconds and (time.time() - start_time) >= time_to_fit_in_seconds:
            break

        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_y = batch["label"].to(device)

            optimizer.zero_grad()
            # output = model(input_ids, attention_mask)
            output = model(input_ids, attention_mask).float()  
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Rank 0 evaluates; rank 1 waits at the broadcast below
        val_score = -np.inf
        if rank == 0:
            val_score = _evaluate_worker(
                model.module, val_loader, task_type, y_mean, y_std, early_stopping_metric, device
            )
            avg_loss = train_loss / len(train_loader)
            if isinstance(epoch_iter, tqdm):
                epoch_iter.set_postfix({
                    "train_loss": f"{avg_loss:.4f}",
                    "val": f"{val_score:.4f}",
                    "best": f"{best_val_score:.4f}",
                })
            logger.info(f"Epoch {epoch:03d}: Val Score = {val_score:.4f} (Best: {best_val_score:.4f})")
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_score": val_score,
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


class LLMBaselineImplementation:

    def __init__(self, early_stopping_metric: Scorer, model_cls: type, **config):
        self.config = config
        self.early_stopping_metric = early_stopping_metric
        self.model: LLMBaseline | None = None
        self.model_cls = model_cls
        self.tokenizer = None

        self.task_type_: TaskType | None = None
        self.device_: torch.device | None = None
        self.n_classes_: int | None = None
        self.label_token_ids_: list[int] | None = None
        self.label_texts_: list[str] | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        load_dotenv()
        self.use_wandb = bool(os.getenv("WANDB_API_KEY"))
        if self.use_wandb:
            print("✅ WandB API Key loaded from .env")
        print(f"Using model_cls: {model_cls.__name__}")

    def _check_is_fitted(self):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

    def _get_label_token_ids(self, unique_labels: list) -> tuple[list[str], list[int]]:
        label_texts = [str(l) for l in unique_labels]
        label_token_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)[0]
            for t in label_texts
        ]
        return label_texts, label_token_ids

    def _make_loader(self, X: pd.DataFrame, y: pd.Series | None, shuffle: bool) -> DataLoader:
        target_name = y.name if y is not None else "target"
        texts = serialize_data(X, target_name)

        labels = None
        if y is not None:
            if self.task_type_ == "regression":
                labels = y.tolist()
            else:
                labels = [self.label_texts_.index(str(v)) for v in y.tolist()]

        dataset = TextLabelDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            task_type=self.task_type_,
            labels=labels,
            label_token_ids=self.label_token_ids_,
            max_length=self.config.get("max_length", 128),
            y_mean=self.y_mean_,
            y_std=self.y_std_,
        )
        batch_size = (
            self.config.get("batch_size", 64)
            if y is not None
            else self.config.get("eval_batch_size", 128)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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

        task_id = self.config.get("task_id") or int(os.getenv("CURRENT_TASK_ID", "0"))
        start_time = time.time()

        gpu_ids = self.config.get("gpu_ids", [0])
        self.device_ = torch.device(f"cuda:{gpu_ids[0]}")

        random_state = self.config.get("random_state", None)
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        problem_type = self.config["problem_type"]
        self.task_type_ = "binclass" if problem_type == "binary" else problem_type
        model_name = self.config.get("model_name", "Qwen/Qwen2.5-0.5B")
        is_regression = self.task_type_ == "regression"

        # Tokenizer (main process; workers reconstruct from model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Labels
        if is_regression:
            self.n_classes_ = 1
            self.y_mean_ = float(y_train.mean())
            self.y_std_ = float(y_train.std())
        else:
            unique_labels = sorted(y_train.unique().tolist())
            self.label_texts_, self.label_token_ids_ = self._get_label_token_ids(unique_labels)
            self.n_classes_ = len(unique_labels)

        # Pre-tokenize datasets in main process (TextLabelDataset stores only tensors → picklable)
        max_length = self.config.get("max_length", 128)
        train_labels = (
            y_train.tolist() if is_regression
            else [self.label_texts_.index(str(v)) for v in y_train.tolist()]
        )
        val_labels = (
            y_val.tolist() if is_regression
            else [self.label_texts_.index(str(v)) for v in y_val.tolist()]
        )
        train_dataset = TextLabelDataset(
            texts=serialize_data(X_train, y_train.name),
            tokenizer=self.tokenizer, task_type=self.task_type_,
            labels=train_labels, label_token_ids=self.label_token_ids_,
            max_length=max_length, y_mean=self.y_mean_, y_std=self.y_std_,
        )
        val_dataset = TextLabelDataset(
            texts=serialize_data(X_val, y_val.name),
            tokenizer=self.tokenizer, task_type=self.task_type_,
            labels=val_labels, label_token_ids=self.label_token_ids_,
            max_length=max_length, y_mean=self.y_mean_, y_std=self.y_std_,
        )

        # Spawn DDP workers
        world_size = len(gpu_ids)
        master_port = _find_free_port()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model_save_path = f.name

        mp.spawn(
            _ddp_worker,
            args=(
                world_size, gpu_ids, self.config, self.task_type_, self.n_classes_,
                self.label_token_ids_, self.label_texts_, self.y_mean_, self.y_std_,
                train_dataset, val_dataset, self.early_stopping_metric,
                model_save_path, start_time, time_to_fit_in_seconds,
                master_port, task_id, self.use_wandb, self.model_cls.__name__, self.model_cls
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
        self.model = self.model_cls(
            model_name=model_name,
            num_classes=self.n_classes_,
            task_type=self.task_type_,
            label_token_ids=self.label_token_ids_,
        ).to(self.device_)
        self.model = get_peft_model(self.model, lora_config)
        if os.path.exists(model_save_path):
            state = torch.load(model_save_path, map_location=self.device_)
            self.model.load_state_dict(state)
            os.unlink(model_save_path)

        gc.collect()
        torch.cuda.empty_cache()

    def _get_raw_outputs(self, X: pd.DataFrame) -> torch.Tensor:
        self._check_is_fitted()
        self.model.eval()
        loader = self._make_loader(X, y=None, shuffle=False)
        outputs = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(
                    batch["input_ids"].to(self.device_),
                    batch["attention_mask"].to(self.device_),
                ).cpu().float()
                outputs.append(out)
        return torch.cat(outputs, dim=0)

    def predict_raw(self, X: pd.DataFrame) -> torch.Tensor:
        return self._get_raw_outputs(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._get_raw_outputs(X)
        if self.task_type_ == "regression":
            return raw.squeeze(-1).numpy() * self.y_std_ + self.y_mean_
        return raw.argmax(dim=-1).numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._get_raw_outputs(X)
        if self.task_type_ == "regression":
            return raw.squeeze(-1).numpy() * self.y_std_ + self.y_mean_
        probas = torch.softmax(raw, dim=-1).numpy()
        return probas[:, 1] if self.task_type_ == "binclass" else probas
