from __future__ import annotations

import os
import argparse
import pickle
from pathlib import Path

import openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

from autogluon.tabular import TabularPredictor
from tabarena.utils.pickle_utils import fetch_all_pickles

# TabArena custom model
import sys
sys.path.insert(0, str(Path(__file__).parent))
from custom_models.tfmllm.wrapper import TFMLLMModel


EXCLUDE_KEYS = {"ag_args_ensemble", "ag_args_fit", "gpu_ids"}
EXCLUDE_KEYS_REFIT = {"ag_args_ensemble", "ag_args_fit", "gpu_ids"}


# ──────────────────────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────────────────────

def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_flat_hps(hp_dict: dict) -> dict:
    flat = {}
    for k, v in hp_dict.items():
        if k in EXCLUDE_KEYS:
            continue
        if isinstance(v, dict):
            continue
        flat[k] = v
    return flat


def build_records(file_paths: list[Path]) -> pd.DataFrame:
    records = []
    for fp in file_paths:
        obj     = load_pickle(fp)
        dataset = obj["task_metadata"].get("name") or obj["task_metadata"].get("dataset", str(fp))
        tid     = obj["task_metadata"].get("tid")
        hps     = extract_flat_hps(obj.get("method_metadata", {}).get("hyperparameters", {}))
        records.append({
            "dataset":      dataset,
            "tid":          tid,
            "metric":       obj.get("metric", ""),
            "metric_error": obj.get("metric_error"),
            **hps,
        })
    return pd.DataFrame(records)


def get_best_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    idx_best = df.groupby("dataset")["metric_error"].idxmin()
    return df.loc[idx_best].reset_index(drop=True)


def load_openml_data(tid: int) -> tuple[pd.DataFrame, pd.Series, str]:
    task    = openml.tasks.get_task(tid)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=task.target_name)
    return X, y, task.target_name


# ──────────────────────────────────────────────────────────────
# TabularAnalysis
# ──────────────────────────────────────────────────────────────

class TabularAnalysis(TabularPredictor):
    """
    TabularPredictor + layer-wise embedding extraction & t-SNE visualization.
    """

    def extract_layer_embeddings(
        self,
        data: pd.DataFrame,
        model_name: str | None = None,
    ) -> dict[str, np.ndarray]:

        if model_name is None:
            all_models = self.model_names()
            tfmllm_models = [m for m in all_models if "TFMLLM" in m and "Ensemble" not in m]
            model_name = tfmllm_models[0] if tfmllm_models else None
            if model_name is None:
                raise ValueError(f"No TFMLLM model found. Available: {all_models}")
            print(f"Auto-selected model: {model_name}")

        ag_model = self._trainer.load_model(model_name)
        if hasattr(ag_model, "models"):
            ag_model = ag_model.models[0]

        impl        = ag_model.model          # TFMLLMImplementation
        torch_model = impl.model              # TFMLLM (LoRA wrapped nn.Module)
        torch_model.eval()

        # 전처리
        X_processed = self._learner.transform_features(data)
        X_inner     = ag_model.preprocess(X_processed)
        X_num, X_cat, _ = impl._prepare_data(X_inner)
        X_num = X_num.to(impl.device_)
        X_cat = X_cat.to(impl.device_)

        # transformer layer 이름 탐색
        # backbone은 보통 base_model.model.backbone.layers 또는 .model.layers
        layer_outputs: dict[str, np.ndarray] = {}

        def make_hook(name: str):
            def hook_fn(module, input, output):
                # output: (N, num_cols, hidden_dim) or tuple
                if isinstance(output, torch.Tensor):
                    t = output
                elif isinstance(output, (tuple, list)):
                    t = next((o for o in output if isinstance(o, torch.Tensor)), None)
                    if t is None:
                        return
                else:
                    return

                if t.ndim == 3:
                    # pooling
                    pooled = t.mean(dim=1) # (N, hidden_dim)
                    layer_outputs[name] = pooled.detach().cpu().float().numpy()
            return hook_fn

        # transformer layer만 hook 등록 (짝수 index)
        hooks = []
        transformer_layers = []
        for name, module in torch_model.named_modules():
            parts = name.split(".")
            if (
                len(parts) == 4
                and parts[0] == "base_model"
                and parts[1] == "model"
                and parts[2] == "backbone"
                and parts[3].isdigit()
            ):
                transformer_layers.append((int(parts[3]), name, module))

        transformer_layers.sort(key=lambda x: x[0])
        print(f"Found {len(transformer_layers)} transformer layers total.")

        for layer_idx, name, module in transformer_layers:
            if layer_idx % 3 == 0:
                hooks.append(module.register_forward_hook(make_hook(f"layer_{layer_idx:02d}")))
                print(f"  Hooking layer_{layer_idx:02d}: {name}")

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                torch_model(X_num, X_cat)

        for h in hooks:
            h.remove()

        # 등록된 layer가 없으면 전체 모듈 이름 출력
        if not layer_outputs:
            print("No transformer layers found. Available module names:")
            for name, _ in torch_model.named_modules():
                print(f"  {name}")

        return layer_outputs


    def plot_tsne_per_layer(
        self,
        data: pd.DataFrame,
        label_col: str,
        model_name: str | None = None,
        layers: list[str] | None = None,
        output_path: str | Path | None = None,
        tsne_kwargs: dict | None = None,
        n_cols: int = 3,
    ) -> None:
        X_data = data.drop(columns=[label_col])
        y_data = data[label_col]

        print("Extracting layer embeddings...")
        layer_embeddings = self.extract_layer_embeddings(
            data=X_data,
            model_name=model_name,
        )

        if not layer_embeddings:
            print("No embeddings extracted. Check model structure.")
            return

        # 시각화할 레이어 필터링
        if layers is not None:
            layer_embeddings = {k: v for k, v in layer_embeddings.items() if k in layers}

        # 2D 이상인 레이어만
        valid_layers = {
            name: emb
            for name, emb in layer_embeddings.items()
            if emb.ndim == 2 and emb.shape[1] >= 2
        }

        if not valid_layers:
            print("No valid 2D+ embedding layers found.")
            print("Available layers & shapes:")
            for k, v in layer_embeddings.items():
                print(f"  {k}: shape={v.shape if hasattr(v, 'shape') else type(v)}")
            return

        print(f"Found {len(valid_layers)} valid layers. Running t-SNE...")

        # class 색상
        classes     = sorted(y_data.unique())
        n_classes   = len(classes)
        class_to_id = {c: i for i, c in enumerate(classes)}
        print(f"Classes ({n_classes}): {classes}")  
        y_ids       = y_data.map(class_to_id).values
        cmap = cm.get_cmap("tab10", n_classes)

        if n_classes == 2:
            colors = ["#e05c3a", "#4a90d9"]  # red, blue
        else:
            colors = [plt.colormaps.get_cmap("tab10")(i) for i in range(n_classes)]


        # subplot 구성
        n_rows = max(1, int(np.ceil(len(valid_layers) / n_cols)))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False,
        )
        axes = axes.flatten()

        _tsne_kwargs = dict(n_components=2, random_state=42, perplexity=min(30, len(y_data) - 1), max_iter=1000)
        if tsne_kwargs:
            _tsne_kwargs.update(tsne_kwargs)

        for ax, (layer_name, emb) in zip(axes, valid_layers.items()):
            print(f"  t-SNE: {layer_name}  shape={emb.shape}")
            tsne   = TSNE(**_tsne_kwargs)
            emb_2d = tsne.fit_transform(emb)

            for cls in classes:
                cid  = class_to_id[cls]
                mask = y_ids == cid
                ax.scatter(
                    emb_2d[mask, 0], emb_2d[mask, 1],
                    color=colors[cid],
                    label=str(cls),
                    # alpha=1.0,
                    s=20,
                    edgecolors="none",
                )

            ax.set_title(layer_name, fontsize=24)
            ax.set_xticks([])
            ax.set_yticks([])
            if n_classes <= 10:
                ax.legend(fontsize=18, markerscale=1.5, loc="best")

        for ax in axes[len(valid_layers):]:
            ax.set_visible(False)

        # fig.suptitle(
        #     f"Layer-wise Embedding Space — t-SNE\n(dataset: {data.shape})",
        #     fontsize=13, fontweight="bold",
        # )
        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {output_path}")
        else:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _get_torch_module(ag_model) -> nn.Module:
        """
        """
        # BAG → 첫 번째 fold
        if hasattr(ag_model, "models"):
            ag_model = ag_model.models[0]

        # TFMLLMModel.model = TFMLLMImplementation
        impl = getattr(ag_model, "model", None)
        if impl is None:
            raise AttributeError(f"ag_model has no .model attr. type={type(ag_model)}")

        if isinstance(impl, nn.Module):
            return impl

        inner = getattr(impl, "model", None)
        if isinstance(inner, nn.Module):
            return inner

        for attr in ["net", "_model", "_net", "backbone", "transformer", "encoder"]:
            obj = getattr(impl, attr, None)
            if isinstance(obj, nn.Module):
                return obj

        raise AttributeError(
            f"Cannot find nn.Module.\n"
            f"  ag_model type : {type(ag_model)}\n"
            f"  impl type     : {type(impl)}\n"
            f"  impl attrs    : {[a for a in dir(impl) if not a.startswith('__')]}"
        )


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main(args):
    base_dir = Path(__file__).parent / "results" / args.exp_name / args.model
    eval_dir = Path(__file__).parent / "evals"  / args.exp_name / "embedding"
    os.makedirs(eval_dir, exist_ok=True)

    print(f"Scanning {base_dir} for results.pkl ...")
    file_paths = fetch_all_pickles(dir_path=str(base_dir), suffix="results.pkl")
    print(f"Found {len(file_paths)} pickle files.")

    df   = build_records(file_paths)
    best = get_best_per_dataset(df)

    # 필터링
    if args.tid:
        best = best[best["tid"] == args.tid]
        if best.empty:
            raise ValueError(f"tid={args.tid} not found in results.")
    elif args.dataset:
        best = best[best["dataset"] == args.dataset]
        if best.empty:
            raise ValueError(f"Dataset '{args.dataset}' not found.")

    for _, row in best.iterrows():
        dataset = row["dataset"]
        tid     = int(row["tid"])

        meta_cols = {"dataset", "tid", "metric", "metric_error"}
        best_hp   = {
            k: row[k]
            for k in row.index
            if k not in meta_cols
            and k not in EXCLUDE_KEYS_REFIT
            and pd.notna(row[k])
        }

        # int로 변환해야 하는 HP 목록
        INT_HPS = {"batch_size", "lora_rank", "lora_alpha", "num_epochs", "token_dim"}

        meta_cols = {"dataset", "tid", "metric", "metric_error"}
        best_hp = {}
        for k in row.index:
            if k in meta_cols or k in EXCLUDE_KEYS_REFIT or pd.isna(row[k]):
                continue
            v = row[k]
            # int로 변환
            if k in INT_HPS:
                v = int(v)
            best_hp[k] = v

        best_hp["gpu_ids"] = [0, 1]

        print(f"\n{'='*60}")
        print(f"Dataset : {dataset}  (tid={tid})")
        print(f"Best HP : {best_hp}")

        # raw data
        X, y, label = load_openml_data(tid)
        train_data  = pd.concat([X, y.rename(label)], axis=1)

        # refit
        predictor_path = eval_dir / dataset
        predictor = TabularAnalysis(
            label=label,
            path=str(predictor_path),
        ).fit(
            train_data=train_data,
            hyperparameters={TFMLLMModel: best_hp},  # 클래스 직접 전달
            num_bag_folds=0,
            num_stack_levels=0,
            verbosity=2,
        )

        # t-SNE plot
        plot_path = eval_dir / f"{dataset}_tsne.png"
        predictor.plot_tsne_per_layer(
            data=train_data,
            label_col=label,
            output_path=plot_path,
        )
        print(f"Done: {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--dataset",  type=str, default=None)
    parser.add_argument("--tid",      type=int, default=None)
    args = parser.parse_args()
    main(args)