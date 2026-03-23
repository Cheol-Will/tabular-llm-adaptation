from __future__ import annotations

import os
import argparse
import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import gaussian_kde

from tabarena.utils.pickle_utils import fetch_all_pickles


EXCLUDE_KEYS = {"ag_args_ensemble", "ag_args_fit", "gpu_ids",}
CONTINUOUS_HPS = {"dropout", "lora_dropout", "lora_lr", "lr", "weight_decay", }
DISCRETE_HPS   = {"batch_size", "lora_rank", "lora_alpha"}


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_flat_hps(hp_dict: dict) -> dict:
    """Flatten hyperparameters, dropping structural keys."""
    flat = {}
    for k, v in hp_dict.items():
        if k in EXCLUDE_KEYS:
            continue
        if isinstance(v, dict):
            continue
        flat[k] = v
    return flat


def build_records(file_paths: list[Path]) -> pd.DataFrame:
    """Load all pickles and return a flat DataFrame with one row per config run."""
    records = []
    for fp in file_paths:
        obj = load_pickle(fp)
        dataset  = obj["task_metadata"].get("name") or obj["task_metadata"].get("dataset", str(fp))
        metric_error     = obj.get("metric_error")
        metric_error_val = obj.get("metric_error_val")
        metric   = obj.get("metric", "")
        hps = extract_flat_hps(obj.get("method_metadata", {}).get("hyperparameters", {}))
        row = {
            "dataset": dataset,
            "metric": metric,
            "metric_error": metric_error,
            "metric_error_val": metric_error_val,
            **hps,
        }
        records.append(row)
    return pd.DataFrame(records)


def get_best_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """For each dataset, pick the row with the lowest metric_error (smaller = better)."""
    idx_best = df.groupby("dataset")["metric_error"].idxmin()
    best = df.loc[idx_best].reset_index(drop=True)
    return best


def save_best_csv(best: pd.DataFrame, out_path: Path) -> None:
    best.to_csv(out_path, index=False)
    print(f"Saved best configs CSV: {out_path}")

def plot_hp_distributions(best: pd.DataFrame, out_path: Path) -> None:
    hp_cols = [c for c in best.columns if c not in {"dataset", "metric", "metric_error", "metric_error_val"}]

    valid_hp_cols = []
    for hp in hp_cols:
        try:
            best[hp].dropna().astype(float)
            valid_hp_cols.append(hp)
        except (ValueError, TypeError):
            print(f"Skipping non-numeric HP column: {hp}")
    hp_cols = valid_hp_cols

    if not hp_cols:
        print("No hyperparameter columns found, skipping plot.")
        return

    n_cols = 3
    n_rows = int(np.ceil(len(hp_cols) / n_cols))

    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    # fig.suptitle("Best HPO Config Distributions (per dataset)", fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.35)

    COLOR_BEST = "#e05c3a"

    def fmt_val(v: float) -> str:
        if v == 0:
            return "0"
        abs_v = abs(v)
        if abs_v >= 1000:
            return f"{v:.0f}"
        elif abs_v >= 1:
            return f"{v:.2f}"
        elif abs_v >= 0.01:
            return f"{v:.4f}"
        else:
            return f"{v:.2e}"

    for i, hp in enumerate(hp_cols):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        best_vals = best[hp].dropna().astype(float)

        if hp in DISCRETE_HPS or (hp not in CONTINUOUS_HPS and best_vals.nunique() <= 8):
            # --- bar chart ---
            counts = best_vals.value_counts().sort_index()
            x = np.arange(len(counts))
            ax.bar(x, counts.values, color=COLOR_BEST, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([fmt_val(v) for v in counts.index], rotation=30, ha="right")
            ax.set_ylabel("count")

        else:
            # --- KDE + rug marks ---
            use_log = best_vals.min() > 0 and best_vals.max() / (best_vals.min() + 1e-12) > 100

            if use_log:
                log_vals = np.log10(best_vals)
                kde = gaussian_kde(log_vals, bw_method=0.4)
                xs_log = np.linspace(log_vals.min(), log_vals.max(), 300)
                xs = 10 ** xs_log
                ys = kde(xs_log)
                ax.set_xscale("log")
            else:
                std = best_vals.std()
                bw = 0.4 if std == 0 else min(0.6, (best_vals.max() - best_vals.min()) * 0.2 / std)
                kde = gaussian_kde(best_vals, bw_method=bw)
                xs = np.linspace(best_vals.min(), best_vals.max(), 300)
                ys = kde(xs)

            ax.plot(xs, ys, color=COLOR_BEST, linewidth=2)
            ax.fill_between(xs, ys, alpha=0.3, color=COLOR_BEST)

            # rug marks
            y_offset = -(ys.max() * 0.05)
            ax.scatter(
                best_vals,
                np.full_like(best_vals, y_offset),
                color=COLOR_BEST, s=60, zorder=5, clip_on=False, marker="|", linewidths=2,
            )

            ax.set_ylabel("density")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_val(v)))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        ax.set_title(hp, fontsize=18)
        # ax.set_xlabel(hp)

    # hide unused subplots
    for j in range(len(hp_cols), n_rows * n_cols):
        fig.add_subplot(gs[j // n_cols, j % n_cols]).set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved HP distribution plot: {out_path}")


def main(args):
    base_dir = Path(__file__).parent / "results" / args.exp_name / args.model
    eval_dir = Path(__file__).parent / "evals" / args.exp_name
    os.makedirs(eval_dir, exist_ok=True)

    print(f"Scanning {base_dir} for results.pkl ...")
    file_paths = fetch_all_pickles(dir_path=str(base_dir), suffix="results.pkl")
    print(f"Found {len(file_paths)} pickle files.")

    df = build_records(file_paths)
    print(f"Total runs loaded: {len(df)}")
    print(df[["dataset", "metric_error"]].head())

    # best config per dataset
    best = get_best_per_dataset(df)
    print(f"\nBest config per dataset ({len(best)} datasets):")
    print(best[["dataset", "metric_error"]].to_string(index=False))

    # --- save CSV ---
    csv_all  = eval_dir / f"{args.model}_all_runs.csv"
    csv_best = eval_dir / f"{args.model}_best_per_dataset.csv"
    df.to_csv(csv_all, index=False)
    save_best_csv(best, csv_best)

    # --- save plots ---
    plot_path = eval_dir / f"{args.model}_hp_distributions.png"
    plot_hp_distributions(best, plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()
    main(args)