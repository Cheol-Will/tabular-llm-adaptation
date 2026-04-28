"""
Shared utilities for analysis framework.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_model_experiments
from tabarena.utils.pickle_utils import fetch_all_pickles
from tabarena.benchmark.task.openml import OpenMLS3TaskWrapper, OpenMLTaskWrapper


EXCLUDE_KEYS = {"ag_args_ensemble", "ag_args_fit", "gpu_ids"}
CONTINUOUS_HPS = {"dropout", "lora_dropout", "lora_lr", "lr", "weight_decay"}
DISCRETE_HPS = {"batch_size", "lora_rank", "lora_alpha", "token_dim", "num_buckets"}


def load_pickle(path: Path) -> dict:
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def fetch_result_files(base_dir: Path, suffix: str = "results.pkl") -> list[Path]:
    """Fetch all result pickle files from experiment directory."""
    if not base_dir.exists():
        return []
    return fetch_all_pickles(dir_path=str(base_dir), suffix=suffix)



def extract_flat_hps(hp_dict: dict) -> dict:
    """
    Flatten hyperparameters, dropping structural keys.

    Args:
        hp_dict: Dictionary of hyperparameters

    Returns:
        Flattened dictionary with scalar values only
    """
    flat = {}
    for k, v in hp_dict.items():
        if k in EXCLUDE_KEYS:
            continue
        if isinstance(v, dict):
            continue
        flat[k] = v
    return flat


def build_records(file_paths: list[Path]) -> pd.DataFrame:
    """
    Load all pickles and return a flat DataFrame with one row per config run.

    Args:
        file_paths: List of paths to result pickle files

    Returns:
        DataFrame with columns: dataset, tid, metric, metric_error, metric_error_val, and hyperparameters
    """
    records = []
    for fp in file_paths:
        obj = load_pickle(fp)
        dataset = obj["task_metadata"].get("name") or obj["task_metadata"].get("dataset", str(fp))
        tid = obj["task_metadata"].get("tid")
        metric_error = obj.get("metric_error")
        metric_error_val = obj.get("metric_error_val")
        metric = obj.get("metric", "")
        hps = extract_flat_hps(obj.get("method_metadata", {}).get("hyperparameters", {}))

        row = {
            "dataset": dataset,
            "tid": tid,
            "metric": metric,
            "metric_error": metric_error,
            "metric_error_val": metric_error_val,
            **hps,
        }
        records.append(row)
    return pd.DataFrame(records)


def get_best_per_dataset(df: pd.DataFrame, metric_col: str = "metric_error") -> pd.DataFrame:
    """
    For each dataset, pick the row with the lowest metric value (smaller = better).

    Args:
        df: DataFrame with results
        metric_col: Column name to optimize (default: "metric_error")

    Returns:
        DataFrame with best configuration per dataset
    """
    idx_best = df.groupby("dataset")[metric_col].idxmin()
    best = df.loc[idx_best].reset_index(drop=True)
    return best



def format_value(v: float) -> str:
    """
    Format a numeric value for display.

    Args:
        v: Value to format

    Returns:
        Formatted string
    """
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


def plot_hp_distributions(
    best: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
) -> None:
    """
    Plot distribution of best hyperparameters across datasets.

    Args:
        best: DataFrame with best configurations per dataset
        out_path: Path to save the plot
        title: Optional plot title
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats import gaussian_kde
    import numpy as np

    hp_cols = [
        c for c in best.columns
        if c not in {"dataset", "tid", "metric", "metric_error", "metric_error_val"}
    ]

    # Filter numeric columns only
    valid_hp_cols = []
    for hp in hp_cols:
        try:
            best[hp].dropna().astype(float)
            valid_hp_cols.append(hp)
        except (ValueError, TypeError):
            print(f"  [INFO] Skipping non-numeric HP column: {hp}")
    hp_cols = valid_hp_cols

    if not hp_cols:
        print("  [WARN] No hyperparameter columns found, skipping plot.")
        return

    n_cols = 3
    n_rows = int(np.ceil(len(hp_cols) / n_cols))

    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.35)

    COLOR_BEST = "#e05c3a"

    for i, hp in enumerate(hp_cols):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        best_vals = best[hp].dropna().astype(float)

        # Decide if discrete or continuous
        print(f"Plotting {hp}")
        if hp in DISCRETE_HPS or (hp not in CONTINUOUS_HPS and best_vals.nunique() <= 8):
            # Bar chart for discrete HPs
            counts = best_vals.value_counts().sort_index()
            x = np.arange(len(counts))
            ax.bar(x, counts.values, color=COLOR_BEST, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([format_value(v) for v in counts.index], rotation=30, ha="right")
            ax.set_ylabel("count")

        else:
            # KDE + rug marks for continuous HPs
            use_log = best_vals.min() > 0 and best_vals.max() / (best_vals.min() + 1e-12) > 100

            if use_log:
                log_vals = np.log10(best_vals)
                kde = gaussian_kde(log_vals, bw_method=0.4)
                xs_log = np.linspace(log_vals.min(), log_vals.max(), 300)
                xs = 10 ** xs_log
                ys = kde(xs_log)
                ax.set_xscale("log")
            else:
                
                if len(best_vals) == 1:
                    continue
                std = best_vals.std()
                bw = 0.4 if std == 0 else min(0.6, (best_vals.max() - best_vals.min()) * 0.2 / std)
                kde = gaussian_kde(best_vals, bw_method=bw)
                xs = np.linspace(best_vals.min(), best_vals.max(), 300)
                ys = kde(xs)

            ax.plot(xs, ys, color=COLOR_BEST, linewidth=2)
            ax.fill_between(xs, ys, alpha=0.3, color=COLOR_BEST)

            # Rug marks
            y_offset = -(ys.max() * 0.05)
            ax.scatter(
                best_vals,
                np.full_like(best_vals, y_offset),
                color=COLOR_BEST,
                s=60,
                zorder=5,
                clip_on=False,
                marker="|",
                linewidths=2,
            )

            ax.set_ylabel("density")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: format_value(v)))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        ax.set_title(hp, fontsize=18)

    # Hide unused subplots
    for j in range(len(hp_cols), n_rows * n_cols):
        fig.add_subplot(gs[j // n_cols, j % n_cols]).set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] HP distribution plot: {out_path}")



def analyze_hpo(
    model: str,
    exp_name: str,
    output_dir: Path,
) -> None:
    """
    Analyze HPO results: aggregate all runs, find best configs per dataset,
    and visualize hyperparameter distributions.

    Args:
        model: Model name (e.g., "TFMLLM", "LLMBaseline")
        exp_name: Experiment name (directory under results/)
        output_dir: Directory to save analysis outputs
    """
    print(f"\n{'='*80}")
    print(f"Analysis: HPO Results")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Experiment: {exp_name}")

    base_dir = Path(__file__).parent / "results" / exp_name / model

    if not base_dir.exists():
        print(f"  [ERROR] Results directory not found: {base_dir}")
        return

    print(f"\nScanning {base_dir} for results.pkl ...")
    file_paths = fetch_result_files(base_dir, suffix="results.pkl")
    print(f"  Found {len(file_paths)} pickle files.")

    if not file_paths:
        print("  [ERROR] No results found. Exiting.")
        return

    # Load all runs
    df = build_records(file_paths)
    print(f"\nTotal runs loaded: {len(df)}")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"\nSample data:")
    print(df[["dataset", "metric", "metric_error"]].head(10).to_string(index=False))

    # Get best config per dataset
    best = get_best_per_dataset(df)
    print(f"\n{'─'*80}")
    print(f"Best config per dataset ({len(best)} datasets):")
    print(f"{'─'*80}")
    print(best[["dataset", "metric", "metric_error"]].to_string(index=False))

    # Save CSVs
    print(f"\n{'─'*80}")
    print("Saving results...")
    print(f"{'─'*80}")

    csv_all = output_dir / f"{model}_all_runs.csv"
    csv_best = output_dir / f"{model}_best_per_dataset.csv"

    df.to_csv(csv_all, index=False)
    print(f"  [SAVED] All runs CSV: {csv_all}")

    best.to_csv(csv_best, index=False)
    print(f"  [SAVED] Best configs CSV: {csv_best}")

    # Plot hyperparameter distributions
    plot_path = output_dir / f"{model}_hp_distributions.png"
    plot_hp_distributions(
        best,
        plot_path,
        title=f"Best HPO Config Distributions — {model}",
    )

    print(f"\n{'='*80}")
    print(f"HPO Analysis Complete!")
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Regression Distribution Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_reg_dist(
    model: str,
    exp_name: str,
    task_id: str,
    output_dir: Path,
) -> None:
    """
    Plot the distribution of ground truth y_test vs predicted values
    for a given task, grouped by config (folds pooled within each config).

    Directory structure expected:
        results/{exp_name}/{model}/data/{config_name}/{task_id}/{fold}/results.pkl

    Args:
        model: Model name (e.g., TFMLLM)
        exp_name: Experiment name (directory under results/)
        task_id: Task ID to analyze
        output_dir: Directory to save the output plot
    """
    base_dir = Path(__file__).parent / "results" / exp_name / model

    # Structure: data/{config_name}/{task_id}/{fold}/results.pkl
    pkl_paths = sorted(base_dir.glob(f"data/*/{task_id}/*/results.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(
            f"No results.pkl found for task_id={task_id} under {base_dir}/data/"
        )

    print(f"Found {len(pkl_paths)} result file(s) for task_id={task_id}")

    # Group by config_name (pool folds within each config)
    # path structure: .../data/{config_name}/{task_id}/{fold}/results.pkl
    #   parts[-4] = config_name, parts[-3] = task_id, parts[-2] = fold
    config_data: dict[str, dict] = {}

    for path in pkl_paths:
        config_name = path.parts[-4]
        fold_name = path.parts[-2]

        with open(path, "rb") as f:
            obj = pickle.load(f)

        sim = obj.get("simulation_artifacts", None)
        if sim is None:
            print(f"  [SKIP] No simulation_artifacts: {config_name}/{fold_name}")
            continue

        y_test = np.asarray(sim["y_test"]).ravel()
        pred_raw = sim["pred_proba_dict_test"]
        if isinstance(pred_raw, dict):
            # BAG: average predictions across bag members
            y_pred = np.mean([np.asarray(v).ravel() for v in pred_raw.values()], axis=0)
        else:
            y_pred = np.asarray(pred_raw).ravel()

        y_val = np.asarray(sim["y_val"]).ravel() if "y_val" in sim else None

        if config_name not in config_data:
            config_data[config_name] = {"y_test": [], "y_pred": [], "y_val": []}
        config_data[config_name]["y_test"].append(y_test)
        config_data[config_name]["y_pred"].append(y_pred)
        if y_val is not None:
            config_data[config_name]["y_val"].append(y_val)
        print(f"  [OK] {config_name} / {fold_name}: n={len(y_test)}")

    if not config_data:
        raise ValueError("No valid simulation_artifacts found in any pkl file.")

    # Pool folds within each config
    config_names = sorted(config_data.keys())
    config_y_test = [np.concatenate(config_data[c]["y_test"]) for c in config_names]
    config_y_pred = [np.concatenate(config_data[c]["y_pred"]) for c in config_names]
    config_y_val = [
        np.concatenate(config_data[c]["y_val"]) if config_data[c]["y_val"] else None
        for c in config_names
    ]

    # At most 10 configs, laid out as 2 columns x 5 rows
    config_names = config_names[:10]
    config_y_test = config_y_test[:10]
    config_y_pred = config_y_pred[:10]
    config_y_val = config_y_val[:10]

    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    for i, (name, y_true, y_pred, y_val) in enumerate(zip(config_names, config_y_test, config_y_pred, config_y_val)):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        bins = 40
        ax.hist(y_true, bins=bins, alpha=0.6, label="y_test", color="steelblue", density=True)
        ax.hist(y_pred, bins=bins, alpha=0.6, label="y_pred", color="tomato", density=True)
        if y_val is not None:
            ax.hist(y_val, bins=bins, alpha=0.6, label="y_val", color="mediumseagreen", density=True)
        n_folds = len(config_data[name]["y_test"])
        ax.set_title(f"{name} ({n_folds} fold(s))")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_yticks([])
        ax.legend()

    # Hide any unused subplots
    for j in range(len(config_names), n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Regression Distribution — {model} | task_id={task_id}", fontsize=13)
    plt.tight_layout()

    out_path = output_dir / f"reg_dist_{model}_{task_id}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[SAVED] {out_path}")


def analyze_attn_map(
    args,
    model: str,
    exp_name: str,
    task_id: str,
    output_dir: Path,
    model_cls_name: str = None,
) -> None:
    
    # construct experiment
    model_experiments = get_model_experiments(args, model, exp_name, num_random_configs=0, model_cls_name=model_cls_name)
    for model_experiment in model_experiments:
        
        # train-test
        task = OpenMLS3TaskWrapper.from_task_id(
            task_id=task_id,
            # s3_dataset_cache=s3_kwargs["dataset_cache"],
        )
        out = model_experiment.run(
            task=task_id,
            fold=[0,0],
            ignore_cache=False,
            debug_mode=False,
        )

    
        # extract attn map


    # load dataset
    # 
    # define model with default hp config
    # 
    # train the model
    # 
    # extract attn map following model types.
    # 
    # Note that plotting logic is different for each model.    
    
    return