"""Generate CSV of baseline results for LR, REALMLP_GPU, and CAT models."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pandas as pd

from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.compare import get_subset_results, TabArenaContext


TARGET_MODELS = ["LR", "REALMLP_GPU", "CAT"]

OUTPUT_PATH = Path(__file__).parent / "evals" / "baseline_results.csv"


def main():
    # Load any method to get paper results via get_subset_results
    end_to_end = EndToEnd.from_cache(methods=["FTTransformer"])
    results = end_to_end.to_results()

    with tempfile.TemporaryDirectory() as tmp:
        table = get_subset_results(
            output_dir=tmp,
            new_results=results.get_results(),
            folds=[0],
            subset=None,
        )

    # Filter to target models
    mask = table["method"].apply(
        lambda m: any(m.startswith(model) for model in TARGET_MODELS)
    )
    table = table[mask].copy()

    # Join with task_metadata to get tid
    ctx = TabArenaContext()
    tid_map = ctx.task_metadata[["dataset", "tid"]]
    table = table.merge(tid_map, on="dataset", how="left")

    # Select and rename columns
    out = table[["dataset", "tid", "method", "metric", "metric_error", "metric_error_val"]].copy()
    out = out.rename(columns={"dataset": "Dataset", "method": "model"})
    out = out.sort_values(["model", "Dataset"]).reset_index(drop=True)

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {OUTPUT_PATH}")
    print(out.head(10).to_string())


if __name__ == "__main__":
    main()
