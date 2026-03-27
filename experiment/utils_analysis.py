from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

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





METRIC_DIRECTION: dict[str, str] = {
    "roc_auc": "max",
    "rmse": "min",
    "log_loss": "min",
}

METRIC_ARROW: dict[str, str] = {
    "roc_auc": "uparrow",
    "rmse": "downarrow",
    "log_loss": "downarrow",
}

METRIC_ORDER = ["roc_auc", "rmse", "log_loss"]

CATEGORY_ORDER = ["ML", "DL", "Foundation"]

BOTTOM_METHODS = ["FTTransformer", "TFMLLM", "LLMBaseline"]

METHOD_CATEGORY: dict[str, str] = {
    "CAT":             "ML",
    "EBM":             "ML",
    "GBM":             "ML",
    "KNN":             "ML",
    "LR":              "ML",
    "RF":              "ML",
    "XGB":             "ML",
    "XT":              "ML",
    "FASTAI":          "DL",
    "MNCA_GPU":        "DL",
    "NN_TORCH":        "DL",
    "REALMLP_GPU":     "DL",
    "TABM_GPU":        "DL",
    "XRFM_GPU":        "DL",
    "REALTABPFN-V2.5": "Foundation",
    "TABDPT_GPU":      "Foundation",
    "TABPFNV2_GPU":    "Foundation",
    "FTTransformer":   "DL",
    "TFMLLM":          "Foundation",
    "LLMBaseline":     "Foundation",
}

# Alias
DATASET_ABBREV: dict[str, str] = {
    "Amazon_employee_access":              "Amazon",
    "QSAR_fish_toxicity":                  "QSAR-Fish",
    "airfoil_self_noise":                  "Airfoil",
    "anneal":                              "Anneal",
    "blood-transfusion-service-center":    "Blood",
    "concrete_compressive_strength":       "Concrete",
    "credit-g":                            "Credit-G",
    "diabetes":                            "Diabetes",
    "maternal_health_risk":                "Maternal",
    "qsar-biodeg":                         "QSAR-Bio",
    "APSFailure":                          "APS",
    "Bank_Customer_Churn":                 "BankChurn",
    "Bioresponse":                         "Bioresp",
    "Diabetes130US":                       "Diab130",
    "GiveMeSomeCredit":                    "Credit",
    "HR_Analytics_Job_Change_of_Data_Scientists": "HR-Job",
    "SDSS17":                              "SDSS17",
    "bank-marketing":                      "BankMkt",
    "churn":                               "Churn",
    "coil2000_insurance_policies":         "Coil2000",
    "credit_card_clients_default":         "CreditCC",
    "heloc":                               "HELOC",
    "hiva_agnostic":                       "HIVA",
    "in_vehicle_coupon_recommendation":    "Coupon",
    "jm1":                                 "JM1",
    "kddcup09_appetency":                  "KDD09",
    "online_shoppers_intention":           "OnlineShop",
    "polish_companies_bankruptcy":         "Polish",
    "seismic-bumps":                       "Seismic",
    "splice":                              "Splice",
    "students_dropout_and_academic_success": "Students",
    "taiwanese_bankruptcy_prediction":     "TaiwanBnk",
    "website_phishing":                    "Phishing",
    "diamonds":                            "Diamonds",
    "healthcare_insurance_expenses":       "Insurance",
    "houses":                              "Houses",
    "miami_housing":                       "Miami",
    "physiochemical_protein":              "Protein",
    "superconductivity":                   "Supercon",
    "E-CommereShippingData":               "EComShip",
    "Fitness_Club":                        "Fitness",
    "Food_Delivery_Time":                  "FoodDel",
    "MIC":                                 "MIC",
    "Marketing_Campaign":                  "MktCamp",
    "NATICUSdroid":                        "NATICUS",
    "QSAR-TID-11":                         "QSAR-TID",
    "Another-Dataset-on-used-Fiat-500":    "Fiat500",
    "Is-this-a-good-customer":             "GoodCust",
    "customer_satisfaction_in_airline":    "Airline",
    "hazelnut-spread-contaminant-detection": "Hazelnut",
    "wine_quality":                        "Wine",
}


def escape_latex(s: str) -> str:
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("_", "\\_"),
        ("%", "\\%"),
        ("&", "\\&"),
        ("#", "\\#"),
        ("$", "\\$"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for char, repl in replacements:
        s = s.replace(char, repl)
    return s


def clean_method_name(name: str) -> str:
    for suffix in [" (tuned + ensemble)", " (tuned)", " (default)"]:
        name = name.replace(suffix, "")
    return name


def sort_columns_by_metric(columns: pd.Index, dataset_metric_map: pd.Series) -> list[str]:
    metric_to_datasets: dict[str, list[str]] = {m: [] for m in METRIC_ORDER}
    for col in columns:
        metric = dataset_metric_map.get(col, None)
        if metric in metric_to_datasets:
            metric_to_datasets[metric].append(col)
    sorted_cols = []
    for metric in METRIC_ORDER:
        sorted_cols.extend(sorted(metric_to_datasets[metric]))
    return sorted_cols


def sort_index_bottom_methods(index: pd.Index) -> list[str]:
    bottom = [m for m in BOTTOM_METHODS if m in index]
    rest   = sorted([m for m in index if m not in BOTTOM_METHODS])
    return rest + bottom

def sort_index_by_category(index: pd.Index) -> list[str]:
    """ML -> DL -> Foundation."""
    bottom = [m for m in BOTTOM_METHODS if m in index]
    rest   = [m for m in index if m not in BOTTOM_METHODS]

    groups: dict[str, list[str]] = {cat: [] for cat in CATEGORY_ORDER}
    ungrouped = []
    for m in rest:
        cat = METHOD_CATEGORY.get(m)
        if cat in groups:
            groups[cat].append(m)
        else:
            ungrouped.append(m)

    sorted_rest = []
    for cat in CATEGORY_ORDER:
        sorted_rest.extend(sorted(groups[cat]))
    sorted_rest.extend(sorted(ungrouped))

    return sorted_rest + bottom


def pivot_main_table(
    table: pd.DataFrame,
    method_category: str,
    dataset_metric_map: pd.Series,
    model: str,
) -> pd.DataFrame:
    table_filtered = table[table["method"].str.contains(method_category, regex=False)].copy()
    table_filtered["method"] = table_filtered["method"].apply(clean_method_name)

    pivot = table_filtered.pivot(index="method", columns="dataset", values="metric_error")
    pivot.columns.name = None
    pivot.index.name = None

    # drop columns that model does not have result on
    model_rows = pivot[pivot.index.str.startswith(model)]
    if not model_rows.empty:
        valid_cols = model_rows.columns[model_rows.notna().all(axis=0)]
        pivot = pivot[valid_cols]

    sorted_cols = sort_columns_by_metric(pivot.columns, dataset_metric_map)
    pivot = pivot[sorted_cols]

    # roc_auc: 1 - value 
    for col in pivot.columns:
        if dataset_metric_map.get(col) == "roc_auc":
            pivot[col] = 1 - pivot[col].astype(float)

    # sort by model category (ml, dl, foundation)
    sorted_idx = sort_index_by_category(pivot.index)
    pivot = pivot.loc[sorted_idx]

    # dataset name alist
    pivot.columns = [DATASET_ABBREV.get(c, c) for c in pivot.columns]

    # add category column
    pivot.insert(0, "category", [METHOD_CATEGORY.get(m, "") for m in pivot.index])

    # metric row 
    abbrev_metric_map = {
        DATASET_ABBREV.get(k, k): v
        for k, v in dataset_metric_map.items()
    }
    data_cols = [c for c in pivot.columns if c != "category"]
    metric_vals = {"category": ""} | {c: abbrev_metric_map.get(c, "") for c in data_cols}
    metric_df = pd.DataFrame([metric_vals], index=["metric"])
    pivot = pd.concat([metric_df, pivot])

    return pivot, abbrev_metric_map


def save_latex(
    pivot: pd.DataFrame,
    abbrev_metric_map: dict[str, str],
    path: Path,
    method_category: str,
) -> None:
    df = pivot.copy()

    metric_row = df.loc[["metric"]]
    numeric_df = df.drop(index="metric").copy()

    category_col = numeric_df["category"].copy()
    numeric_df   = numeric_df.drop(columns=["category"]).astype(float)
    metric_row   = metric_row.drop(columns=["category"])

    original_cols: list[str] = list(numeric_df.columns)
    escaped_cols = [escape_latex(str(c)) for c in original_cols]
    escaped_idx  = [escape_latex(str(i)) for i in numeric_df.index]

    first_bottom_escaped = None
    for m in BOTTOM_METHODS:
        escaped_m = escape_latex(m)
        if escaped_m in escaped_idx:
            first_bottom_escaped = escaped_m
            break

    numeric_df.columns = escaped_cols
    numeric_df.index   = escaped_idx
    category_col.index = escaped_idx
    metric_row.columns = escaped_cols
    metric_row.index   = ["metric"]

    def fmt_cell(val: float, orig_col: str, escaped_col: str) -> str:
        try:
            col_vals = numeric_df[escaped_col].dropna()
            metric_name = abbrev_metric_map.get(orig_col, "")
            direction = METRIC_DIRECTION.get(metric_name, "min")
            best_val = col_vals.max() if direction == "max" else col_vals.min()
            is_best = float(val) == best_val
            s = f"{float(val):.4f}"
            return f"\\textbf{{{s}}}" if is_best else s
        except (ValueError, TypeError):
            return str(val)

    formatted = numeric_df.copy().astype(object)
    for orig_col, esc_col in zip(original_cols, escaped_cols):
        for idx in numeric_df.index:
            v = numeric_df.at[idx, esc_col]
            formatted.at[idx, esc_col] = fmt_cell(v, orig_col, esc_col)

    formatted.insert(0, "category", category_col)

    # merge metric columns for visibility
    metric_groups: list[tuple[str, int]] = []  # (metric_name, count)
    for metric_name in METRIC_ORDER:
        cols_in_group = [c for c in original_cols if abbrev_metric_map.get(c) == metric_name]
        if cols_in_group:
            metric_groups.append((metric_name, len(cols_in_group)))

    metric_header_cells = ["", ""]  # method, category
    for metric_name, count in metric_groups:
        # print(metric_name)
        arrow = METRIC_ARROW.get(metric_name)
        metric_header_cells.append(
            f"\\multicolumn{{{count}}}{{c}}{{{escape_latex(metric_name)} (\\{arrow})}}"
        )
    metric_header_row = " & ".join(metric_header_cells) + " \\\\"

    # \cmidrule under metric
    cmidrule_parts = []
    col_offset = 3  # method(1) + category(2) + 1-indexed
    for metric_name, count in metric_groups:
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_offset}-{col_offset + count - 1}}}")
        col_offset += count
    cmidrule_row = " ".join(cmidrule_parts)

    # set dataset names
    dataset_header_cells = ["\\textbf{Method}", "\\textbf{Type}"] + [
        f"\\textbf{{{escape_latex(c)}}}" for c in original_cols
    ]
    dataset_header_row = " & ".join(dataset_header_cells) + " \\\\"

    final_df = pd.concat([formatted])
    col_fmt = "ll|" + "r" * len(escaped_cols)

    body_latex = final_df.to_latex(
        escape=False,
        column_format=col_fmt,
        na_rep="--",
        header=False,  
    ).strip()

    body_lines = body_latex.split("\n")
    start_idx = next(i for i, l in enumerate(body_lines) if l.strip().startswith("\\begin{tabular}"))
    end_idx   = next(i for i, l in enumerate(body_lines) if l.strip() == "\\end{tabular}")
    inner_lines = body_lines[start_idx + 1 : end_idx]  

    # \toprule, metric header, cmidrule, dataset header, \midrule
    new_inner = []
    toprule_inserted = False
    for line in inner_lines:
        new_inner.append(line)
        if line.strip() == "\\toprule" and not toprule_inserted:
            new_inner.append(metric_header_row)
            new_inner.append(cmidrule_row)
            new_inner.append(dataset_header_row)
            new_inner.append("\\midrule")
            toprule_inserted = True
        # for ours and important baselines, add \midrule
        if first_bottom_escaped and line.strip().startswith(first_bottom_escaped + " &"):
            new_inner.insert(len(new_inner) - 1, "\\midrule")

    tabular_str = "\n".join(
        [f"\\begin{{tabular}}{{{col_fmt}}}"] + new_inner + ["\\end{tabular}"]
    )

    tag = method_category.strip("()").replace(" + ", "_").replace(" ", "_")
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{%",
        tabular_str,
        "}",
        f"\\caption{{Results ({tag})}}",
        f"\\label{{tab:results_{tag}}}",
        "\\end{table}",
    ]
    path.write_text("\n".join(latex_lines), encoding="utf-8")
