"""Benchmark all forecasting models across all datasets.

Runs every model in the registry over each univariate series, computing
error metrics (MSE, RMSE, MAE, MAPE) on the held-out test split in the
series' original units, and prints one table per dataset.

Run from the repo root so ``conf``/``models``/``utils`` resolve as packages:

    uv run python benchmark.py                     # all models, all datasets
    uv run python benchmark.py --datasets melbmin lynx
    uv run python benchmark.py --models KNN SVM RF --csv results/benchmark.csv
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from conf.processing import DataProcessing
from utils.plot_dashes import get_metrics_error

from models.adaline import Adaline
from models.ann_models import MLP, GridSCN, GridELM
from models.k_near import KNEAR
from models.kernel_method import SVM
from models.tree_based_ensemble import RF, GBoosting

# Model registry: name -> zero-arg constructor (uses each model's default grid).
# Ordered roughly fast -> slow.
MODELS = {
    "Adaline": Adaline,
    "KNN": KNEAR,
    "SVM": SVM,
    "ELM": GridELM,
    "SCN": GridSCN,
    "MLP": MLP,
    "RF": RF,
    "GBoosting": GBoosting,
}

DATA_DIR = Path("data")
DATASETS = ["airlines", "lynx", "coloradoRiver", "melbmin"]


def _raw_test_pred(result: dict) -> np.ndarray:
    """Return the model's test prediction in the series' original units.

    Models that predict in normalized space expose ``pred_test_denom``;
    models that predict in raw units expose only ``pred_test``.
    """
    denom = result.get("pred_test_denom")
    return np.asarray(denom if denom is not None else result["pred_test"])


def load_series(name: str) -> pd.DataFrame:
    """Load a univariate series file (single column with a ``y`` header)."""
    path = DATA_DIR / f"{name}.txt"
    return pd.read_csv(path)


def run_model(model_name: str, data: dict) -> dict:
    """Train one model on a prepared ``data`` dict and return its test metrics."""
    model = MODELS[model_name]()
    start = time.perf_counter()
    result = model.train(data)
    elapsed = time.perf_counter() - start

    pred_test = _raw_test_pred(result)
    metrics = get_metrics_error(data["target_test"], pred_test)
    metrics["seconds"] = elapsed
    return metrics


def benchmark(
    datasets: list[str], model_names: list[str], n_lags: int, seed: int
) -> pd.DataFrame:
    """Run every requested model over every requested dataset.

    Returns a long-form DataFrame (one row per dataset x model).
    """
    rows = []
    for ds in datasets:
        series = load_series(ds)
        data = DataProcessing(series, n_lags=n_lags).prepare_data()

        for name in model_names:
            # Reseed before each model so stochastic models are reproducible
            # and independent of run order.
            random.seed(seed)
            np.random.seed(seed)
            try:
                metrics = run_model(name, data)
                metrics.update({"dataset": ds, "model": name})
            except Exception as exc:  # keep the table intact if one model fails
                metrics = {
                    "dataset": ds,
                    "model": name,
                    "MSE": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "MAPE": np.nan,
                    "seconds": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(metrics)

    return pd.DataFrame(rows)


def print_tables(df: pd.DataFrame) -> None:
    """Print one metrics table per dataset, best (lowest RMSE) first."""
    cols = ["model", "MSE", "RMSE", "MAE", "MAPE", "seconds"]
    for ds, group in df.groupby("dataset", sort=False):
        table = group.sort_values("RMSE").reset_index(drop=True)
        print(f"\n=== {ds} ===")
        print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        failed = group[group.get("error").notna()] if "error" in group else group.iloc[0:0]
        for _, r in failed.iterrows():
            print(f"  ! {r['model']} failed: {r['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--n-lags", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, default=None, help="optional path to save results")
    args = parser.parse_args()

    df = benchmark(args.datasets, args.models, args.n_lags, args.seed)
    print_tables(df)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nSaved results to {args.csv}")


if __name__ == "__main__":
    main()
