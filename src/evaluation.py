# src/evaluation.py

from typing import Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .config import OUTPUTS_DIR

def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute R2, MAE, RMSE, MAPE (as %) for regression.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100.0

    return {
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }

def save_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    filename: str = "metrics_summary.csv"
) -> Path:
    """
    Save a multi-model metrics table to outputs/metrics.
    """
    df = pd.DataFrame(metrics_dict).T
    path = OUTPUTS_DIR / "metrics" / filename
    df.to_csv(path, index=True)
    return path

def parity_plot(
    y_true,
    y_pred,
    title: str = "",
    fname: str = "parity_plot.png",
    train_or_test: str = ""
) -> Path:
    """
    Save a parity plot (y_pred vs y_true) with ±20% bounds.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=20, alpha=0.7)

    # y = x and ±20% lines
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(x_line, x_line, "k-", label="y = x")
    ax.plot(x_line, 0.8 * x_line, "b--", label="y = 0.8x")
    ax.plot(x_line, 1.2 * x_line, "b--", label="y = 1.2x")

    ax.set_xlabel("Experimental Nu (kN)")
    ax.set_ylabel("Predicted Nu (kN)")
    full_title = title
    if train_or_test:
        full_title += f" ({train_or_test})"
    ax.set_title(full_title)
    ax.legend()

    fig_path = OUTPUTS_DIR / "figures" / fname
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return fig_path

def save_predictions(
    specimen_ids,
    y_true,
    y_pred,
    model_name: str,
    split: str
) -> Path:
    """
    Save prediction CSV with specimen id, true, pred.
    """
    df = pd.DataFrame({
        "specimen_id": specimen_ids,
        "Nu_exp": np.asarray(y_true).ravel(),
        f"Nu_pred_{model_name}_{split}": np.asarray(y_pred).ravel()
    })
    path = OUTPUTS_DIR / "predictions" / f"pred_{model_name}_{split}.csv"
    df.to_csv(path, index=False)
    return path
