"""Regression metrics for yield prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return R², MAE, RMSE, MAPE, WAPE, and accuracy.

    accuracy_pct = 100 * (1 - WAPE), floored at 0, where WAPE (weighted absolute
    percentage error) = sum(|actual - predicted|) / sum(|actual|). WAPE is used
    instead of plain MAPE because MAPE is dominated by near-zero yields, which
    can make a good model look inaccurate. The synopsis target (85%) is compared
    against accuracy_pct. MAPE is also reported for reference.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(y_true_arr - y_pred_arr)
    total_actual = float(np.sum(np.abs(y_true_arr)))
    wape = float(np.sum(abs_error) / total_actual) if total_actual > 0 else 1.0
    return {
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mape_pct": 100.0 * float(mean_absolute_percentage_error(y_true_arr, y_pred_arr)),
        "wape_pct": 100.0 * wape,
        "accuracy_pct": 100.0 * max(0.0, 1.0 - wape),
    }
