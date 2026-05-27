from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def allocation_metrics(y_binary, prob, y_units, pred_units, threshold: float = 0.35) -> dict:
    y_binary = np.asarray(y_binary).astype(int)
    prob = np.asarray(prob).reshape(-1)
    y_units = np.asarray(y_units).astype(int)
    pred_units = np.asarray(pred_units).astype(int)
    pred_binary = (prob >= float(threshold)).astype(int)
    # Zero units where probability does not clear threshold.
    final_units = pred_units.copy()
    final_units[pred_binary == 0] = 0
    p, r, f1, _ = precision_recall_fscore_support(y_binary, pred_binary, average="binary", zero_division=0)
    fp = int(((y_binary == 0) & (pred_binary == 1)).sum())
    tn = int(((y_binary == 0) & (pred_binary == 0)).sum())
    fn = int(((y_binary == 1) & (pred_binary == 0)).sum())
    tp = int(((y_binary == 1) & (pred_binary == 1)).sum())
    pos = y_units > 0
    return {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "exact_unit_accuracy": float((final_units == y_units).mean()) if len(y_units) else 0.0,
        "positive_unit_accuracy": float((final_units[pos] == y_units[pos]).mean()) if pos.sum() else 0.0,
        "unit_mae": float(np.mean(np.abs(final_units - y_units))) if len(y_units) else 0.0,
        "positive_unit_mae": float(np.mean(np.abs(final_units[pos] - y_units[pos]))) if pos.sum() else 0.0,
        "predicted_positive_rows": int(pred_binary.sum()),
        "actual_positive_rows": int(y_binary.sum()),
    }


def threshold_sweep(y_binary, prob, y_units, pred_units) -> pd.DataFrame:
    rows = []
    for t in np.round(np.arange(0.01, 0.951, 0.01), 2):
        rows.append(allocation_metrics(y_binary, prob, y_units, pred_units, threshold=float(t)))
    return pd.DataFrame(rows)


def best_threshold_from_sweep(sweep: pd.DataFrame, mode: str = "balanced") -> float:
    if sweep.empty:
        return 0.35
    tmp = sweep.copy()
    if mode == "conservative":
        tmp["score"] = tmp["precision"] * 0.65 + tmp["f1"] * 0.35 - tmp["false_positive_rate"] * 0.5
    elif mode == "aggressive":
        tmp["score"] = tmp["recall"] * 0.60 + tmp["f1"] * 0.40
    else:
        tmp["score"] = tmp["f1"] * 0.70 + tmp["exact_unit_accuracy"] * 0.20 - tmp["unit_mae"] * 0.10
    return float(tmp.sort_values("score", ascending=False).iloc[0]["threshold"])


def confusion_table(y_binary, prob, threshold: float) -> pd.DataFrame:
    y_binary = np.asarray(y_binary).astype(int)
    pred = (np.asarray(prob).reshape(-1) >= threshold).astype(int)
    return pd.DataFrame(
        [[int(((y_binary == 0) & (pred == 0)).sum()), int(((y_binary == 0) & (pred == 1)).sum())],
         [int(((y_binary == 1) & (pred == 0)).sum()), int(((y_binary == 1) & (pred == 1)).sum())]],
        index=["Actual No Alloc", "Actual Alloc"],
        columns=["Predicted No Alloc", "Predicted Alloc"],
    )
