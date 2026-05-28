from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd

from allocation_simulator import AllocationConfig, apply_allocation_simulation
from data_io import dataframe_to_csv_bytes
from features import build_feature_frame
from neural_model import align_to_training_columns, to_dense_float32
from schema import build_column_map

DEFAULT_MODEL_PATH = Path("allocation_ai_base_sklearn_mlp.joblib")
DEFAULT_METADATA_PATH = Path("allocation_ai_metadata.json")


def read_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_model_bundle(model_file: Any | None = None, default_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Load a prediction bundle from an uploaded file or the included base model.

    Expected bundle keys:
      - preprocessor
      - feature_columns
      - unit_model
      - alloc_model
    """
    if model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
            tmp.write(model_file.getbuffer())
            tmp.flush()
            p = Path(tmp.name)
        bundle = joblib.load(p)
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        bundle = joblib.load(Path(default_path))

    required = {"preprocessor", "feature_columns", "unit_model", "alloc_model"}
    missing = required.difference(bundle.keys()) if isinstance(bundle, dict) else required
    if missing:
        raise ValueError(f"Uploaded model is not an Allocation AI prediction bundle. Missing keys: {sorted(missing)}")
    return bundle


def predict_arrays(df: pd.DataFrame, bundle: dict) -> Tuple[np.ndarray, np.ndarray]:
    X = build_feature_frame(df)
    X = align_to_training_columns(X, list(bundle["feature_columns"]))
    Xt = to_dense_float32(bundle["preprocessor"].transform(X.replace([np.inf, -np.inf], np.nan)))

    unit_model = bundle["unit_model"]
    alloc_model = bundle["alloc_model"]

    units = np.asarray(unit_model.predict(Xt), dtype=int)
    if hasattr(alloc_model, "predict_proba"):
        classes = list(alloc_model.classes_)
        proba = alloc_model.predict_proba(Xt)
        prob = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(Xt), dtype="float32")
    else:
        prob = np.asarray(alloc_model.predict(Xt), dtype="float32")
    return units, np.asarray(prob, dtype="float32")


def predict_to_outputs(df: pd.DataFrame, bundle: dict, cfg: AllocationConfig) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    units, prob = predict_arrays(df, bundle)
    final_alloc, audit = apply_allocation_simulation(df, units, prob, cfg)

    out = df.copy()
    cmap = build_column_map(out)
    final_col = cmap.get("final_alloc") or "Final Alloc."
    if final_col not in out.columns:
        out[final_col] = ""
    out[final_col] = final_alloc.values
    if "__row_order" in out.columns:
        out = out.sort_values("__row_order")

    final_numeric = pd.to_numeric(audit["final_alloc"], errors="coerce").fillna(0)
    prob_s = pd.to_numeric(audit["probability"], errors="coerce").fillna(0)
    summary = {
        "rows": int(len(out)),
        "allocated_rows": int((final_numeric > 0).sum()),
        "total_final_alloc": int(final_numeric.sum()),
        "mean_probability": float(prob_s.mean() if len(prob_s) else 0),
        "z_no_alloc_overrides": int(pd.to_numeric(audit.get("z_no_alloc_override", pd.Series(0, index=audit.index)), errors="coerce").fillna(0).sum()),
        "review_rows_allocated": int(((audit["flag"].astype(str).str.upper().str.contains("REVIEW", na=False)) & (final_numeric > 0)).sum()),
    }
    return out, audit, summary
