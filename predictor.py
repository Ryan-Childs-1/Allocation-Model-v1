from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from allocation_simulator import AllocationConfig, apply_allocation_simulation
from features import build_feature_frame
from neural_model import align_to_training_columns, load_artifacts, to_dense_float32
from schema import build_column_map


def _metadata(model_dir: Path) -> dict:
    p = model_dir / "allocation_ai_metadata.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _predict_with_base_sklearn(df: pd.DataFrame, model_dir: Path):
    base_path = model_dir / "allocation_ai_base_sklearn_mlp.joblib"
    if not base_path.exists():
        raise FileNotFoundError(
            "No Keras model or base neural model was found. Train the model first or include allocation_ai_base_sklearn_mlp.joblib."
        )
    base = joblib.load(base_path)
    pre = base["preprocessor"]
    feature_columns = base["feature_columns"]
    X = build_feature_frame(df)
    X = align_to_training_columns(X, feature_columns)
    Xt = to_dense_float32(pre.transform(X.replace([np.inf, -np.inf], np.nan)))
    unit_model = base["unit_model"]
    alloc_model = base["alloc_model"]
    units = unit_model.predict(Xt).astype(int)
    if hasattr(alloc_model, "predict_proba"):
        classes = list(alloc_model.classes_)
        proba = alloc_model.predict_proba(Xt)
        prob = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(Xt), dtype="float32")
    else:
        prob = alloc_model.predict(Xt).astype(float)
    return units, np.asarray(prob, dtype="float32"), {"backend": "base_sklearn_mlp"}


def predict_arrays(df: pd.DataFrame, model_dir: str | Path = "."):
    model_dir = Path(model_dir)
    keras_files_ready = (model_dir / "allocation_ai_model.keras").exists() and (model_dir / "allocation_ai_preprocessor.joblib").exists() and (model_dir / "allocation_ai_feature_columns.joblib").exists()
    if not keras_files_ready:
        return _predict_with_base_sklearn(df, model_dir)
    model, pre, feature_columns = load_artifacts(model_dir)
    X = build_feature_frame(df)
    X = align_to_training_columns(X, feature_columns)
    Xt = to_dense_float32(pre.transform(X.replace([np.inf, -np.inf], np.nan)))
    preds = model.predict(Xt, batch_size=4096, verbose=0)
    units = np.argmax(np.asarray(preds["units"]), axis=1).astype(int)
    prob = np.asarray(preds["alloc_prob"]).reshape(-1)
    return units, prob, preds


def predict_to_csv_dataframe(df: pd.DataFrame, model_dir: str | Path, cfg: AllocationConfig, use_saved_threshold: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_dir = Path(model_dir)
    meta = _metadata(model_dir)
    if use_saved_threshold and meta.get("recommended_threshold"):
        cfg = AllocationConfig(
            min_probability=float(meta["recommended_threshold"]),
            demand_cap_extra_flm=cfg.demand_cap_extra_flm,
            allow_review_rows=cfg.allow_review_rows,
            alloc_rec_influence=cfg.alloc_rec_influence,
            prefer_left_dc=cfg.prefer_left_dc,
            allow_no_alloc_rows=cfg.allow_no_alloc_rows,
            no_alloc_min_probability=cfg.no_alloc_min_probability,
            no_alloc_min_need_flm_units=cfg.no_alloc_min_need_flm_units,
        )
    units, prob, _ = predict_arrays(df, model_dir=model_dir)
    final_alloc, audit = apply_allocation_simulation(df, units, prob, cfg)
    out = df.copy()
    cmap = build_column_map(out)
    fa_col = cmap.get("final_alloc") or "Final Alloc."
    if fa_col not in out.columns:
        out[fa_col] = ""
    out[fa_col] = final_alloc.values
    out = out.sort_values("__row_order") if "__row_order" in out.columns else out
    return out, audit
