from __future__ import annotations

import json
import tempfile
import zipfile
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


def _load_json_safely(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pick_model_from_artifact_dir(root: Path) -> Path:
    """Find the app-compatible prediction bundle inside an exported training artifact folder.

    The Jupyter trainer can export many files in one zip: PyTorch checkpoints, logs,
    threshold sweeps, metadata, datasets, and one Streamlit-compatible joblib/pkl model.
    This function intentionally prefers the app-compatible bundle and ignores raw
    checkpoint files such as .pt/.keras because the prediction-only app expects the
    compressed sklearn-style bundle with preprocessor + feature columns + models.
    """
    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".joblib", ".pkl"}]
    if not candidates:
        raise ValueError("No .joblib or .pkl app-compatible model bundle was found inside the uploaded artifact zip.")

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        # Avoid accidentally loading datasets or logs that are pickled.
        penalty = 0
        if "dataset" in name or "training_data" in name or "progress" in name:
            penalty -= 100
        s = 0
        if "app_compatible" in name:
            s += 1000
        if "prediction" in name:
            s += 600
        if "base_sklearn_mlp" in name:
            s += 400
        if "model" in name:
            s += 200
        if "bundle" in name:
            s += 100
        if path.suffix.lower() == ".joblib":
            s += 20
        return (s + penalty, -len(path.parts), str(path))

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def _find_metadata_in_artifact_dir(root: Path) -> dict:
    metadata: dict = {}
    json_files = [p for p in root.rglob("*.json") if p.is_file()]
    priority = sorted(
        json_files,
        key=lambda p: (
            1000 if "metadata" in p.name.lower() else 0,
            500 if "app_compatible" in p.name.lower() else 0,
            200 if "result" in p.name.lower() else 0,
            -len(p.parts),
            str(p),
        ),
        reverse=True,
    )
    for path in priority:
        loaded = _load_json_safely(path)
        if loaded:
            metadata.update(loaded)
    return metadata


def load_model_bundle(model_file: Any | None = None, default_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Load a prediction bundle from an upload or the included base model.

    Accepted upload types:
      - .joblib / .pkl: direct Allocation AI prediction bundle
      - .zip: artifact export containing an app-compatible .joblib/.pkl model,
        metadata, threshold sweep, checkpoints, and logs

    Expected model bundle keys:
      - preprocessor
      - feature_columns
      - unit_model
      - alloc_model
    """
    artifact_metadata: dict = {}
    artifact_model_name = "included base model"

    if model_file is not None:
        upload_name = getattr(model_file, "name", "uploaded_model")
        suffix = Path(upload_name).suffix.lower()
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            upload_path = tmp_root / upload_name
            upload_path.write_bytes(model_file.getbuffer())

            if suffix == ".zip":
                extract_dir = tmp_root / "artifact_zip"
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(upload_path, "r") as z:
                    z.extractall(extract_dir)
                model_path = _pick_model_from_artifact_dir(extract_dir)
                artifact_metadata = _find_metadata_in_artifact_dir(extract_dir)
                artifact_model_name = model_path.name
                bundle = joblib.load(model_path)
            elif suffix in {".joblib", ".pkl"}:
                artifact_model_name = upload_name
                bundle = joblib.load(upload_path)
            else:
                raise ValueError("Unsupported model upload. Please upload .zip, .joblib, or .pkl.")
    else:
        bundle = joblib.load(Path(default_path))
        artifact_metadata = read_metadata()

    required = {"preprocessor", "feature_columns", "unit_model", "alloc_model"}
    missing = required.difference(bundle.keys()) if isinstance(bundle, dict) else required
    if missing:
        raise ValueError(
            "Uploaded model is not an Allocation AI prediction bundle. "
            f"Missing keys: {sorted(missing)}. If you uploaded a training artifact zip, "
            "make sure it includes allocation_ai_app_compatible_model.joblib or another app-compatible model bundle."
        )

    # Attach non-model info for the UI and threshold defaults without affecting prediction.
    bundle["__artifact_metadata"] = artifact_metadata
    bundle["__artifact_model_name"] = artifact_model_name
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
