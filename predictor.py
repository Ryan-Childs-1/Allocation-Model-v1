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


# -----------------------------------------------------------------------------
# scikit-learn pickle compatibility helpers
# -----------------------------------------------------------------------------
def _walk_estimator_tree(obj: Any, seen: set[int] | None = None):
    """Yield sklearn-like objects contained inside common estimator containers.

    This intentionally walks both public sklearn containers and selected object
    attributes because artifact zips may be trained under one sklearn version and
    loaded under another. The compatibility issues normally surface only at
    transform/predict time, so repairs must happen immediately after joblib.load().
    """
    if obj is None:
        return
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    yield obj

    # Pipeline-like: .steps = [(name, estimator), ...]
    steps = getattr(obj, "steps", None)
    if steps:
        for _name, step in steps:
            yield from _walk_estimator_tree(step, seen)

    # ColumnTransformer-like fitted transformers.
    transformers = getattr(obj, "transformers_", None) or getattr(obj, "transformers", None)
    if transformers:
        for item in transformers:
            if not item or len(item) < 2:
                continue
            trans = item[1]
            if trans in (None, "drop", "passthrough"):
                continue
            yield from _walk_estimator_tree(trans, seen)

    # FeatureUnion-like transformer_list.
    transformer_list = getattr(obj, "transformer_list", None)
    if transformer_list:
        for _name, trans in transformer_list:
            yield from _walk_estimator_tree(trans, seen)

    # Final defensive pass: walk a few nested sklearn attrs that sometimes hold
    # cloned/fitted estimators but are not exposed through the containers above.
    for attr in ("estimator", "estimator_", "base_estimator", "base_estimator_", "calibrated_classifiers_"):
        try:
            val = getattr(obj, attr, None)
        except Exception:
            val = None
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            for child in val:
                yield from _walk_estimator_tree(child, seen)
        else:
            yield from _walk_estimator_tree(val, seen)


def _repair_sklearn_pickle_compat(bundle: dict) -> dict:
    """Repair known sklearn version-mismatch issues in uploaded model bundles.

    Jupyter artifacts are often serialized with a different sklearn release than
    Streamlit hosting uses. This function fills missing private attributes that
    newer sklearn expects. It does not retrain the model or intentionally change
    predictions; it only makes old pickles callable again.
    """
    if not isinstance(bundle, dict):
        return bundle

    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        SimpleImputer = None
    try:
        from sklearn.preprocessing import OneHotEncoder
    except Exception:
        OneHotEncoder = None

    repairs: list[str] = []
    root_objects = [
        bundle.get("preprocessor"),
        bundle.get("unit_model"),
        bundle.get("alloc_model"),
    ]

    for root in root_objects:
        if root is None:
            continue
        for est in _walk_estimator_tree(root):
            cls_name = est.__class__.__name__

            # sklearn newer compatibility for older ColumnTransformer pickles.
            if cls_name == "ColumnTransformer":
                if not hasattr(est, "force_int_remainder_cols"):
                    try:
                        est.force_int_remainder_cols = "deprecated"
                        repairs.append("ColumnTransformer.force_int_remainder_cols")
                    except Exception:
                        pass
                if not hasattr(est, "verbose_feature_names_out"):
                    try:
                        est.verbose_feature_names_out = True
                        repairs.append("ColumnTransformer.verbose_feature_names_out")
                    except Exception:
                        pass

            # sklearn newer compatibility for older SimpleImputer pickles.
            if (SimpleImputer is not None and isinstance(est, SimpleImputer)) or cls_name == "SimpleImputer":
                if not hasattr(est, "_fill_dtype"):
                    fill_dtype = getattr(est, "_fit_dtype", None)
                    stats = getattr(est, "statistics_", None)
                    if fill_dtype is None and stats is not None:
                        fill_dtype = getattr(stats, "dtype", None)
                    if fill_dtype is None:
                        fill_dtype = object if getattr(est, "strategy", None) == "constant" else float
                    try:
                        est._fill_dtype = fill_dtype
                        repairs.append("SimpleImputer._fill_dtype")
                    except Exception:
                        pass
                if not hasattr(est, "keep_empty_features"):
                    try:
                        est.keep_empty_features = False
                        repairs.append("SimpleImputer.keep_empty_features")
                    except Exception:
                        pass
                if not hasattr(est, "indicator_"):
                    try:
                        est.indicator_ = None
                        repairs.append("SimpleImputer.indicator_")
                    except Exception:
                        pass
                if not hasattr(est, "add_indicator"):
                    try:
                        est.add_indicator = False
                        repairs.append("SimpleImputer.add_indicator")
                    except Exception:
                        pass

            # sklearn 1.2+ renamed OneHotEncoder(sparse -> sparse_output).
            if (OneHotEncoder is not None and isinstance(est, OneHotEncoder)) or cls_name == "OneHotEncoder":
                if not hasattr(est, "sparse_output"):
                    try:
                        est.sparse_output = getattr(est, "sparse", True)
                        repairs.append("OneHotEncoder.sparse_output")
                    except Exception:
                        pass
                if not hasattr(est, "_infrequent_enabled"):
                    try:
                        est._infrequent_enabled = False
                        repairs.append("OneHotEncoder._infrequent_enabled")
                    except Exception:
                        pass
                if not hasattr(est, "feature_name_combiner"):
                    try:
                        est.feature_name_combiner = "concat"
                        repairs.append("OneHotEncoder.feature_name_combiner")
                    except Exception:
                        pass

    existing = bundle.get("__compat_repairs", [])
    bundle["__compat_repairs"] = list(dict.fromkeys(list(existing) + repairs))
    return bundle

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
                bundle = _repair_sklearn_pickle_compat(bundle)
            elif suffix in {".joblib", ".pkl"}:
                artifact_model_name = upload_name
                bundle = joblib.load(upload_path)
                bundle = _repair_sklearn_pickle_compat(bundle)
            else:
                raise ValueError("Unsupported model upload. Please upload .zip, .joblib, or .pkl.")
    else:
        bundle = joblib.load(Path(default_path))
        bundle = _repair_sklearn_pickle_compat(bundle)
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


def _predict_chunk(X_chunk: pd.DataFrame, bundle: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Transform and predict one chunk.

    Newer uploaded artifacts can use large sklearn ColumnTransformer outputs.
    Processing the entire workbook at once can exceed hosted Streamlit memory,
    especially after sparse one-hot output is converted to dense float32 for MLP
    models. Chunking keeps memory bounded and fixes crashes that appear only
    with the stronger Jupyter-trained model.
    """
    Xt = bundle["preprocessor"].transform(X_chunk.replace([np.inf, -np.inf], np.nan))
    Xt = to_dense_float32(Xt)

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


def predict_arrays(df: pd.DataFrame, bundle: dict) -> Tuple[np.ndarray, np.ndarray]:
    X = build_feature_frame(df)
    X = align_to_training_columns(X, list(bundle["feature_columns"]))

    # Hosted Streamlit memory guard. The older included model was small enough to
    # predict in one pass, but stronger Jupyter artifacts can produce much larger
    # dense arrays. Keep chunks modest; model prediction time remains acceptable.
    chunk_size = int(bundle.get("prediction_chunk_size", 2500) or 2500)
    chunk_size = max(250, min(chunk_size, 10000))

    all_units = []
    all_prob = []
    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        u, p = _predict_chunk(X.iloc[start:end], bundle)
        all_units.append(u)
        all_prob.append(p)

    if all_units:
        units = np.concatenate(all_units).astype(int)
        prob = np.concatenate(all_prob).astype("float32")
    else:
        units = np.zeros(0, dtype=int)
        prob = np.zeros(0, dtype="float32")
    return units, prob


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
    review_mask = audit["flag"].astype(str).str.upper().str.contains("REVIEW", na=False) if "flag" in audit.columns else pd.Series(False, index=audit.index)
    summary = {
        "rows": int(len(out)),
        "allocated_rows": int((final_numeric > 0).sum()),
        "total_final_alloc": int(final_numeric.sum()),
        "mean_probability": float(prob_s.mean() if len(prob_s) else 0),
        "z_no_alloc_overrides": int(pd.to_numeric(audit.get("z_no_alloc_override", pd.Series(0, index=audit.index)), errors="coerce").fillna(0).sum()),
        "review_rows_allocated": int((review_mask & (final_numeric > 0)).sum()),
        "review_total_final_alloc": int(final_numeric.where(review_mask, 0).sum()),
        "review_pass_1_added": int(pd.to_numeric(audit.get("review_pass_1_added", pd.Series(0, index=audit.index)), errors="coerce").fillna(0).sum()),
        "review_pass_2_added": int(pd.to_numeric(audit.get("review_pass_2_added", pd.Series(0, index=audit.index)), errors="coerce").fillna(0).sum()),
        "review_pass_3_added": int(pd.to_numeric(audit.get("review_pass_3_added", pd.Series(0, index=audit.index)), errors="coerce").fillna(0).sum()),
    }
    return out, audit, summary
