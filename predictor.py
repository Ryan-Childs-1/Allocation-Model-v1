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



def _joblib_part_paths(path: str | Path) -> list[Path]:
    """Return ordered split-part paths for a model stored as file.joblib.part01, part02, ...

    This keeps each repository file below hosting limits while allowing the app
    to reconstruct the built-in model in memory at runtime.
    """
    p = Path(path)
    parent = p.parent if str(p.parent) != "." else Path(".")
    parts = sorted(parent.glob(p.name + ".part*"))
    return [x for x in parts if x.is_file()]


def _load_joblib_with_split_support(path: str | Path):
    """Load a joblib bundle from either a normal file or split .partXX files."""
    p = Path(path)
    if p.exists():
        return joblib.load(p)
    parts = _joblib_part_paths(p)
    if parts:
        import io
        data = b"".join(part.read_bytes() for part in parts)
        return joblib.load(io.BytesIO(data))
    raise FileNotFoundError(
        f"Model file was not found: {p}. Also looked for split parts like {p.name}.part01."
    )


def _model_storage_note(path: str | Path) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    parts = _joblib_part_paths(p)
    if parts:
        total_mb = sum(x.stat().st_size for x in parts) / (1024 * 1024)
        return f"{p.name} split into {len(parts)} parts ({total_mb:.1f} MB total)"
    return str(p)

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
        if "transfer_model" in name or "transfer" in name:
            s += 1100
        if "app_compatible" in name:
            s += 1000
        if "camp" in name:
            s += 250
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


def load_model_bundle(model_file: Any | None = None, default_path: str | Path = DEFAULT_MODEL_PATH, default_metadata_path: str | Path = DEFAULT_METADATA_PATH, model_label: str = "Base NN Model") -> dict:
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
    artifact_model_name = str(model_label or "Base NN Model")

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
                bundle = _load_joblib_with_split_support(model_path)
                bundle = _repair_sklearn_pickle_compat(bundle)
            elif suffix in {".joblib", ".pkl"}:
                artifact_model_name = upload_name
                bundle = _load_joblib_with_split_support(upload_path)
                bundle = _repair_sklearn_pickle_compat(bundle)
            else:
                raise ValueError("Unsupported model upload. Please upload .zip, .joblib, or .pkl.")
    else:
        bundle = _load_joblib_with_split_support(Path(default_path))
        bundle = _repair_sklearn_pickle_compat(bundle)
        artifact_metadata = read_metadata(default_metadata_path)

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
    if model_file is None:
        bundle["__model_storage_note"] = _model_storage_note(default_path)
    else:
        bundle["__model_storage_note"] = artifact_model_name
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



# -----------------------------------------------------------------------------
# Model inspection + prediction explanation helpers
# -----------------------------------------------------------------------------
def _simplify_feature_name(name: str) -> str:
    """Map transformed sklearn feature names back to readable base feature names."""
    text = str(name)
    # Common ColumnTransformer names: num__num__d60, cat__cat__item_123
    for prefix in ("num__", "cat__"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.startswith("num__"):
        return text[len("num__"):]
    if text.startswith("cat__"):
        # Keep only the field name before one-hot category text where possible.
        rest = text[len("cat__"):]
        for field in ("site", "item", "upc", "flag"):
            if rest == field or rest.startswith(field + "_"):
                return field
        return rest.split("_")[0]
    return text.split("_")[0] if "_" in text else text


def _feature_family(name: str) -> str:
    n = _simplify_feature_name(name).lower()
    if "alloc_rec" in n:
        return "Alloc. Rec."
    if "left_dc" in n or "dc_avail" in n or "dc_to" in n or "to_dc" in n:
        return "DC / Left DC"
    if "demand" in n or "d60" in n or "d30" in n or "l30" in n or "ttm" in n or "lw" in n or "velocity" in n:
        return "Demand / velocity"
    if "need" in n or "gap" in n or "pressure" in n or "scarcity" in n:
        return "Need / scarcity"
    if "supply" in n or "qoh" in n:
        return "Supply"
    if "review" in n or "flag" in n or "no_alloc" in n:
        return "Flag / review"
    if "rank" in n or "share" in n or "item_total" in n or "site_total" in n or "dept" in n or "class" in n:
        return "Group / rank context"
    if "retail" in n or "cost" in n or "margin" in n or "gm" in n:
        return "Retail / margin"
    if n in {"item", "site", "upc"}:
        return "Categorical identity"
    return "Other"


def model_feature_importance(bundle: dict, top_n: int = 40) -> pd.DataFrame:
    """Approximate feature usage from first-layer MLP weights.

    This is not causal feature attribution. It is a practical, model-inspection
    view showing which transformed inputs have the largest first-layer weight
    magnitudes in the unit and allocation heads.
    """
    try:
        pre = bundle.get("preprocessor")
        names = list(pre.get_feature_names_out()) if hasattr(pre, "get_feature_names_out") else []
    except Exception:
        names = []

    rows = []
    for label, model_key in [("unit_model", "unit_model"), ("alloc_model", "alloc_model")]:
        model = bundle.get(model_key)
        coefs = getattr(model, "coefs_", None)
        if not coefs:
            continue
        w = np.asarray(coefs[0])
        imp = np.mean(np.abs(w), axis=1)
        if not names or len(names) != len(imp):
            names = [f"transformed_feature_{i}" for i in range(len(imp))]
        for fname, val in zip(names, imp):
            base = _simplify_feature_name(fname)
            rows.append({
                "model_head": label,
                "transformed_feature": str(fname),
                "base_feature": base,
                "feature_family": _feature_family(base),
                "importance": float(val),
            })
    if not rows:
        return pd.DataFrame(columns=["feature_family", "base_feature", "importance"])
    df = pd.DataFrame(rows)
    grouped = (df.groupby(["feature_family", "base_feature"], as_index=False)["importance"]
                 .mean()
                 .sort_values("importance", ascending=False))
    return grouped.head(int(top_n)).reset_index(drop=True)


def prediction_feature_relationships(df: pd.DataFrame, audit_df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Show which engineered numeric features move most with predicted outputs.

    This is computed on the uploaded file after prediction, using correlation and
    coverage. It helps users see which columns/signals were most related to the
    generated allocation decisions for that run.
    """
    try:
        X = build_feature_frame(df)
    except Exception:
        return pd.DataFrame()
    if audit_df is None or audit_df.empty:
        return pd.DataFrame()
    target_prob = pd.to_numeric(audit_df.get("probability", pd.Series(0, index=audit_df.index)), errors="coerce").fillna(0)
    target_alloc = pd.to_numeric(audit_df.get("final_alloc", pd.Series(0, index=audit_df.index)), errors="coerce").fillna(0)
    rows = []
    for c in [c for c in X.columns if c.startswith("num__")]:
        s = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        coverage = float(s.notna().mean()) if len(s) else 0.0
        if coverage <= 0 or s.nunique(dropna=True) <= 1:
            continue
        sf = s.fillna(s.median())
        try:
            corr_prob = float(np.corrcoef(sf.values, target_prob.values)[0, 1])
        except Exception:
            corr_prob = 0.0
        try:
            corr_alloc = float(np.corrcoef(sf.values, target_alloc.values)[0, 1])
        except Exception:
            corr_alloc = 0.0
        if not np.isfinite(corr_prob): corr_prob = 0.0
        if not np.isfinite(corr_alloc): corr_alloc = 0.0
        base = c.replace("num__", "")
        rows.append({
            "feature": base,
            "feature_family": _feature_family(base),
            "coverage": coverage,
            "corr_with_probability": corr_prob,
            "corr_with_final_alloc": corr_alloc,
            "relationship_strength": max(abs(corr_prob), abs(corr_alloc)),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("relationship_strength", ascending=False).head(int(top_n)).reset_index(drop=True)


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
