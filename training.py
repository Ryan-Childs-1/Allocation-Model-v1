from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from features import build_feature_frame, build_targets
from metrics import allocation_metrics, best_threshold_from_sweep, threshold_sweep
from neural_model import (
    align_to_training_columns,
    architecture_for_mode,
    build_allocation_model,
    import_keras,
    make_preprocessor,
    save_artifacts,
    to_dense_float32,
)


def _split_indices(ds: pd.DataFrame, validation_method: str, validation_split: float):
    n = len(ds)
    idx = np.arange(n)
    y = ds["__target_alloc_binary"].astype(int).values if "__target_alloc_binary" in ds else None
    if validation_method == "holdout_by_file" and "__source_file" in ds.columns and ds["__source_file"].nunique() > 1:
        groups = ds["__source_file"].astype(str).values
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=42)
        return next(splitter.split(idx, y, groups))
    if validation_method == "holdout_by_item" and "cat__item" in ds.columns and ds["cat__item"].nunique() > 1:
        groups = ds["cat__item"].astype(str).values
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=42)
        return next(splitter.split(idx, y, groups))
    stratify = y if y is not None and len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    return train_test_split(idx, test_size=validation_split, random_state=42, stratify=stratify)


def prepare_training_matrices(
    dataset: pd.DataFrame,
    max_units: int = 80,
    existing_feature_columns: Optional[List[str]] = None,
    existing_preprocessor=None,
):
    target_cols = [c for c in dataset.columns if c.startswith("__target_")]
    meta_cols = ["__source_file", "__source_file_index"]
    X = dataset.drop(columns=target_cols + meta_cols, errors="ignore").copy()
    y_units = dataset["__target_units"].clip(0, int(max_units)).astype(int).values
    y_binary = dataset["__target_alloc_binary"].astype(int).values
    y_overstock = dataset.get("__target_overstock_risk", pd.Series(0, index=dataset.index)).astype(int).values
    y_review = dataset.get("__target_review_sensitive", pd.Series(0, index=dataset.index)).astype(int).values

    if existing_feature_columns is not None:
        X = align_to_training_columns(X, existing_feature_columns)
        feature_columns = existing_feature_columns
    else:
        feature_columns = [c for c in X.columns if c.startswith("num__") or c.startswith("cat__") or c in {"__excel_row", "__row_order"}]
        X = X[feature_columns]

    if existing_preprocessor is None:
        pre, _ = make_preprocessor(X)
        Xt = to_dense_float32(pre.fit_transform(X.replace([np.inf, -np.inf], np.nan)))
    else:
        pre = existing_preprocessor
        Xt = to_dense_float32(pre.transform(X.replace([np.inf, -np.inf], np.nan)))

    # Weight positive allocations more; weight larger allocation units slightly more; avoid overpowering extreme rows.
    class_w = compute_sample_weight("balanced", y_binary).astype("float32")
    unit_w = 1.0 + np.minimum(y_units, 10).astype("float32") * 0.08
    sw = (class_w * unit_w).astype("float32")
    return X, Xt, y_units, y_binary, y_overstock, y_review, sw, pre, feature_columns


class TargetScoreCallback:
    def __init__(
        self,
        total_epochs: int,
        X_val,
        y_units_val,
        y_binary_val,
        target_f1: float = 0.0,
        target_precision: float = 0.0,
        target_recall: float = 0.0,
        max_minutes: float = 0.0,
        progress=None,
        chart=None,
        status=None,
        metric_box=None,
        log_table=None,
    ):
        self.total_epochs = int(total_epochs)
        self.X_val = X_val
        self.y_units_val = y_units_val
        self.y_binary_val = y_binary_val
        self.target_f1 = float(target_f1 or 0.0)
        self.target_precision = float(target_precision or 0.0)
        self.target_recall = float(target_recall or 0.0)
        self.max_seconds = float(max_minutes or 0.0) * 60.0
        self.progress = progress
        self.chart = chart
        self.status = status
        self.metric_box = metric_box
        self.log_table = log_table
        self.start = time.time()
        self.history: List[Dict] = []

    def make(self):
        keras, _ = import_keras()
        outer = self

        class CB(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                preds = self.model.predict(outer.X_val, verbose=0, batch_size=4096)
                prob = np.asarray(preds["alloc_prob"]).reshape(-1)
                pred_units = np.argmax(np.asarray(preds["units"]), axis=1)
                # Use 0.35 for live reporting; final threshold is swept after training.
                m = allocation_metrics(outer.y_binary_val, prob, outer.y_units_val, pred_units, threshold=0.35)
                row = {"epoch": epoch + 1, **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float, np.number))}, **m}
                row["elapsed_minutes"] = round((time.time() - outer.start) / 60.0, 3)
                outer.history.append(row)
                if outer.progress is not None:
                    outer.progress.progress(min((epoch + 1) / max(outer.total_epochs, 1), 1.0), text=f"Epoch {epoch+1}/{outer.total_epochs} · F1 {m['f1']:.3f} · Precision {m['precision']:.3f} · Recall {m['recall']:.3f}")
                if outer.status is not None:
                    outer.status.write(f"Epoch {epoch+1}: loss={logs.get('loss', 0):.4f}, val_loss={logs.get('val_loss', 0):.4f}, F1={m['f1']:.3f}, unit MAE={m['unit_mae']:.3f}")
                if outer.chart is not None:
                    chart_df = pd.DataFrame(outer.history)[["epoch", "loss", "val_loss", "f1", "precision", "recall", "unit_mae"]].set_index("epoch")
                    outer.chart.line_chart(chart_df)
                if outer.metric_box is not None:
                    c1, c2, c3, c4 = outer.metric_box.columns(4)
                    c1.metric("Live F1", f"{m['f1']:.3f}")
                    c2.metric("Precision", f"{m['precision']:.3f}")
                    c3.metric("Recall", f"{m['recall']:.3f}")
                    c4.metric("Unit MAE", f"{m['unit_mae']:.3f}")
                if outer.log_table is not None and len(outer.history) % 5 == 0:
                    outer.log_table.dataframe(pd.DataFrame(outer.history).tail(25), use_container_width=True)
                target_hit = (
                    (outer.target_f1 <= 0 or m["f1"] >= outer.target_f1)
                    and (outer.target_precision <= 0 or m["precision"] >= outer.target_precision)
                    and (outer.target_recall <= 0 or m["recall"] >= outer.target_recall)
                )
                time_hit = outer.max_seconds > 0 and (time.time() - outer.start) >= outer.max_seconds
                if target_hit or time_hit:
                    self.model.stop_training = True
                    if outer.status is not None:
                        outer.status.write("Stopping criterion reached." if target_hit else "Maximum training time reached.")
        return CB()


def train_neural_network(
    dataset: pd.DataFrame,
    model_dir: str | Path = ".",
    epochs: int = 300,
    batch_size: int = 1024,
    learning_mode: str = "balanced",
    max_units: int = 80,
    continue_training: bool = False,
    learning_rate: float = 0.001,
    validation_split: float = 0.18,
    validation_method: str = "holdout_by_file",
    threshold_mode: str = "balanced",
    target_f1: float = 0.0,
    target_precision: float = 0.0,
    target_recall: float = 0.0,
    max_minutes: float = 0.0,
    progress=None,
    chart=None,
    status=None,
    metric_box=None,
    log_table=None,
) -> Dict:
    keras, _ = import_keras()
    model_dir = Path(model_dir)
    start = time.time()

    existing_model = None
    existing_pre = None
    existing_cols = None
    if continue_training:
        model_path = model_dir / "allocation_ai_model.keras"
        pre_path = model_dir / "allocation_ai_preprocessor.joblib"
        cols_path = model_dir / "allocation_ai_feature_columns.joblib"
        if model_path.exists() and pre_path.exists() and cols_path.exists():
            existing_model = keras.models.load_model(model_path)
            existing_pre = joblib.load(pre_path)
            existing_cols = joblib.load(cols_path)

    X_raw, Xt, y_units, y_binary, y_overstock, y_review, sw, pre, feature_columns = prepare_training_matrices(
        dataset, max_units=max_units, existing_feature_columns=existing_cols, existing_preprocessor=existing_pre
    )
    train_idx, val_idx = _split_indices(dataset, validation_method, validation_split)
    X_train, X_val = Xt[train_idx], Xt[val_idx]
    yu_train, yu_val = y_units[train_idx], y_units[val_idx]
    yb_train, yb_val = y_binary[train_idx], y_binary[val_idx]
    yo_train, yo_val = y_overstock[train_idx], y_overstock[val_idx]
    yr_train, yr_val = y_review[train_idx], y_review[val_idx]
    sw_train = sw[train_idx]

    if existing_model is not None:
        model = existing_model
    else:
        arch = architecture_for_mode(learning_mode)
        model = build_allocation_model(Xt.shape[1], max_units=max_units, learning_rate=learning_rate, **arch)

    live_cb = TargetScoreCallback(
        epochs, X_val, yu_val, yb_val,
        target_f1=target_f1, target_precision=target_precision, target_recall=target_recall, max_minutes=max_minutes,
        progress=progress, chart=chart, status=status, metric_box=metric_box, log_table=log_table,
    )
    callbacks = [
        live_cb.make(),
        keras.callbacks.ModelCheckpoint(str(model_dir / "allocation_ai_best_checkpoint.keras"), monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(8, int(epochs) // 25), min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=max(35, int(epochs) // 6), restore_best_weights=True),
        keras.callbacks.CSVLogger(str(model_dir / "training_log.csv"), append=bool(continue_training)),
    ]

    history = model.fit(
        X_train,
        {"units": yu_train, "alloc_prob": yb_train, "overstock_risk": yo_train, "review_sensitive": yr_train},
        validation_data=(X_val, {"units": yu_val, "alloc_prob": yb_val, "overstock_risk": yo_val, "review_sensitive": yr_val}),
        sample_weight={"units": sw_train, "alloc_prob": sw_train, "overstock_risk": np.ones_like(sw_train), "review_sensitive": np.ones_like(sw_train)},
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=0,
        callbacks=callbacks,
    )

    preds = model.predict(X_val, batch_size=int(batch_size), verbose=0)
    pred_units = np.argmax(np.asarray(preds["units"]), axis=1).astype(int)
    prob = np.asarray(preds["alloc_prob"]).reshape(-1)
    sweep = threshold_sweep(yb_val, prob, yu_val, pred_units)
    best_t = best_threshold_from_sweep(sweep, mode=threshold_mode)
    best = sweep.loc[(sweep["threshold"] - best_t).abs().idxmin()].to_dict()

    metadata = {
        "model_type": "single_advanced_keras_integer_flm_unit_network",
        "backend": "keras_torch",
        "rows_trained": int(len(Xt)),
        "positive_rows": int((y_units > 0).sum()),
        "positive_rate": float((y_units > 0).mean()) if len(y_units) else 0.0,
        "max_units_class": int(max_units),
        "epochs_requested": int(epochs),
        "epochs_completed_this_run": int(len(history.history.get("loss", []))),
        "batch_size": int(batch_size),
        "learning_mode": learning_mode,
        "validation_method": validation_method,
        "validation_split": float(validation_split),
        "threshold_mode": threshold_mode,
        "recommended_threshold": float(best_t),
        "validation_precision": float(best.get("precision", 0)),
        "validation_recall": float(best.get("recall", 0)),
        "validation_f1": float(best.get("f1", 0)),
        "validation_false_positive_rate": float(best.get("false_positive_rate", 0)),
        "validation_integer_unit_accuracy": float(best.get("exact_unit_accuracy", 0)),
        "validation_positive_integer_unit_accuracy": float(best.get("positive_unit_accuracy", 0)),
        "validation_integer_unit_mae": float(best.get("unit_mae", 0)),
        "feature_count_after_preprocessing": int(Xt.shape[1]),
        "raw_feature_columns": int(len(feature_columns)),
        "trained_seconds": round(time.time() - start, 2),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
    }
    save_artifacts(model, pre, feature_columns, model_dir, metadata)
    sweep.to_csv(model_dir / "validation_threshold_sweep.csv", index=False)
    pd.DataFrame(live_cb.history).to_csv(model_dir / "training_progress_live.csv", index=False)
    return metadata
