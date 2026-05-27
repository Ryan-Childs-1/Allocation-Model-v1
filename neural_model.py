from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Web-hosted Streamlit version uses Keras 3 on the PyTorch backend.
os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


class AllocationFeaturePreprocessor:
    """Fast hosted preprocessor for allocation workbooks.

    Uses scaled numeric fields plus stable hashed categorical bins. This avoids giant
    one-hot matrices that can break hosted Streamlit training sessions when many
    item/UPC/site values appear across multiple files.
    """
    def __init__(self, feature_columns: List[str] | None = None, cat_bins: int = 64):
        self.feature_columns = list(feature_columns or [])
        self.numeric_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.cat_bins = int(cat_bins)
        self.medians: Dict[str, float] = {}
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.output_feature_count_: int = 0

    def fit(self, X: pd.DataFrame):
        X = X.copy()
        if not self.feature_columns:
            self.feature_columns = [c for c in X.columns if c.startswith("num__") or c.startswith("cat__")]
        self.numeric_cols = [c for c in self.feature_columns if c.startswith("num__")]
        self.cat_cols = [c for c in self.feature_columns if c.startswith("cat__")]
        for c in self.numeric_cols:
            s = pd.to_numeric(X[c] if c in X.columns else pd.Series(np.nan, index=X.index), errors="coerce")
            med = float(s.median()) if s.notna().any() else 0.0
            filled = s.fillna(med).replace([np.inf, -np.inf], med)
            mean = float(filled.mean()) if len(filled) else 0.0
            std = float(filled.std()) if len(filled) and float(filled.std() or 0) > 1e-9 else 1.0
            self.medians[c] = med
            self.means[c] = mean
            self.stds[c] = std
        self.output_feature_count_ = len(self.numeric_cols) + len(self.cat_cols) * self.cat_bins
        return self

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.feature_columns:
            if c not in out.columns:
                out[c] = "" if c.startswith("cat__") else 0.0
        return out[self.feature_columns]

    def transform(self, X: pd.DataFrame):
        X = self._align(X)
        parts = []
        if self.numeric_cols:
            nums = []
            for c in self.numeric_cols:
                med = self.medians.get(c, 0.0)
                s = pd.to_numeric(X[c], errors="coerce").fillna(med).replace([np.inf, -np.inf], med)
                nums.append(((s - self.means.get(c, 0.0)) / self.stds.get(c, 1.0)).astype("float32").values)
            parts.append(np.vstack(nums).T.astype("float32"))
        for c in self.cat_cols:
            s = X[c].astype(str).fillna("")
            h = (hash_pandas_object(s, index=False).values % self.cat_bins).astype("int64")
            mat = np.zeros((len(X), self.cat_bins), dtype="float32")
            if len(X):
                mat[np.arange(len(X)), h] = 1.0
            parts.append(mat)
        if not parts:
            return np.zeros((len(X), 0), dtype="float32")
        return np.hstack(parts).astype("float32")

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)


def import_keras():
    try:
        import keras
        from keras import layers
        return keras, layers
    except Exception as exc:
        raise RuntimeError(
            "Keras could not start. This hosted app uses Keras 3 with the PyTorch backend. "
            "Install with: pip install -r requirements.txt"
        ) from exc


def make_preprocessor(X: pd.DataFrame) -> Tuple[AllocationFeaturePreprocessor, List[str]]:
    X = X.drop(columns=["__excel_row", "__row_order"], errors="ignore")
    feature_columns = [c for c in X.columns if c.startswith("num__") or c.startswith("cat__")]
    pre = AllocationFeaturePreprocessor(feature_columns=feature_columns, cat_bins=64)
    return pre, feature_columns


def align_to_training_columns(X: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    out = X.copy()
    for c in feature_columns:
        if c not in out.columns:
            out[c] = "" if c.startswith("cat__") else 0.0
    return out[feature_columns]


def to_dense_float32(arr) -> np.ndarray:
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.asarray(arr, dtype="float32")


def build_allocation_model(
    n_features: int,
    max_units: int = 80,
    width: int = 384,
    depth: int = 5,
    dropout: float = 0.18,
    learning_rate: float = 0.001,
    l2: float = 1e-5,
):
    keras, layers = import_keras()
    inp = keras.Input(shape=(n_features,), name="features")
    x = layers.Dense(width, activation="gelu", kernel_regularizer=keras.regularizers.l2(l2))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    for i in range(depth):
        residual = x
        x = layers.Dense(width, activation="gelu", kernel_regularizer=keras.regularizers.l2(l2), name=f"resblock_{i}_dense1")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(width, activation="gelu", kernel_regularizer=keras.regularizers.l2(l2), name=f"resblock_{i}_dense2")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add(name=f"resblock_{i}_add")([residual, x])
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)

    shared = layers.Dense(max(96, width // 2), activation="gelu", name="shared_dense_1")(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(dropout / 2)(shared)
    shared = layers.Dense(max(64, width // 4), activation="gelu", name="shared_dense_2")(shared)

    units_out = layers.Dense(max_units + 1, activation="softmax", name="units")(shared)
    alloc_out = layers.Dense(1, activation="sigmoid", name="alloc_prob")(shared)
    overstock_out = layers.Dense(1, activation="sigmoid", name="overstock_risk")(shared)
    review_out = layers.Dense(1, activation="sigmoid", name="review_sensitive")(shared)

    model = keras.Model(
        inputs=inp,
        outputs={"units": units_out, "alloc_prob": alloc_out, "overstock_risk": overstock_out, "review_sensitive": review_out},
        name="AllocationAI_Hosted_Integer_FLM_Network",
    )
    opt = keras.optimizers.AdamW(learning_rate=float(learning_rate), weight_decay=1e-4, clipnorm=1.0)
    model.compile(
        optimizer=opt,
        loss={
            "units": "sparse_categorical_crossentropy",
            "alloc_prob": "binary_crossentropy",
            "overstock_risk": "binary_crossentropy",
            "review_sensitive": "binary_crossentropy",
        },
        loss_weights={"units": 1.0, "alloc_prob": 0.55, "overstock_risk": 0.18, "review_sensitive": 0.12},
        metrics={"units": ["sparse_categorical_accuracy"], "alloc_prob": ["accuracy"]},
    )
    return model


def architecture_for_mode(mode: str):
    if mode == "maximum":
        return {"width": 768, "depth": 8, "dropout": 0.24}
    if mode == "wide":
        return {"width": 896, "depth": 5, "dropout": 0.26}
    if mode == "deep":
        return {"width": 512, "depth": 9, "dropout": 0.22}
    if mode == "fast":
        return {"width": 256, "depth": 3, "dropout": 0.16}
    return {"width": 448, "depth": 5, "dropout": 0.20}


def save_artifacts(model, preprocessor, feature_columns: List[str], model_dir: str | Path, metadata: Dict):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "allocation_ai_model.keras")
    joblib.dump(preprocessor, model_dir / "allocation_ai_preprocessor.joblib", compress=3)
    joblib.dump(feature_columns, model_dir / "allocation_ai_feature_columns.joblib", compress=3)
    (model_dir / "allocation_ai_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def load_artifacts(model_dir: str | Path = "."):
    keras, _ = import_keras()
    model_dir = Path(model_dir)
    model_path = model_dir / "allocation_ai_model.keras"
    pre_path = model_dir / "allocation_ai_preprocessor.joblib"
    cols_path = model_dir / "allocation_ai_feature_columns.joblib"
    if not model_path.exists() or not pre_path.exists() or not cols_path.exists():
        raise FileNotFoundError("Missing model artifacts. Train the neural network first or upload saved artifacts in Model Manager.")
    return keras.models.load_model(model_path), joblib.load(pre_path), joblib.load(cols_path)
