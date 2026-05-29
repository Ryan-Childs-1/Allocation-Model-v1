from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


class AllocationFeaturePreprocessor:
    """Feature preprocessor used by the included compressed sklearn neural model.

    This class is intentionally kept in `neural_model.py` so older saved joblib
    models can load successfully. It uses scaled numeric features and stable
    hashed categorical bins, which is light enough for hosted Streamlit.
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
