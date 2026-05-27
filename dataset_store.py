from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from data_io import read_allocation_file, save_upload
from features import build_feature_frame, build_targets

DATASET_PATH = Path("allocation_training_dataset.parquet")
DATASET_PKL_PATH = Path("allocation_training_dataset_base.pkl")
DATASET_META_PATH = Path("allocation_training_dataset_meta.csv")


def build_dataset_from_frames(frames: List[pd.DataFrame], source_names: Optional[List[str]] = None, max_units: int = 80) -> pd.DataFrame:
    parts = []
    for i, df in enumerate(frames):
        name = source_names[i] if source_names and i < len(source_names) else f"file_{i+1}"
        X = build_feature_frame(df)
        y = build_targets(df, max_units=max_units)
        part = pd.concat([X, y], axis=1)
        part["__source_file"] = name
        part["__source_file_index"] = i
        useful = (
            part["cat__flag"].astype(str).str.len().gt(0)
            | part["__target_units"].gt(0)
            | part["num__alloc_rec"].fillna(0).gt(0)
            | part["num__demand_basis"].fillna(0).gt(0)
        )
        parts.append(part.loc[useful].copy())
    if not parts:
        raise ValueError("No valid training rows were found.")
    return pd.concat(parts, ignore_index=True)


def build_dataset_from_uploads(uploads, sheet_name: str = "3.3 Working Table", max_rows: Optional[int] = None, max_units: int = 80, progress=None) -> pd.DataFrame:
    frames, names = [], []
    total = len(uploads)
    for i, up in enumerate(uploads):
        path = save_upload(up, suffix=Path(up.name).suffix)
        df = read_allocation_file(path, sheet_name=sheet_name, max_rows=max_rows)
        frames.append(df)
        names.append(up.name)
        if progress is not None:
            progress.progress((i + 1) / max(total, 1), text=f"Read {i+1}/{total}: {up.name}")
    return build_dataset_from_frames(frames, names, max_units=max_units)


def save_dataset(ds: pd.DataFrame, path: Path = DATASET_PATH):
    # Prefer parquet on hosted Streamlit, but always write a pickle fallback so the app
    # remains usable if pyarrow is unavailable in a local or restricted environment.
    try:
        ds.to_parquet(path, index=False)
    except Exception:
        ds.to_pickle(DATASET_PKL_PATH)
    meta = ds.groupby("__source_file", dropna=False).agg(
        rows=("__target_units", "size"),
        positive_rows=("__target_units", lambda s: int((s > 0).sum())),
        positive_rate=("__target_units", lambda s: float((s > 0).mean())),
        max_units=("__target_units", "max"),
    ).reset_index()
    meta.to_csv(DATASET_META_PATH, index=False)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    if DATASET_PKL_PATH.exists():
        return pd.read_pickle(DATASET_PKL_PATH)
    raise FileNotFoundError("No cached dataset exists yet. Build one from Dataset Builder / Training Session tab.")


def dataset_exists() -> bool:
    return DATASET_PATH.exists() or DATASET_PKL_PATH.exists()
