from __future__ import annotations

import numpy as np
import pandas as pd

from schema import CATEGORICAL_FIELDS, NUMERIC_FIELDS, build_column_map, normalize_header


def _num(s, default=0.0):
    if s is None:
        return pd.Series(default)
    return pd.to_numeric(s, errors="coerce")


def _txt(s):
    if s is None:
        return pd.Series("")
    return s.astype(str).fillna("")


def get_col(df: pd.DataFrame, field: str):
    cmap = build_column_map(df)
    c = cmap.get(field)
    if c is None or c not in df.columns:
        return None
    return df[c]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    cmap = build_column_map(df)
    n = len(df)
    out = pd.DataFrame(index=df.index)
    out["__row_order"] = df.get("__row_order", pd.Series(range(n), index=df.index)).values
    out["__excel_row"] = df.get("__excel_row", pd.Series(range(2, n + 2), index=df.index)).values

    # Direct numeric fields.
    for field in NUMERIC_FIELDS:
        s = df[cmap[field]] if cmap.get(field) in df.columns else pd.Series(np.nan, index=df.index)
        out[f"num__{field}"] = pd.to_numeric(s, errors="coerce")

    # Direct categorical fields.
    for field in CATEGORICAL_FIELDS:
        s = df[cmap[field]] if cmap.get(field) in df.columns else pd.Series("", index=df.index)
        out[f"cat__{field}"] = s.astype(str).fillna("").str.strip()

    # Safe defaults.
    flm = out["num__flm"].replace([np.inf, -np.inf], np.nan).fillna(1)
    flm = flm.where(flm > 0, 1).round().astype(float)
    out["num__flm"] = flm

    # Core demand / inventory engineered features.
    l30 = out["num__l30"].fillna(0)
    d30 = out["num__d30"].fillna(0)
    d60 = out["num__d60"].fillna(0)
    lw = out["num__lw"].fillna(0)
    ttm = out["num__ttm"].fillna(0)
    proj = out["num__proj_demand"].fillna(0)
    supply = out["num__supply"].fillna(0)
    qoh = out["num__qoh"].fillna(0)
    dc = out["num__dc_avail"].fillna(out["num__left_dc"]).fillna(0)
    left_dc = out["num__left_dc"].fillna(dc).fillna(0)
    alloc_rec = out["num__alloc_rec"].fillna(0)
    final_supply = out["num__final_supply"].fillna(supply)

    demand_basis = np.maximum.reduce([
        d60.values,
        proj.values,
        (l30 * 2.0).values,
        (d30 * 2.0).values,
        (ttm / 6.0).values,
        (lw * 8.0).values,
    ])
    out["num__demand_basis"] = demand_basis
    out["num__need_gap"] = np.maximum(0, out["num__demand_basis"] - supply)
    out["num__need_gap_flm_units"] = out["num__need_gap"] / flm
    out["num__alloc_rec_flm_units"] = alloc_rec / flm
    out["num__dc_avail_flm_units"] = dc / flm
    out["num__left_dc_flm_units"] = left_dc / flm
    out["num__supply_to_demand_ratio"] = supply / np.maximum(out["num__demand_basis"], 1)
    out["num__final_supply_to_demand_ratio"] = final_supply / np.maximum(out["num__demand_basis"], 1)
    out["num__projected_demand_gap"] = np.maximum(0, proj - supply)
    out["num__projected_gap_flm_units"] = out["num__projected_demand_gap"] / flm
    out["num__d60_gap"] = np.maximum(0, d60 - supply)
    out["num__d60_gap_flm_units"] = out["num__d60_gap"] / flm

    flag = out["cat__flag"].str.upper()
    out["num__is_allocate_flag"] = flag.str.contains("ALLOC", na=False).astype(float)
    out["num__is_review_flag"] = flag.str.contains("REVIEW", na=False).astype(float)
    out["num__is_no_alloc_flag"] = flag.str.contains("NO", na=False).astype(float)

    # Group-level context, preserving original row order.
    item = out["cat__item"].replace("", "__missing_item__")
    for name, base in [
        ("item_total_demand_basis", out["num__demand_basis"]),
        ("item_total_need_gap", out["num__need_gap"]),
        ("item_total_alloc_rec", alloc_rec),
        ("item_max_dc_avail", dc),
    ]:
        out[f"num__{name}"] = base.groupby(item).transform("sum" if "total" in name else "max").fillna(0)
    out["num__item_row_count"] = item.groupby(item).transform("size").astype(float)
    out["num__share_item_demand"] = out["num__demand_basis"] / np.maximum(out["num__item_total_demand_basis"], 1)
    out["num__share_item_need"] = out["num__need_gap"] / np.maximum(out["num__item_total_need_gap"], 1)
    out["num__row_rank_within_item"] = out.groupby(item).cumcount().astype(float)
    out["num__cum_alloc_rec_before_item"] = alloc_rec.groupby(item).cumsum().shift(1).fillna(0)
    out["num__projected_remaining_dc_after_rec"] = np.maximum(0, out["num__item_max_dc_avail"] - out["num__cum_alloc_rec_before_item"])

    # Clean numeric infinities.
    num_cols = [c for c in out.columns if c.startswith("num__")]
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    return out


def build_targets(df: pd.DataFrame, max_units: int = 80) -> pd.DataFrame:
    cmap = build_column_map(df)
    n = len(df)
    final_col = cmap.get("final_alloc")
    flm_col = cmap.get("flm")
    flag_col = cmap.get("flag")
    final = pd.to_numeric(df[final_col], errors="coerce").fillna(0) if final_col in df.columns else pd.Series(0, index=df.index)
    flm = pd.to_numeric(df[flm_col], errors="coerce").fillna(1) if flm_col in df.columns else pd.Series(1, index=df.index)
    flm = flm.where(flm > 0, 1).round()
    units = np.rint(final / flm).clip(0, int(max_units)).fillna(0).astype(int)
    y = pd.DataFrame(index=df.index)
    y["__target_units"] = units
    y["__target_alloc_binary"] = (units > 0).astype(int)

    # Auxiliary targets.
    feat = build_feature_frame(df)
    final_supply = feat["num__final_supply"].fillna(feat["num__supply"]).fillna(0)
    demand_basis = feat["num__demand_basis"].fillna(0)
    flm2 = feat["num__flm"].fillna(1).where(feat["num__flm"] > 0, 1)
    y["__target_overstock_risk"] = (final_supply > (demand_basis + flm2)).astype(int)
    y["__target_review_sensitive"] = feat["num__is_review_flag"].fillna(0).astype(int)
    return y
