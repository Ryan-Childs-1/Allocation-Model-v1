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

    # Three-pass Review context features. These do not use the target Final Alloc.
    # They help the neural network learn pass-aware Review behavior:
    #   pass 1 = zero/blank Review scan,
    #   pass 2 = alloc-rec supported additions,
    #   pass 3 = remaining demand-supported top-ups.
    # There is intentionally no max-FLM-per-pass feature or cap.
    out["num__review_pass1_zero_scan_candidate"] = (out["num__is_review_flag"] > 0).astype(float)
    out["num__review_pass2_alloc_rec_candidate"] = ((out["num__is_review_flag"] > 0) & (out["num__alloc_rec_flm_units"] > 0)).astype(float)
    out["num__review_pass3_need_topup_candidate"] = ((out["num__is_review_flag"] > 0) & (out["num__need_gap_flm_units"] > 0)).astype(float)
    out["num__review_alloc_rec_vs_need_units"] = out["num__alloc_rec_flm_units"] - out["num__need_gap_flm_units"]
    out["num__review_target_pressure_units"] = np.maximum(out["num__alloc_rec_flm_units"], out["num__need_gap_flm_units"])

    # ------------------------------------------------------------------
    # High-impact group/rank allocation context, preserving original row order.
    # These features help the model understand scarcity and relative priority
    # inside each item group instead of treating every row as isolated.
    # They do NOT use Final Alloc, so they are safe at prediction time.
    # ------------------------------------------------------------------
    item = out["cat__item"].replace("", "__missing_item__")
    item_groups = item.groupby(item).groups

    group_specs = [
        ("item_total_demand_basis", out["num__demand_basis"], "sum"),
        ("item_total_need_gap", out["num__need_gap"], "sum"),
        ("item_total_alloc_rec", alloc_rec, "sum"),
        ("item_total_projected_demand", proj, "sum"),
        ("item_total_d60", d60, "sum"),
        ("item_total_supply", supply, "sum"),
        ("item_total_qoh", qoh, "sum"),
        ("item_max_dc_avail", dc, "max"),
        ("item_max_left_dc", left_dc, "max"),
    ]
    for name, base, agg in group_specs:
        out[f"num__{name}"] = base.groupby(item).transform(agg).fillna(0)

    out["num__item_row_count"] = item.groupby(item).transform("size").astype(float)
    out["num__share_item_demand"] = out["num__demand_basis"] / np.maximum(out["num__item_total_demand_basis"], 1)
    out["num__share_item_need"] = out["num__need_gap"] / np.maximum(out["num__item_total_need_gap"], 1)
    out["num__share_item_alloc_rec"] = alloc_rec / np.maximum(out["num__item_total_alloc_rec"], 1)
    out["num__share_item_supply"] = supply / np.maximum(out["num__item_total_supply"], 1)

    # Scarcity and pressure signals. Values above 1 imply item-level demand/recommendation
    # exceeds available DC and row prioritization matters more.
    item_dc = np.maximum(out["num__item_max_left_dc"], out["num__item_max_dc_avail"])
    out["num__item_dc_to_need_ratio"] = item_dc / np.maximum(out["num__item_total_need_gap"], 1)
    out["num__item_dc_to_alloc_rec_ratio"] = item_dc / np.maximum(out["num__item_total_alloc_rec"], 1)
    out["num__item_need_to_dc_ratio"] = out["num__item_total_need_gap"] / np.maximum(item_dc, 1)
    out["num__item_alloc_rec_to_dc_ratio"] = out["num__item_total_alloc_rec"] / np.maximum(item_dc, 1)
    out["num__item_supply_gap_pressure"] = np.maximum(0, out["num__item_total_demand_basis"] - out["num__item_total_supply"]) / np.maximum(item_dc, 1)

    # Row order and cumulative pressure. These approximate how much item-level
    # opportunity has already appeared before the current row.
    out["num__row_rank_within_item"] = out.groupby(item).cumcount().astype(float)
    out["num__row_order_pct_within_item"] = out["num__row_rank_within_item"] / np.maximum(out["num__item_row_count"] - 1, 1)
    out["num__cum_alloc_rec_before_item"] = alloc_rec.groupby(item).cumsum().groupby(item).shift(1).fillna(0)
    out["num__cum_need_before_item"] = out["num__need_gap"].groupby(item).cumsum().groupby(item).shift(1).fillna(0)
    out["num__cum_demand_before_item"] = out["num__demand_basis"].groupby(item).cumsum().groupby(item).shift(1).fillna(0)
    out["num__remaining_need_after_row_order"] = np.maximum(0, out["num__item_total_need_gap"] - out["num__cum_need_before_item"] - out["num__need_gap"])
    out["num__remaining_alloc_rec_after_row_order"] = np.maximum(0, out["num__item_total_alloc_rec"] - out["num__cum_alloc_rec_before_item"] - alloc_rec)
    out["num__projected_remaining_dc_after_rec"] = np.maximum(0, item_dc - out["num__cum_alloc_rec_before_item"])

    # Within-item descending ranks. Lower rank number = stronger row by that signal.
    out["num__rank_need_within_item"] = out["num__need_gap"].groupby(item).rank(method="average", ascending=False).fillna(0)
    out["num__rank_demand_within_item"] = out["num__demand_basis"].groupby(item).rank(method="average", ascending=False).fillna(0)
    out["num__rank_alloc_rec_within_item"] = alloc_rec.groupby(item).rank(method="average", ascending=False).fillna(0)
    out["num__pct_rank_need_within_item"] = out["num__rank_need_within_item"] / np.maximum(out["num__item_row_count"], 1)
    out["num__pct_rank_demand_within_item"] = out["num__rank_demand_within_item"] / np.maximum(out["num__item_row_count"], 1)
    out["num__pct_rank_alloc_rec_within_item"] = out["num__rank_alloc_rec_within_item"] / np.maximum(out["num__item_row_count"], 1)

    # Clean numeric infinities.
    num_cols = [c for c in out.columns if c.startswith("num__")]
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    return out


def build_targets(df: pd.DataFrame, max_units: int = 80) -> pd.DataFrame:
    """Build integer FLM-unit targets plus three-pass Review auxiliary targets.

    Important: max_units is treated as a requested starting class count, not as a
    hard business limit. If the data contains a larger historical allocation in
    FLM units, targets are allowed to expand to that observed maximum so the
    model can learn the correct number of FLMs.
    """
    cmap = build_column_map(df)
    final_col = cmap.get("final_alloc")
    flm_col = cmap.get("flm")
    final = pd.to_numeric(df[final_col], errors="coerce").fillna(0) if final_col in df.columns else pd.Series(0, index=df.index)
    flm = pd.to_numeric(df[flm_col], errors="coerce").fillna(1) if flm_col in df.columns else pd.Series(1, index=df.index)
    flm = flm.where(flm > 0, 1).round()

    raw_units = np.rint(final / flm).fillna(0).astype(int)
    observed_max = int(max(int(max_units), raw_units.max() if len(raw_units) else 0))
    units = raw_units.clip(0, observed_max).astype(int)

    y = pd.DataFrame(index=df.index)
    y["__target_units"] = units
    y["__target_alloc_binary"] = (units > 0).astype(int)

    feat = build_feature_frame(df)
    final_supply = feat["num__final_supply"].fillna(feat["num__supply"]).fillna(0)
    demand_basis = feat["num__demand_basis"].fillna(0)
    flm2 = feat["num__flm"].fillna(1).where(feat["num__flm"] > 0, 1)
    is_review = feat["num__is_review_flag"].fillna(0).astype(int)

    y["__target_overstock_risk"] = (final_supply > (demand_basis + flm2)).astype(int)
    y["__target_review_sensitive"] = is_review

    # Synthetic three-pass Review labels. Historical workbooks only show the
    # final human allocation, not intermediate pass states, so these labels train
    # pass-specific gate behavior without limiting the total FLM units:
    #   pass1: Review row should move from zero/blank to nonzero.
    #   pass2: Review row likely deserves additional allocation beyond a single
    #          starter unit, often supported by Alloc. Rec.
    #   pass3: Review row likely deserves final top-up attention, usually larger
    #          allocations or larger remaining need.
    alloc_rec_units = feat["num__alloc_rec_flm_units"].fillna(0)
    need_units = feat["num__need_gap_flm_units"].fillna(0)
    y["__target_review_pass1"] = ((is_review > 0) & (units > 0)).astype(int)
    y["__target_review_pass2"] = ((is_review > 0) & (units > 1) & ((alloc_rec_units >= 1) | (need_units >= 1))).astype(int)
    y["__target_review_pass3"] = ((is_review > 0) & (units > 2) & ((alloc_rec_units >= 2) | (need_units >= 2))).astype(int)
    return y

