from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from schema import build_column_map


@dataclass
class AllocationConfig:
    min_probability: float = 0.35
    demand_cap_extra_flm: float = 1.0
    allow_review_rows: bool = True
    alloc_rec_influence: str = "balanced"  # feature_only, soft_cap, balanced, hard_cap
    prefer_left_dc: bool = True

    # New in this version: Z - No Alloc. rows are no longer hard-blocked.
    # They can be allocated when the neural model or workbook demand signals justify it.
    allow_no_alloc_rows: bool = True
    no_alloc_min_probability: float = 0.65
    no_alloc_min_need_flm_units: float = 1.0


def _num_series(df: pd.DataFrame, field: str, default: float = 0.0) -> pd.Series:
    cmap = build_column_map(df)
    c = cmap.get(field)
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _txt_series(df: pd.DataFrame, field: str) -> pd.Series:
    cmap = build_column_map(df)
    c = cmap.get(field)
    if c in df.columns:
        return df[c].astype(str).fillna("")
    return pd.Series("", index=df.index)


def _round_down_to_flm(x: float, flm: float) -> int:
    try:
        flm = int(round(float(flm)))
    except Exception:
        flm = 1
    if flm <= 0:
        flm = 1
    if x < flm:
        return 0
    return int(np.floor(float(x) / flm) * flm)


def _is_z_no_alloc_flag(row_flag: str) -> bool:
    """Detect the workbook's no-allocation flag without blocking normal Allocate rows."""
    text = (row_flag or "").upper().strip()
    # Common examples: "Z - No Alloc.", "NO ALLOC", "Z NO ALLOC".
    return (("NO" in text and "ALLOC" in text) or text.startswith("Z - NO") or text.startswith("Z NO"))


def apply_allocation_simulation(
    original_df: pd.DataFrame,
    predicted_units: np.ndarray,
    probabilities: np.ndarray,
    cfg: AllocationConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Convert neural integer-unit predictions into valid Final Alloc integers/blanks.

    Simulation preserves original row order and reduces remaining DC by item after each allocation.

    Important behavior:
    - Allocate and Review rows are eligible normally.
    - Z - No Alloc. rows are now eligible only when the model/demand signals justify it.
    - All final values remain integer FLM multiples or blank.
    """
    df = original_df.copy()
    n = len(df)
    predicted_units = np.asarray(predicted_units).astype(int)
    probabilities = np.asarray(probabilities).reshape(-1)
    if len(predicted_units) != n:
        predicted_units = np.resize(predicted_units, n)
    if len(probabilities) != n:
        probabilities = np.resize(probabilities, n)

    item = _txt_series(df, "item").replace("", "__missing_item__")
    flag = _txt_series(df, "flag").str.upper()
    flm = _num_series(df, "flm", 1).fillna(1).where(lambda s: s > 0, 1).round().astype(int)
    supply = _num_series(df, "supply", 0).fillna(0)
    dc_avail = _num_series(df, "dc_avail", 0).fillna(0)
    left_dc = _num_series(df, "left_dc", np.nan)
    d60 = _num_series(df, "d60", 0).fillna(0)
    proj = _num_series(df, "proj_demand", 0).fillna(0)
    l30 = _num_series(df, "l30", 0).fillna(0)
    d30 = _num_series(df, "d30", 0).fillna(0)
    ttm = _num_series(df, "ttm", 0).fillna(0)
    lw = _num_series(df, "lw", 0).fillna(0)
    alloc_rec = _num_series(df, "alloc_rec", 0).fillna(0)

    demand_basis = pd.Series(
        np.maximum.reduce([
            d60.values,
            proj.values,
            (l30 * 2).values,
            (d30 * 2).values,
            (ttm / 6).values,
            (lw * 8).values,
        ]),
        index=df.index,
    )

    # Initialize remaining DC by item. Prefer Left DC if it appears meaningful; otherwise use DC Avail.
    remaining = {}
    for it, gidx in item.groupby(item).groups.items():
        ld = left_dc.loc[gidx].dropna()
        da = dc_avail.loc[gidx].dropna()
        if cfg.prefer_left_dc and len(ld) and ld.max() > 0:
            remaining[it] = float(ld.max())
        elif len(da) and da.max() > 0:
            remaining[it] = float(da.max())
        elif len(ld):
            remaining[it] = float(max(ld.max(), 0))
        else:
            remaining[it] = 0.0

    final_alloc = []
    audit_rows: List[dict] = []
    for pos, idx in enumerate(df.index):
        it = item.loc[idx]
        f = max(int(flm.loc[idx]), 1)
        prob = float(probabilities[pos])
        units = max(int(predicted_units[pos]), 0)
        raw_alloc = int(units * f)
        left_before = float(remaining.get(it, 0.0))
        row_flag = flag.loc[idx]
        reasons = []

        is_no_alloc = _is_z_no_alloc_flag(row_flag)
        is_review = "REVIEW" in row_flag
        is_allocate = ("ALLOC" in row_flag) and not is_no_alloc

        # Demand-protective cap: do not push supply beyond demand basis + extra FLM buffer.
        demand_cap = max(
            0.0,
            float(demand_basis.loc[idx]) + float(cfg.demand_cap_extra_flm) * f - float(supply.loc[idx]),
        )
        demand_cap_before_alloc_rec = demand_cap

        if cfg.alloc_rec_influence == "hard_cap" and alloc_rec.loc[idx] > 0:
            demand_cap = min(demand_cap, float(alloc_rec.loc[idx]))
        elif cfg.alloc_rec_influence == "balanced" and alloc_rec.loc[idx] > 0:
            demand_cap = min(demand_cap, max(float(alloc_rec.loc[idx]) + f, f))
        elif cfg.alloc_rec_influence == "soft_cap" and alloc_rec.loc[idx] > 0:
            demand_cap = min(demand_cap, max(float(alloc_rec.loc[idx]) + 2 * f, f))
        # feature_only: no alloc rec cap.

        need_units = max(0.0, demand_cap_before_alloc_rec / f)
        alloc_rec_units = max(0.0, float(alloc_rec.loc[idx]) / f) if f else 0.0

        eligible = is_allocate or (cfg.allow_review_rows and is_review)

        # New behavior: Z - No Alloc. can be considered when necessary.
        no_alloc_override = False
        if is_no_alloc:
            model_override = prob >= float(cfg.no_alloc_min_probability) and raw_alloc >= f
            demand_override = (
                alloc_rec_units >= float(cfg.no_alloc_min_need_flm_units)
                and need_units >= float(cfg.no_alloc_min_need_flm_units)
            )
            if cfg.allow_no_alloc_rows and left_before > 0 and (model_override or demand_override):
                eligible = True
                no_alloc_override = True
                reasons.append("z_no_alloc_overridden_by_model_or_demand")
                # If the model predicted zero because the historical flag was usually no-allocation,
                # seed a conservative integer allocation from Alloc. Rec. / demand need.
                if raw_alloc < f:
                    seed_units = int(np.floor(max(min(alloc_rec_units, need_units), 0)))
                    if seed_units < 1 and need_units >= cfg.no_alloc_min_need_flm_units:
                        seed_units = 1
                    raw_alloc = max(raw_alloc, seed_units * f)
            else:
                eligible = False
                reasons.append("z_no_alloc_not_needed")

        if not eligible:
            raw_alloc = 0
            if not is_no_alloc:
                reasons.append("not_allocate_or_review")
        if prob < cfg.min_probability and not no_alloc_override:
            raw_alloc = 0
            reasons.append("below_probability_threshold")
        if left_before <= 0:
            raw_alloc = 0
            reasons.append("no_left_dc")

        capped = min(float(raw_alloc), left_before, demand_cap)
        final = _round_down_to_flm(capped, f)
        if final <= 0:
            output_value = ""
            final_int = 0
            if not reasons:
                reasons.append("rounded_or_capped_to_blank")
        else:
            output_value = int(final)
            final_int = int(final)
            if final_int < raw_alloc:
                reasons.append("capped_by_dc_demand_or_alloc_rec")
            else:
                reasons.append("approved")
        remaining[it] = max(0.0, left_before - final_int)
        final_alloc.append(output_value)
        audit_rows.append({
            "row_order": int(df.get("__row_order", pd.Series(range(n), index=df.index)).loc[idx]),
            "excel_row": int(df.get("__excel_row", pd.Series(range(2, n + 2), index=df.index)).loc[idx]),
            "item": it,
            "flag": row_flag,
            "is_z_no_alloc": bool(is_no_alloc),
            "z_no_alloc_override": bool(no_alloc_override),
            "probability": prob,
            "predicted_units": units,
            "flm": f,
            "raw_neural_alloc": raw_alloc,
            "need_units": float(need_units),
            "alloc_rec_units": float(alloc_rec_units),
            "demand_basis": float(demand_basis.loc[idx]),
            "demand_cap": float(demand_cap),
            "alloc_rec": float(alloc_rec.loc[idx]),
            "left_dc_before": left_before,
            "final_alloc": final_int if final_int > 0 else "",
            "left_dc_after": float(remaining[it]),
            "reason": "; ".join(reasons),
        })
    return pd.Series(final_alloc, index=df.index, name="Final Alloc."), pd.DataFrame(audit_rows)
