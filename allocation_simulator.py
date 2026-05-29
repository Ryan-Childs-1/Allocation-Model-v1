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

    # Z - No Alloc. rows are not hard-blocked; they can be allocated when the
    # neural model or workbook demand signals justify it.
    allow_no_alloc_rows: bool = True
    no_alloc_min_probability: float = 0.65
    no_alloc_min_need_flm_units: float = 1.0

    # Review rows are intentionally handled in multiple passes.
    # Pass 1 is a conservative zero/blank scan. Passes 2 and 3 may add more
    # inventory to the same Review rows if demand, Alloc. Rec., probability,
    # and remaining Left DC still support it.
    review_passes: int = 3
    review_pass1_min_probability: float = 0.55
    review_pass2_min_probability: float = 0.70
    review_pass3_min_probability: float = 0.85


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
    """Return an integer allocation.

    Normal allocations are rounded down to FLM multiples. However, when the
    remaining available DC is positive but below one FLM, return the remaining
    integer units instead of blanking the row. This supports low remaining counts
    such as FLM=6, Left DC=4 -> Final Alloc=4.
    """
    try:
        flm = int(round(float(flm)))
    except Exception:
        flm = 1
    if flm <= 0:
        flm = 1
    try:
        x = float(x)
    except Exception:
        return 0
    if not np.isfinite(x) or x <= 0:
        return 0
    if x < flm:
        return int(max(0, np.floor(x)))
    return int(np.floor(x / flm) * flm)


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

    Simulation preserves input row order in the final CSV and reduces remaining DC by
    item after each accepted allocation.

    Review-specific behavior:
    - Non-Review rows are processed once in original row order.
    - Review rows are then intentionally processed up to three times.
      * Pass 1 is a conservative zero/blank scan: it is designed to catch Review
        rows that the model predicted as zero but the workbook demand / Alloc. Rec.
        signal says should receive at least one FLM.
      * Pass 2 can add the remaining model/demand-supported amount.
      * Pass 3 can add any final top-up still supported by demand, Alloc. Rec., model confidence, and Left DC.
    - Each Review pass sees the reduced item-level Left DC from previous passes.
    - Final values are integer units or blank. Most allocations are FLM multiples,
      but if remaining Left DC is positive and below one FLM, the app can allocate
      those remaining units instead of incorrectly blanking the row.
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

    final_by_idx = {idx: 0 for idx in df.index}
    audit_by_idx: dict[object, dict] = {}

    review_passes = int(getattr(cfg, "review_passes", 3) or 3)
    review_passes = max(1, min(review_passes, 3))
    review_thresholds = {
        1: float(getattr(cfg, "review_pass1_min_probability", 0.55)),
        2: float(getattr(cfg, "review_pass2_min_probability", 0.70)),
        3: float(getattr(cfg, "review_pass3_min_probability", 0.85)),
    }
    def _base_row_values(pos: int, idx):
        f = max(int(flm.loc[idx]), 1)
        prob = float(probabilities[pos])
        units = max(int(predicted_units[pos]), 0)
        raw_alloc = int(units * f)
        row_flag = flag.loc[idx]
        is_no_alloc = _is_z_no_alloc_flag(row_flag)
        is_review = "REVIEW" in row_flag
        is_allocate = ("ALLOC" in row_flag) and not is_no_alloc

        demand_cap_no_alloc_rec = max(
            0.0,
            float(demand_basis.loc[idx]) + float(cfg.demand_cap_extra_flm) * f - float(supply.loc[idx]),
        )
        demand_cap = demand_cap_no_alloc_rec
        if cfg.alloc_rec_influence == "hard_cap" and alloc_rec.loc[idx] > 0:
            demand_cap = min(demand_cap, float(alloc_rec.loc[idx]))
        elif cfg.alloc_rec_influence == "balanced" and alloc_rec.loc[idx] > 0:
            demand_cap = min(demand_cap, max(float(alloc_rec.loc[idx]) + f, f))
        elif cfg.alloc_rec_influence == "soft_cap" and alloc_rec.loc[idx] > 0:
            demand_cap = min(demand_cap, max(float(alloc_rec.loc[idx]) + 2 * f, f))

        need_units = max(0.0, demand_cap_no_alloc_rec / f)
        alloc_rec_units = max(0.0, float(alloc_rec.loc[idx]) / f) if f else 0.0
        return {
            "f": f,
            "prob": prob,
            "units": units,
            "raw_alloc": raw_alloc,
            "row_flag": row_flag,
            "is_no_alloc": is_no_alloc,
            "is_review": is_review,
            "is_allocate": is_allocate,
            "demand_cap": float(demand_cap),
            "demand_cap_no_alloc_rec": float(demand_cap_no_alloc_rec),
            "need_units": float(need_units),
            "alloc_rec_units": float(alloc_rec_units),
        }

    def _init_audit(pos: int, idx, vals: dict) -> dict:
        return {
            "row_order": int(df.get("__row_order", pd.Series(range(n), index=df.index)).loc[idx]),
            "excel_row": int(df.get("__excel_row", pd.Series(range(2, n + 2), index=df.index)).loc[idx]),
            "item": item.loc[idx],
            "flag": vals["row_flag"],
            "is_review": bool(vals["is_review"]),
            "is_z_no_alloc": bool(vals["is_no_alloc"]),
            "z_no_alloc_override": False,
            "probability": vals["prob"],
            "predicted_units": vals["units"],
            "flm": vals["f"],
            "raw_neural_alloc": vals["raw_alloc"],
            "need_units": vals["need_units"],
            "alloc_rec_units": vals["alloc_rec_units"],
            "demand_basis": float(demand_basis.loc[idx]),
            "demand_cap": vals["demand_cap"],
            "alloc_rec": float(alloc_rec.loc[idx]),
            "left_dc_before": None,
            "final_alloc": "",
            "left_dc_after": None,
            "review_passes_attempted": 0,
            "review_pass_1_added": 0,
            "review_pass_2_added": 0,
            "review_pass_3_added": 0,
            "review_total_added": 0,
            "allocated_on_pass": "",
            "reason": "",
        }

    def _available_after_existing(idx, vals: dict, left_before: float) -> float:
        current = float(final_by_idx.get(idx, 0))
        # Demand cap applies to the row's total allocation, so available increment is cap minus current.
        return max(0.0, min(left_before, vals["demand_cap"] - current))

    def _commit_allocation(idx, vals: dict, desired_increment: float, pass_label: str, reasons: list[str]) -> int:
        it = item.loc[idx]
        f = vals["f"]
        left_before = float(remaining.get(it, 0.0))
        inc_cap = _available_after_existing(idx, vals, left_before)
        increment = _round_down_to_flm(min(float(desired_increment), inc_cap), f)
        if increment <= 0:
            reasons.append(f"{pass_label}_rounded_or_capped_to_blank")
            return 0

        remaining[it] = max(0.0, left_before - increment)
        final_by_idx[idx] = int(final_by_idx.get(idx, 0) + increment)
        audit = audit_by_idx.setdefault(idx, _init_audit(0, idx, vals))
        if audit["left_dc_before"] is None:
            audit["left_dc_before"] = left_before
        audit["left_dc_after"] = float(remaining[it])
        audit["final_alloc"] = int(final_by_idx[idx])
        audit["review_total_added"] = int(audit.get("review_total_added", 0) + increment) if vals["is_review"] else audit.get("review_total_added", 0)
        if vals["is_review"]:
            pass_num = int(pass_label.replace("review_pass_", "")) if pass_label.startswith("review_pass_") else 0
            if pass_num in (1, 2, 3):
                key = f"review_pass_{pass_num}_added"
                audit[key] = int(audit.get(key, 0) + increment)
                existing_passes = str(audit.get("allocated_on_pass", ""))
                audit["allocated_on_pass"] = (existing_passes + ("," if existing_passes else "") + str(pass_num))
        reasons.append(f"{pass_label}_approved")
        return int(increment)

    # ------------------------------------------------------------------
    # Pass A: process all non-Review rows once in original order.
    # ------------------------------------------------------------------
    for pos, idx in enumerate(df.index):
        vals = _base_row_values(pos, idx)
        audit = _init_audit(pos, idx, vals)
        reasons: list[str] = []
        audit_by_idx[idx] = audit

        if vals["is_review"]:
            reasons.append("deferred_to_review_three_pass_logic")
            audit["reason"] = "; ".join(reasons)
            continue

        it = item.loc[idx]
        left_before = float(remaining.get(it, 0.0))
        audit["left_dc_before"] = left_before
        eligible = vals["is_allocate"]

        # Z - No Alloc. can be considered when necessary.
        if vals["is_no_alloc"]:
            model_override = vals["prob"] >= float(cfg.no_alloc_min_probability) and vals["raw_alloc"] >= vals["f"]
            demand_override = (
                vals["alloc_rec_units"] >= float(cfg.no_alloc_min_need_flm_units)
                and vals["need_units"] >= float(cfg.no_alloc_min_need_flm_units)
            )
            if cfg.allow_no_alloc_rows and left_before > 0 and (model_override or demand_override):
                eligible = True
                audit["z_no_alloc_override"] = True
                reasons.append("z_no_alloc_overridden_by_model_or_demand")
                if vals["raw_alloc"] < vals["f"]:
                    seed_units = int(np.floor(max(min(vals["alloc_rec_units"], vals["need_units"]), 0)))
                    if seed_units < 1 and vals["need_units"] >= cfg.no_alloc_min_need_flm_units:
                        seed_units = 1
                    vals["raw_alloc"] = max(vals["raw_alloc"], seed_units * vals["f"])
            else:
                eligible = False
                reasons.append("z_no_alloc_not_needed")

        if not eligible:
            reasons.append("not_allocate_row")
        elif vals["prob"] < cfg.min_probability:
            reasons.append("below_probability_threshold")
        elif left_before <= 0:
            reasons.append("no_left_dc")
        else:
            added = _commit_allocation(idx, vals, vals["raw_alloc"], "main_pass", reasons)
            if added and added < vals["raw_alloc"]:
                reasons.append("capped_by_dc_demand_or_alloc_rec")

        audit["left_dc_after"] = float(remaining.get(it, 0.0))
        audit["final_alloc"] = int(final_by_idx[idx]) if final_by_idx[idx] > 0 else ""
        if not reasons and final_by_idx[idx] <= 0:
            reasons.append("blank")
        audit["reason"] = "; ".join(reasons)

    # ------------------------------------------------------------------
    # Pass B: intentionally revisit Review rows up to three times.
    # ------------------------------------------------------------------
    review_positions = [(pos, idx) for pos, idx in enumerate(df.index) if "REVIEW" in flag.loc[idx]]

    for pass_num in range(1, review_passes + 1):
        threshold = review_thresholds.get(pass_num, cfg.min_probability)
        for pos, idx in review_positions:
            vals = _base_row_values(pos, idx)
            audit = audit_by_idx.setdefault(idx, _init_audit(pos, idx, vals))
            audit["review_passes_attempted"] = int(audit.get("review_passes_attempted", 0) + 1)
            reasons = [r for r in str(audit.get("reason", "")).split("; ") if r]

            it = item.loc[idx]
            left_before = float(remaining.get(it, 0.0))
            if audit.get("left_dc_before") is None:
                audit["left_dc_before"] = left_before

            if not cfg.allow_review_rows:
                reasons.append(f"review_pass_{pass_num}_review_rows_disabled")
                audit["reason"] = "; ".join(dict.fromkeys(reasons))
                audit["left_dc_after"] = left_before
                continue
            if left_before <= 0:
                reasons.append(f"review_pass_{pass_num}_no_left_dc")
                audit["reason"] = "; ".join(dict.fromkeys(reasons))
                audit["left_dc_after"] = left_before
                continue

            current_alloc = int(final_by_idx.get(idx, 0))
            f = vals["f"]
            raw_alloc = vals["raw_alloc"]
            need_seed_units = int(np.floor(max(min(vals["alloc_rec_units"], vals["need_units"]), 0)))

            desired_increment = 0
            if pass_num == 1:
                # First review pass intentionally looks for rows still at zero/blank.
                # It is conservative and can seed one FLM when workbook demand signals justify it,
                # even if the neural unit class predicted zero.
                if current_alloc == 0:
                    zero_scan_supported = (
                        vals["need_units"] >= 1.0
                        and (vals["alloc_rec_units"] >= 1.0 or vals["prob"] >= threshold)
                    )
                    if zero_scan_supported:
                        desired_increment = max(f, min(max(raw_alloc, f), f))
                        reasons.append("review_pass_1_zero_scan_supported")
                    else:
                        reasons.append("review_pass_1_zero_scan_no_action")
                else:
                    reasons.append("review_pass_1_already_allocated")
            elif pass_num == 2:
                # Second pass can add the remaining model/demand-supported amount.
                # There is intentionally NO artificial max-FLM-per-pass cap here.
                enough_signal = vals["prob"] >= threshold or need_seed_units >= 1
                target_total = max(raw_alloc, int(np.floor(vals["need_units"])) * f)
                if enough_signal and target_total > current_alloc:
                    desired_increment = target_total - current_alloc
                    reasons.append("review_pass_2_incremental_add_supported_no_flm_cap")
                else:
                    reasons.append("review_pass_2_no_action")
            elif pass_num == 3:
                # Third pass is the highest-confidence final top-up. It can add the full remaining
                # justified amount; there is intentionally NO artificial max-FLM-per-pass cap.
                enough_signal = vals["prob"] >= threshold or (vals["alloc_rec_units"] >= (current_alloc / f + 1) and vals["need_units"] >= (current_alloc / f + 1))
                target_total = max(raw_alloc, int(np.floor(min(vals["alloc_rec_units"], vals["need_units"]))) * f)
                if enough_signal and target_total > current_alloc:
                    desired_increment = target_total - current_alloc
                    reasons.append("review_pass_3_final_top_up_supported_no_flm_cap")
                else:
                    reasons.append("review_pass_3_no_action")

            if desired_increment > 0:
                before_add = int(final_by_idx.get(idx, 0))
                added = _commit_allocation(idx, vals, desired_increment, f"review_pass_{pass_num}", reasons)
                if added and int(final_by_idx.get(idx, 0)) < before_add + desired_increment:
                    reasons.append(f"review_pass_{pass_num}_capped_by_dc_demand_or_alloc_rec")
            # Preserve the row-level after-state from the last pass that actually changed
            # this row. If no allocation has happened for the row, record the current DC
            # state so the audit still explains why it stayed blank.
            if desired_increment > 0 or not audit.get("left_dc_after"):
                audit["left_dc_after"] = float(remaining.get(it, 0.0))
            audit["final_alloc"] = int(final_by_idx[idx]) if final_by_idx[idx] > 0 else ""
            audit["reason"] = "; ".join(dict.fromkeys(reasons))

    final_alloc = [int(final_by_idx[idx]) if int(final_by_idx.get(idx, 0)) > 0 else "" for idx in df.index]
    audit_rows = [audit_by_idx.get(idx, {}) for idx in df.index]
    audit_df = pd.DataFrame(audit_rows)
    return pd.Series(final_alloc, index=df.index, name="Final Alloc."), audit_df
