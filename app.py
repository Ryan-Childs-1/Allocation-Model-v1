from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from allocation_simulator import AllocationConfig
from data_io import dataframe_to_csv_bytes, read_allocation_file, save_upload
from schema import ColumnDiagnostics, build_column_map

# Predictor imports are loaded as a module instead of named imports so the app
# cannot crash from a stale/mismatched predictor.py during deployment. If the
# import itself fails, the Streamlit UI will show a clear error instead of a
# redacted top-level ImportError.
try:
    import predictor as _predictor
    _PREDICTOR_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - deployment safety path
    _predictor = None
    _PREDICTOR_IMPORT_ERROR = _exc


def _require_predictor():
    if _predictor is None:
        raise ImportError(
            "Could not import predictor.py. Make sure app.py, predictor.py, features.py, "
            "neural_model.py, allocation_simulator.py, data_io.py, and schema.py were all "
            "uploaded from the same app zip. Original error: "
            f"{type(_PREDICTOR_IMPORT_ERROR).__name__}: {_PREDICTOR_IMPORT_ERROR}"
        )
    return _predictor


def read_metadata(*args, **kwargs):
    if _predictor is None:
        return {"import_error": f"{type(_PREDICTOR_IMPORT_ERROR).__name__}: {_PREDICTOR_IMPORT_ERROR}"}
    return _predictor.read_metadata(*args, **kwargs)


def load_model_bundle(*args, **kwargs):
    return _require_predictor().load_model_bundle(*args, **kwargs)


def predict_to_outputs(*args, **kwargs):
    return _require_predictor().predict_to_outputs(*args, **kwargs)


def model_feature_importance(*args, **kwargs):
    mod = _require_predictor()
    if not hasattr(mod, "model_feature_importance"):
        return pd.DataFrame()
    return mod.model_feature_importance(*args, **kwargs)


def prediction_feature_relationships(*args, **kwargs):
    mod = _require_predictor()
    if not hasattr(mod, "prediction_feature_relationships"):
        return pd.DataFrame()
    return mod.prediction_feature_relationships(*args, **kwargs)

st.set_page_config(page_title="Allocation AI Predictor", page_icon="🎯", layout="wide")

APP_TITLE = "🎯 Allocation AI Predictor"
BASE_MODEL_LABEL = "Base NN Model"


def _safe_json_from_zip(uploaded_file: Any) -> dict:
    """Lightweight metadata preview for uploaded model artifact zips."""
    try:
        name = getattr(uploaded_file, "name", "")
        if not name.lower().endswith(".zip"):
            return {}
        data = uploaded_file.getbuffer()
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            json_names = [n for n in z.namelist() if n.lower().endswith(".json")]

            def score(n: str) -> tuple[int, str]:
                low = n.lower()
                s = 0
                if "camp" in low:
                    s += 400
                if "app_compatible" in low:
                    s += 350
                if "metadata" in low:
                    s += 250
                if "torch" in low:
                    s += 50
                if "baseline" in low:
                    s -= 150
                return s, n

            for n in sorted(json_names, key=score, reverse=True):
                try:
                    obj = json.loads(z.read(n).decode("utf-8", errors="replace"))
                    if isinstance(obj, dict):
                        obj["__metadata_file"] = n
                        return obj
                except Exception:
                    continue
    except Exception:
        return {}
    return {}


def _metadata_best_threshold(meta: dict, fallback: float = 0.05) -> float:
    if not isinstance(meta, dict):
        return fallback
    candidates = [meta.get("best_threshold"), meta.get("recommended_threshold")]
    best_metrics = meta.get("best_validation_metrics")
    if isinstance(best_metrics, dict):
        candidates.append(best_metrics.get("threshold"))
    smoke = meta.get("streamlit_reload_smoke_test")
    if isinstance(smoke, dict):
        candidates.append(smoke.get("threshold"))
    for val in candidates:
        try:
            if val is not None:
                f = float(val)
                if 0.0 < f <= 1.0:
                    return f
        except Exception:
            pass
    return fallback


def _metadata_summary(meta: dict) -> dict:
    best = meta.get("best_validation_metrics", {}) if isinstance(meta.get("best_validation_metrics", {}), dict) else {}
    smoke = meta.get("streamlit_reload_smoke_test", {}) if isinstance(meta.get("streamlit_reload_smoke_test", {}), dict) else {}
    return {
        "backend": meta.get("backend", "unknown"),
        "artifact_short_name": meta.get("artifact_short_name", ""),
        "rows_total": meta.get("rows_total", meta.get("rows_trained", meta.get("base_model_rows_trained", 0))),
        "rows_train": meta.get("rows_train", None),
        "rows_validation": meta.get("rows_validation", None),
        "positive_rows_total": meta.get("positive_rows_total", None),
        "threshold": _metadata_best_threshold(meta, 0.05),
        "f1": meta.get("validation_f1", best.get("f1", None)),
        "precision": meta.get("validation_precision", best.get("precision", None)),
        "recall": meta.get("validation_recall", best.get("recall", None)),
        "unit_accuracy": best.get("unit_accuracy", None),
        "positive_unit_accuracy": best.get("positive_unit_accuracy", None),
        "unit_mae": best.get("unit_mae", None),
        "false_positive_rate": best.get("false_positive_rate", None),
        "smoke_test_f1": smoke.get("f1", None),
        "smoke_test_precision": smoke.get("precision", None),
        "smoke_test_recall": smoke.get("recall", None),
        "smoke_test_unit_accuracy": smoke.get("unit_accuracy", None),
    }


def _fmt_num(x, digits=3, default="—"):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return default
        return f"{float(x):.{digits}f}"
    except Exception:
        return default


def _load_baseline_metrics() -> dict:
    p = Path("camp_alloc_rec_baseline_metrics.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


included_meta = read_metadata()
included_summary = _metadata_summary(included_meta)
baseline_metrics = _load_baseline_metrics()

# Sidebar model + prediction settings.
with st.sidebar:
    st.header("AI model selector")
    uploaded_models = st.file_uploader(
        "Upload additional model artifacts",
        type=["zip", "joblib", "pkl"],
        accept_multiple_files=True,
        help=(
            "Optional: upload one or more trained artifact zips or app-compatible .joblib/.pkl bundles. "
            "The included model is your latest trained Base NN Model."
        ),
    )

    model_labels = [BASE_MODEL_LABEL]
    if uploaded_models:
        model_labels.extend([f"Uploaded: {m.name}" for m in uploaded_models])

    selected_label = st.selectbox("Choose model for prediction", model_labels, index=0)
    selected_uploaded_model = None
    if selected_label.startswith("Uploaded:") and uploaded_models:
        selected_idx = model_labels.index(selected_label) - 1
        selected_uploaded_model = uploaded_models[selected_idx]

    selected_preview_meta = included_meta
    if selected_uploaded_model is not None:
        selected_preview_meta = _safe_json_from_zip(selected_uploaded_model)
        if not selected_preview_meta:
            selected_preview_meta = {"backend": "uploaded direct model", "best_threshold": included_summary.get("threshold", 0.05)}

    selected_summary = _metadata_summary(selected_preview_meta)
    st.caption(f"Selected model: **{selected_label}**")
    st.caption(f"Suggested threshold: **{selected_summary['threshold']:.2f}**")

    with st.expander("Selected model metadata preview", expanded=False):
        st.json(selected_summary)
        if selected_preview_meta.get("__metadata_file"):
            st.caption(f"Metadata preview file: `{selected_preview_meta['__metadata_file']}`")

    st.header("Prediction controls")
    default_threshold = _metadata_best_threshold(selected_preview_meta, _metadata_best_threshold(included_meta, 0.05))
    use_model_threshold = st.checkbox("Use selected model's saved threshold", value=True)
    manual_probability = st.slider(
        "Manual minimum allocation probability",
        min_value=0.01,
        max_value=0.99,
        value=min(max(float(default_threshold), 0.01), 0.99),
        step=0.01,
        disabled=use_model_threshold,
    )
    min_probability = float(default_threshold if use_model_threshold else manual_probability)

    demand_extra = st.slider("Demand cap extra FLM", 0.0, 8.0, 1.0, 0.25)
    allow_review = st.checkbox("Allow Review rows", value=True)
    review_passes = st.slider("Review-row passes", 1, 3, 3, 1)
    review_pass1_prob = st.slider("Review pass 1 zero-scan probability", 0.10, 0.99, 0.55, 0.01)
    review_pass2_prob = st.slider("Review pass 2 add-more probability", 0.10, 0.99, 0.70, 0.01)
    review_pass3_prob = st.slider("Review pass 3 final top-up probability", 0.10, 0.99, 0.85, 0.01)
    allow_no_alloc = st.checkbox("Allow Z - No Alloc. rows if justified", value=True)
    no_alloc_prob = st.slider("Z - No Alloc override probability", 0.10, 0.99, 0.75, 0.01)
    no_alloc_need = st.slider("Z - No Alloc minimum need / Alloc. Rec. units", 0.0, 10.0, 1.0, 0.5)
    prefer_left_dc = st.checkbox("Prefer Left DC over DC Avail", value=True)
    alloc_rec_influence = st.selectbox(
        "Alloc. Rec. influence",
        ["feature_only", "soft_cap", "balanced", "hard_cap"],
        index=2,
        help="Controls whether Alloc. Rec. is only a model feature or also constrains the simulator output.",
    )

st.title(APP_TITLE)
st.caption("Prediction-only app · Base NN Model included · multi-model selector · completed CSV + audit + AI insights")
if _PREDICTOR_IMPORT_ERROR is not None:
    st.error("The prediction engine did not import correctly. This usually means the GitHub repo has mismatched files from different app versions.")
    st.exception(_PREDICTOR_IMPORT_ERROR)
    st.stop()

predict_tab, insights_tab, process_tab, model_tab = st.tabs([
    "Predict Allocation",
    "Prediction Insights",
    "AI Process & Model Info",
    "Model Metrics",
])

with predict_tab:
    st.markdown("## 1. Upload allocation file")
    file = st.file_uploader("Upload .xlsb, .xlsx, or .csv allocation file", type=["xlsb", "xlsx", "csv"])
    sheet = st.text_input("Sheet name for Excel files", value="3.3 Working Table")

    if file is not None:
        try:
            path = save_upload(file, suffix=Path(file.name).suffix)
            with st.spinner("Reading file and preserving row order..."):
                df = read_allocation_file(path, sheet_name=sheet)
            st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns from `{file.name}`")

            with st.expander("Detected column mapping", expanded=False):
                diag = ColumnDiagnostics(rows=len(df), columns=len(df.columns), header_map=build_column_map(df))
                st.dataframe(pd.DataFrame(diag.as_rows()), use_container_width=True, height=360)

            st.markdown("## 2. Run prediction")
            if st.button("Predict Final Alloc", type="primary"):
                cfg = AllocationConfig(
                    min_probability=float(min_probability),
                    demand_cap_extra_flm=float(demand_extra),
                    allow_review_rows=bool(allow_review),
                    alloc_rec_influence=str(alloc_rec_influence),
                    prefer_left_dc=bool(prefer_left_dc),
                    allow_no_alloc_rows=bool(allow_no_alloc),
                    no_alloc_min_probability=float(no_alloc_prob),
                    no_alloc_min_need_flm_units=float(no_alloc_need),
                    review_passes=int(review_passes),
                    review_pass1_min_probability=float(review_pass1_prob),
                    review_pass2_min_probability=float(review_pass2_prob),
                    review_pass3_min_probability=float(review_pass3_prob),
                )
                with st.spinner("Loading selected model and running sequential allocation simulation..."):
                    bundle = load_model_bundle(selected_uploaded_model)
                    out_df, audit_df, summary = predict_to_outputs(df, bundle, cfg)

                artifact_meta = bundle.get("__artifact_metadata", {}) if isinstance(bundle, dict) else {}
                if artifact_meta:
                    summary["artifact_metadata"] = artifact_meta
                if isinstance(bundle, dict):
                    summary["model_source"] = bundle.get("__artifact_model_name", selected_label)
                    summary["model_selection"] = selected_label
                    summary["min_probability_used"] = min_probability
                    summary["selected_model_threshold"] = default_threshold
                    if bundle.get("__compat_repairs"):
                        summary["compatibility_repairs"] = bundle.get("__compat_repairs")

                    try:
                        fi = model_feature_importance(bundle, top_n=60)
                    except Exception:
                        fi = pd.DataFrame()
                    try:
                        rel = prediction_feature_relationships(df, audit_df, top_n=60)
                    except Exception:
                        rel = pd.DataFrame()
                else:
                    fi = pd.DataFrame()
                    rel = pd.DataFrame()

                st.session_state["input_df"] = df
                st.session_state["out_df"] = out_df
                st.session_state["audit_df"] = audit_df
                st.session_state["summary"] = summary
                st.session_state["feature_importance"] = fi
                st.session_state["feature_relationships"] = rel
                st.success("Prediction complete. Final Alloc values are integers or blank.")

        except Exception as exc:
            st.error("File loading or prediction setup failed.")
            st.exception(exc)

    if "out_df" in st.session_state:
        out_df = st.session_state["out_df"]
        audit_df = st.session_state["audit_df"]
        summary = st.session_state["summary"]

        st.markdown("## 3. Prediction summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{summary['rows']:,}")
        c2.metric("Allocated rows", f"{summary['allocated_rows']:,}")
        c3.metric("Total Final Alloc", f"{summary['total_final_alloc']:,}")
        c4.metric("Mean probability", f"{summary['mean_probability']:.3f}")
        c5.metric("Z - No Alloc overrides", f"{summary['z_no_alloc_overrides']:,}")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Review rows allocated", f"{summary.get('review_rows_allocated', 0):,}")
        r2.metric("Review total alloc", f"{summary.get('review_total_final_alloc', 0):,}")
        r3.metric("Review pass 1 added", f"{summary.get('review_pass_1_added', 0):,}")
        r4.metric("Review pass 2 added", f"{summary.get('review_pass_2_added', 0):,}")
        r5.metric("Review pass 3 added", f"{summary.get('review_pass_3_added', 0):,}")

        if summary.get("model_source"):
            st.info(f"Model used: `{summary['model_selection']}` / `{summary['model_source']}` · threshold used: `{summary.get('min_probability_used', '')}`")
        if summary.get("compatibility_repairs"):
            st.warning("Applied sklearn compatibility repairs to model bundle: " + ", ".join(summary["compatibility_repairs"]))

        with st.expander("Run metadata", expanded=False):
            st.json(summary)

        st.markdown("## 4. Preview outputs")
        left, right = st.columns(2)
        with left:
            st.subheader("Completed allocation preview")
            st.dataframe(out_df.head(250), use_container_width=True, height=420)
        with right:
            st.subheader("Audit preview")
            st.dataframe(audit_df.head(250), use_container_width=True, height=420)

        completed_csv = dataframe_to_csv_bytes(out_df)
        audit_csv = dataframe_to_csv_bytes(audit_df)
        summary_bytes = json.dumps(summary, indent=2, default=str).encode("utf-8")
        feature_imp_csv = dataframe_to_csv_bytes(st.session_state.get("feature_importance", pd.DataFrame()))
        rel_csv = dataframe_to_csv_bytes(st.session_state.get("feature_relationships", pd.DataFrame()))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("completed_allocation.csv", completed_csv)
            z.writestr("allocation_audit.csv", audit_csv)
            z.writestr("prediction_summary.json", summary_bytes)
            z.writestr("model_feature_importance.csv", feature_imp_csv)
            z.writestr("prediction_feature_relationships.csv", rel_csv)
        zip_bytes = zip_buffer.getvalue()

        st.markdown("## 5. Downloads")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download completed CSV", completed_csv, "completed_allocation.csv", "text/csv")
        d2.download_button("Download audit CSV", audit_csv, "allocation_audit.csv", "text/csv")
        d3.download_button("Download output ZIP", zip_bytes, "allocation_ai_prediction_output.zip", "application/zip")
    else:
        st.info("Upload an allocation file, select a model, then run prediction.")

with insights_tab:
    st.markdown("## Prediction Insights")
    if "audit_df" not in st.session_state:
        st.info("Run a prediction first to populate run-specific insights.")
    else:
        audit_df = st.session_state["audit_df"].copy()
        out_df = st.session_state["out_df"].copy()
        fi = st.session_state.get("feature_importance", pd.DataFrame()).copy()
        rel = st.session_state.get("feature_relationships", pd.DataFrame()).copy()

        st.markdown("### Allocation mix")
        final_num = pd.to_numeric(audit_df.get("final_alloc", pd.Series(0, index=audit_df.index)), errors="coerce").fillna(0)
        audit_df["final_alloc_numeric"] = final_num
        audit_df["allocated"] = final_num > 0
        mix = audit_df.groupby("flag", dropna=False).agg(
            rows=("flag", "size"),
            allocated_rows=("allocated", "sum"),
            total_alloc=("final_alloc_numeric", "sum"),
            avg_probability=("probability", "mean"),
        ).reset_index().sort_values("total_alloc", ascending=False)
        st.dataframe(mix, use_container_width=True, height=260)
        if not mix.empty:
            st.bar_chart(mix.set_index("flag")["total_alloc"])

        st.markdown("### Top allocated items")
        item_mix = audit_df.groupby("item", dropna=False).agg(
            rows=("item", "size"),
            allocated_rows=("allocated", "sum"),
            total_alloc=("final_alloc_numeric", "sum"),
            avg_probability=("probability", "mean"),
        ).reset_index().sort_values("total_alloc", ascending=False).head(25)
        st.dataframe(item_mix, use_container_width=True, height=320)
        if not item_mix.empty:
            st.bar_chart(item_mix.set_index("item")["total_alloc"])

        st.markdown("### Why rows were changed or blanked")
        reason_counts = (
            audit_df.get("reason", pd.Series("", index=audit_df.index))
            .astype(str)
            .str.split("; ")
            .explode()
            .replace("", np.nan)
            .dropna()
            .value_counts()
            .head(30)
            .rename_axis("reason")
            .reset_index(name="rows")
        )
        st.dataframe(reason_counts, use_container_width=True, height=320)
        if not reason_counts.empty:
            st.bar_chart(reason_counts.set_index("reason")["rows"])

        st.markdown("### Model-estimated feature usage")
        st.caption("Approximate model usage from first-layer neural network weights. This is model inspection, not causal proof.")
        if fi.empty:
            st.info("No model feature-importance information was available for this selected model.")
        else:
            fam = fi.groupby("feature_family", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Feature families")
                st.dataframe(fam, use_container_width=True, height=320)
                st.bar_chart(fam.set_index("feature_family")["importance"])
            with c2:
                st.subheader("Top base features")
                st.dataframe(fi.head(30), use_container_width=True, height=420)

        st.markdown("### Run-specific feature relationships")
        st.caption("Correlations between engineered numeric features and this run's predicted probability/final allocation.")
        if rel.empty:
            st.info("No run-specific feature relationship table was available.")
        else:
            st.dataframe(rel.head(40), use_container_width=True, height=420)
            top_rel = rel.head(20).set_index("feature")["relationship_strength"]
            if len(top_rel):
                st.bar_chart(top_rel)

with process_tab:
    st.markdown("## How Allocation AI works from start to finish")
    st.markdown(
        """
        **1. Historical training files**  
        Completed allocation files are read from `.xlsb`, `.xlsx`, or `.csv` sources. The trainer maps the allocation columns, preserves row order, and learns from historical **Final Alloc** values.

        **2. Feature engineering**  
        The model receives direct workbook signals like demand, supply, `Proj. Demand`, `Alloc. Rec.`, `Left DC`, `Demand Check`, `Helper`, flags, item, site, and UPC. It also receives expanded relationships such as item-level scarcity, row ranks within an item, cumulative demand before the row, store pressure, department/class pressure, velocity ratios, and partial leftover DC signals.

        **3. Neural training**  
        The Camp trainer builds a neural allocation model with integer FLM-unit targets, allocation probability, Review behavior, `Z - No Alloc.` override learning, scarcity learning, ordinal unit loss, focal allocation loss, OneCycle learning-rate scheduling, and start/stop/resume checkpoints.

        **4. Streamlit export**  
        The Jupyter trainer exports one app-compatible model bundle. This app loads that bundle as the included **Base NN Model**, and can also accept additional model artifact zips from future training runs.

        **5. Prediction**  
        The selected model predicts integer FLM units and allocation probability for each row. Predictions are then passed through the allocation simulator.

        **6. Allocation simulation**  
        The simulator preserves original row order, updates remaining `Left DC` by item, supports three-pass Review rows, allows justified `Z - No Alloc.` overrides, and allows leftover units below one FLM when that is all that remains.

        **7. Outputs**  
        The app returns a completed CSV, audit CSV, model feature-inspection CSV, run-specific feature relationship CSV, and prediction summary JSON.
        """
    )
    st.markdown("### Key behavior rules")
    behavior = pd.DataFrame([
        {"Behavior": "Integer output", "Description": "Final Alloc is an integer or blank."},
        {"Behavior": "No artificial Review FLM cap", "Description": "Review passes are not limited to 1 FLM per pass."},
        {"Behavior": "Partial leftover DC", "Description": "If Left DC is positive but below one FLM, the app can allocate the remaining units."},
        {"Behavior": "Three Review passes", "Description": "Review rows can be revisited up to three times after the main pass."},
        {"Behavior": "Z - No Alloc override", "Description": "No Alloc rows can receive allocation only when model/demand signals justify it."},
        {"Behavior": "Model selector", "Description": "Use the included Base NN Model or upload additional trained models."},
    ])
    st.dataframe(behavior, use_container_width=True, hide_index=True)

with model_tab:
    st.markdown("## Base NN Model metrics")
    st.caption("This page describes the included model that appears as **Base NN Model** in the selector.")
    if included_meta:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows total", f"{int(included_summary.get('rows_total') or 0):,}")
        c2.metric("Training rows", f"{int(included_summary.get('rows_train') or 0):,}" if included_summary.get("rows_train") else "—")
        c3.metric("Validation rows", f"{int(included_summary.get('rows_validation') or 0):,}" if included_summary.get("rows_validation") else "—")
        c4.metric("Positive rows", f"{int(included_summary.get('positive_rows_total') or 0):,}" if included_summary.get("positive_rows_total") else "—")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Best threshold", f"{included_summary.get('threshold', 0):.2f}")
        b2.metric("F1", _fmt_num(included_summary.get("f1")))
        b3.metric("Precision", _fmt_num(included_summary.get("precision")))
        b4.metric("Recall", _fmt_num(included_summary.get("recall")))
        b5.metric("Unit MAE", _fmt_num(included_summary.get("unit_mae"), 4))

        st.markdown("### Streamlit reload smoke test")
        st.caption("This is the validation sample used to confirm the exported app-compatible model reloads and predicts correctly.")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Smoke F1", _fmt_num(included_summary.get("smoke_test_f1")))
        s2.metric("Smoke precision", _fmt_num(included_summary.get("smoke_test_precision")))
        s3.metric("Smoke recall", _fmt_num(included_summary.get("smoke_test_recall")))
        s4.metric("Smoke unit accuracy", _fmt_num(included_summary.get("smoke_test_unit_accuracy")))

        with st.expander("Full Base NN Model metadata", expanded=False):
            st.json(included_meta)
    else:
        st.info("No Base NN Model metadata found.")

    sweep_path = Path("allocation_ai_threshold_sweep.csv")
    if sweep_path.exists():
        st.markdown("### Threshold sweep")
        try:
            sweep = pd.read_csv(sweep_path)
            st.dataframe(sweep, use_container_width=True, height=360)
            if "threshold" in sweep.columns and "f1" in sweep.columns:
                st.line_chart(sweep.set_index("threshold")[["f1", "precision", "recall"]])
        except Exception as exc:
            st.caption(f"Could not read threshold sweep: {exc}")

    if baseline_metrics:
        st.markdown("### Alloc. Rec. baseline")
        st.caption("Reference comparison against workbook allocation recommendation logic, when exported by the trainer.")
        st.json(baseline_metrics)

