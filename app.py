from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from allocation_simulator import AllocationConfig
from data_io import dataframe_to_csv_bytes, read_allocation_file, save_upload
from predictor import load_model_bundle, predict_to_outputs, read_metadata
from schema import ColumnDiagnostics, build_column_map

st.set_page_config(page_title="Allocation AI Predictor", page_icon="🎯", layout="wide")

st.title("🎯 Allocation AI Predictor")
st.caption("Prediction-only app · upload allocation file · optionally upload updated model artifacts zip · download completed CSV")

meta = read_metadata()
with st.sidebar:
    st.header("Model")
    uploaded_model = st.file_uploader(
        "Optional updated model bundle or training artifact ZIP",
        type=["zip", "joblib", "pkl"],
        help=(
            "Upload either a direct .joblib/.pkl prediction bundle or the full artifact .zip "
            "created by the Jupyter trainer. The app will find the app-compatible model inside the zip."
        ),
    )
    if uploaded_model:
        st.success(f"Using uploaded model/artifacts: {uploaded_model.name}")
    else:
        st.info("Using included base model")

    st.header("Prediction settings")
    default_threshold = float(meta.get("recommended_threshold", meta.get("best_threshold", 0.35))) if meta else 0.35
    min_probability = st.slider("Minimum allocation probability", 0.01, 0.95, min(max(default_threshold, 0.01), 0.95), 0.01)
    demand_extra = st.slider("Demand cap extra FLM", 0.0, 6.0, 1.0, 0.25)
    allow_review = st.checkbox("Allow Review rows", value=True)
    review_passes = st.slider(
        "Review-row passes", 1, 3, 3, 1,
        help="Review rows are revisited up to three times. Pass 1 scans zero/blank Review rows; passes 2 and 3 can add incremental allocations if justified.",
    )
    review_pass1_prob = st.slider("Review pass 1 zero-scan probability", 0.10, 0.95, 0.55, 0.01)
    review_pass2_prob = st.slider("Review pass 2 add-more probability", 0.10, 0.95, 0.70, 0.01)
    review_pass3_prob = st.slider("Review pass 3 final top-up probability", 0.10, 0.95, 0.85, 0.01)
    st.caption("Review passes are not capped by a max FLM-per-pass setting. Each pass may add the full justified amount, limited only by Left DC, demand protection, Alloc. Rec. influence, and integer FLM rounding.")
    allow_no_alloc = st.checkbox("Allow Z - No Alloc. rows if justified", value=True)
    no_alloc_prob = st.slider("Z - No Alloc override probability", 0.10, 0.95, 0.65, 0.01)
    no_alloc_need = st.slider("Z - No Alloc minimum need / Alloc. Rec. units", 0.0, 10.0, 1.0, 0.5)
    prefer_left_dc = st.checkbox("Prefer Left DC over DC Avail", value=True)
    alloc_rec_influence = st.selectbox(
        "Alloc. Rec. influence",
        ["feature_only", "soft_cap", "balanced", "hard_cap"],
        index=2,
        help="Controls whether Alloc. Rec. is only a feature or also caps the final allocation.",
    )

if meta:
    best = meta.get("best_validation_metrics", {}) if isinstance(meta.get("best_validation_metrics", {}), dict) else {}
    st.markdown("### Included model summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows trained", f"{int(meta.get('rows_total', meta.get('rows_trained', meta.get('base_model_rows_trained', 0)))):,}")
    c2.metric("Best threshold", f"{float(meta.get('best_threshold', meta.get('recommended_threshold', 0))):.2f}")
    c3.metric("F1", f"{float(meta.get('validation_f1', best.get('f1', 0))):.3f}")
    c4.metric("Precision", f"{float(meta.get('validation_precision', best.get('precision', 0))):.3f}")
    c5.metric("Recall", f"{float(meta.get('validation_recall', best.get('recall', 0))):.3f}")

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Unit accuracy", f"{float(best.get('unit_accuracy', 0)):.3f}")
    c7.metric("Positive unit accuracy", f"{float(best.get('positive_unit_accuracy', 0)):.3f}")
    c8.metric("Unit MAE", f"{float(best.get('unit_mae', 0)):.4f}")
    c9.metric("False positive rate", f"{float(best.get('false_positive_rate', 0)):.4f}")

    sweep_path = Path("allocation_ai_threshold_sweep.csv")
    if sweep_path.exists():
        with st.expander("Included model threshold sweep", expanded=False):
            try:
                st.dataframe(pd.read_csv(sweep_path), use_container_width=True, height=300)
            except Exception as exc:
                st.caption(f"Could not read threshold sweep: {exc}")

    torch_meta_path = Path("allocation_ai_torch_metadata.json")
    if torch_meta_path.exists():
        with st.expander("Training findings from PyTorch model", expanded=False):
            try:
                torch_meta = json.loads(torch_meta_path.read_text(encoding="utf-8"))
                torch_best = torch_meta.get("best_validation_metrics", {})
                st.write(
                    "The uploaded trainer also produced a PyTorch checkpoint. "
                    "This hosted prediction app uses the exported app-compatible model, "
                    "but the PyTorch training metrics are shown here for comparison."
                )
                st.json({
                    "torch_backend": torch_meta.get("backend"),
                    "torch_best_threshold": torch_meta.get("best_threshold"),
                    "torch_precision": torch_best.get("precision"),
                    "torch_recall": torch_best.get("recall"),
                    "torch_f1": torch_best.get("f1"),
                    "torch_unit_accuracy": torch_best.get("unit_accuracy"),
                    "torch_positive_unit_accuracy": torch_best.get("positive_unit_accuracy"),
                    "torch_unit_mae": torch_best.get("unit_mae"),
                })
            except Exception as exc:
                st.caption(f"Could not read PyTorch metadata: {exc}")

st.markdown("### 1. Upload allocation file")
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

        st.markdown("### 2. Run prediction")
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
            with st.spinner("Loading model/artifacts and running sequential allocation simulation..."):
                bundle = load_model_bundle(uploaded_model)
                out_df, audit_df, summary = predict_to_outputs(df, bundle, cfg)

            artifact_meta = bundle.get("__artifact_metadata", {}) if isinstance(bundle, dict) else {}
            if artifact_meta:
                summary["artifact_metadata"] = artifact_meta
            if isinstance(bundle, dict):
                summary["model_source"] = bundle.get("__artifact_model_name", "uploaded/included model")
                if bundle.get("__compat_repairs"):
                    summary["compatibility_repairs"] = bundle.get("__compat_repairs")

            st.session_state["out_df"] = out_df
            st.session_state["audit_df"] = audit_df
            st.session_state["summary"] = summary
            st.success("Prediction complete. Final Alloc values are integers or blank.")

    except Exception as exc:
        st.error("File loading or prediction setup failed.")
        st.exception(exc)

if "out_df" in st.session_state:
    out_df = st.session_state["out_df"]
    audit_df = st.session_state["audit_df"]
    summary = st.session_state["summary"]

    st.markdown("### 3. Prediction summary")
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
        st.info(f"Model source used: `{summary['model_source']}`")
    if summary.get("compatibility_repairs"):
        st.warning("Applied sklearn compatibility repairs to uploaded model bundle: " + ", ".join(summary["compatibility_repairs"]))
    if isinstance(summary.get("artifact_metadata"), dict) and summary["artifact_metadata"]:
        with st.expander("Uploaded artifact metadata", expanded=False):
            st.json(summary["artifact_metadata"])

    st.markdown("### 4. Preview")
    preview_cols = [c for c in ["Item", "Site", "Flag", "Final Alloc.", "Left DC", "Dc Avail", "Proj. Demand", "Alloc. Rec."] if c in out_df.columns]
    st.dataframe(out_df[preview_cols].head(200) if preview_cols else out_df.head(200), use_container_width=True, height=360)

    with st.expander("Audit preview", expanded=False):
        st.dataframe(audit_df.head(500), use_container_width=True, height=420)

    completed_csv = dataframe_to_csv_bytes(out_df)
    audit_csv = dataframe_to_csv_bytes(audit_df)
    summary_json = json.dumps(summary, indent=2).encode("utf-8")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("completed_allocation.csv", completed_csv)
        z.writestr("allocation_audit.csv", audit_csv)
        z.writestr("prediction_summary.json", summary_json)

    st.markdown("### 5. Download")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download completed CSV", completed_csv, "completed_allocation.csv", mime="text/csv")
    d2.download_button("Download audit CSV", audit_csv, "allocation_audit.csv", mime="text/csv")
    d3.download_button("Download output ZIP", zip_buf.getvalue(), "allocation_ai_prediction_output.zip", mime="application/zip")

st.divider()
st.caption("Prediction-only app. Review passes have no artificial max-FLM add cap; outputs are still limited by Left DC, demand protection, Alloc. Rec. influence, and integer FLM rounding. To update predictions, upload a newer .joblib/.pkl model bundle or a full training artifact .zip in the sidebar.")
