from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from allocation_simulator import AllocationConfig
from data_io import dataframe_to_csv_bytes, dataframe_to_parquet_bytes, read_allocation_file, save_upload
from dataset_store import DATASET_META_PATH, DATASET_PATH, DATASET_PKL_PATH, dataset_exists, build_dataset_from_uploads, load_dataset, save_dataset
from features import build_targets
from metrics import allocation_metrics, confusion_table, threshold_sweep, best_threshold_from_sweep
from predictor import predict_arrays, predict_to_csv_dataframe
from schema import ColumnDiagnostics, build_column_map
from training import train_neural_network

st.set_page_config(page_title="Allocation AI Advanced Neural Training Studio", page_icon="🧠", layout="wide")

MODEL_DIR = Path(".")
META_PATH = MODEL_DIR / "allocation_ai_metadata.json"


def read_meta() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def artifact_zip_bytes() -> bytes:
    names = [
        "allocation_ai_model.keras",
        "allocation_ai_best_checkpoint.keras",
        "allocation_ai_preprocessor.joblib",
        "allocation_ai_feature_columns.joblib",
        "allocation_ai_metadata.json",
        "training_log.csv",
        "training_progress_live.csv",
        "validation_threshold_sweep.csv",
        "allocation_training_dataset.parquet",
        "allocation_training_dataset_base.pkl",
        "allocation_ai_base_sklearn_mlp.joblib",
        "allocation_training_dataset_meta.csv",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            p = MODEL_DIR / name
            if p.exists():
                z.write(p, arcname=name)
    return buf.getvalue()


def install_uploaded_artifact(upload, filename: str):
    if upload is None:
        return
    Path(filename).write_bytes(upload.getbuffer())


st.title("🧠 Allocation AI Advanced Neural Training Studio")
st.caption("Hosted Streamlit app · base pre-trained neural model included · advanced Keras/Torch retraining · integer FLM-unit predictions · CSV output")

meta = read_meta()
if meta:
    st.success(
        f"Current model: {meta.get('rows_trained', 0):,} rows · "
        f"F1 {meta.get('validation_f1', 0):.3f} · precision {meta.get('validation_precision', 0):.3f} · "
        f"recall {meta.get('validation_recall', 0):.3f} · threshold {meta.get('recommended_threshold', 0.35):.2f}"
    )
else:
    st.info("No Keras model artifacts found yet. The included compressed base neural model will be used for predictions until you train a stronger Keras/Torch model.")

with st.sidebar:
    st.header("Prediction controls")
    use_saved_threshold = st.checkbox("Use saved tuned threshold", value=True)
    default_threshold = float(meta.get("recommended_threshold", 0.35)) if meta else 0.35
    manual_threshold = st.slider("Manual probability threshold", 0.01, 0.95, min(max(default_threshold, 0.01), 0.95), 0.01)
    demand_extra = st.slider("Demand cap extra FLM", 0.0, 6.0, 1.0, 0.25)
    allow_review = st.checkbox("Allow Review rows", value=True)
    prefer_left_dc = st.checkbox("Prefer Left DC over DC Avail", value=True)
    alloc_rec_influence = st.selectbox("Alloc. Rec. influence", ["feature_only", "soft_cap", "balanced", "hard_cap"], index=2)
    st.divider()
    st.header("Training defaults")
    epochs = st.number_input("Epochs", min_value=1, max_value=50000, value=500, step=50)
    batch_size = st.selectbox("Batch size", [128, 256, 512, 1024, 2048, 4096, 8192], index=3)
    max_units = st.number_input("Maximum FLM-unit class", min_value=5, max_value=500, value=100, step=5)
    learning_mode = st.selectbox("Architecture", ["fast", "balanced", "deep", "wide", "maximum"], index=1)
    learning_rate = st.select_slider("Learning rate", options=[0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005], value=0.001)
    validation_split = st.slider("Validation split", 0.05, 0.40, 0.18, 0.01)
    validation_method = st.selectbox("Validation method", ["holdout_by_file", "holdout_by_item", "random_rows"], index=0)
    threshold_mode = st.selectbox("Threshold tuning goal", ["balanced", "conservative", "aggressive"], index=0)

(t_predict, t_train, t_dataset, t_evaluate, t_audit, t_model, t_inspect, t_help) = st.tabs([
    "1. Predict Allocation",
    "2. Advanced Training Session",
    "3. Dataset Builder",
    "4. Evaluate Model",
    "5. Prediction Audit",
    "6. Model Manager",
    "7. Inspect File",
    "8. Workflow",
])

with t_predict:
    st.subheader("Predict Final Alloc and download a completed CSV")
    uploaded = st.file_uploader("Upload a new allocation workbook or CSV", type=["xlsb", "xlsx", "csv"], key="predict_upload")
    sheet = st.text_input("Sheet name", value="3.3 Working Table", key="predict_sheet")
    if uploaded and st.button("Run prediction", type="primary"):
        try:
            path = save_upload(uploaded, suffix=Path(uploaded.name).suffix)
            with st.spinner("Reading file while preserving row order..."):
                df = read_allocation_file(path, sheet_name=sheet)
            cfg = AllocationConfig(
                min_probability=manual_threshold,
                demand_cap_extra_flm=demand_extra,
                allow_review_rows=allow_review,
                alloc_rec_influence=alloc_rec_influence,
                prefer_left_dc=prefer_left_dc,
            )
            with st.spinner("Running neural prediction and sequential Left DC simulation..."):
                out_df, audit_df = predict_to_csv_dataframe(df, MODEL_DIR, cfg, use_saved_threshold=use_saved_threshold)
            st.session_state["last_output_df"] = out_df
            st.session_state["last_audit_df"] = audit_df
            final_num = pd.to_numeric(audit_df["final_alloc"], errors="coerce").fillna(0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(out_df):,}")
            c2.metric("Allocated rows", f"{int(final_num.gt(0).sum()):,}")
            c3.metric("Total Final Alloc", f"{int(final_num.sum()):,}")
            c4.metric("Mean probability", f"{audit_df['probability'].mean():.3f}")
            st.success("Completed. Final Alloc values are integers; no-allocation rows are blank.")
            base = Path(uploaded.name).stem
            st.download_button("Download completed allocation CSV", dataframe_to_csv_bytes(out_df), f"{base} - Allocation AI Output.csv", mime="text/csv")
            st.download_button("Download allocation audit CSV", dataframe_to_csv_bytes(audit_df), f"{base} - Allocation AI Audit.csv", mime="text/csv")
        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)

with t_train:
    st.subheader("Advanced multi-file neural training session")
    st.write("Upload many completed historical allocation files and train one Keras neural network. The app can train for many epochs and shows progress, metrics, and checkpoints.")

    source = st.radio("Training data source", ["Upload multiple files now", "Use cached dataset"], horizontal=True)
    session_uploads = None
    train_sheet = "3.3 Working Table"
    max_rows = 0
    if source == "Upload multiple files now":
        session_uploads = st.file_uploader("Training files", type=["xlsb", "xlsx", "csv"], accept_multiple_files=True, key="session_uploads")
        train_sheet = st.text_input("Training sheet", value="3.3 Working Table", key="session_sheet")
        max_rows = st.number_input("Max rows per file; 0 = all rows", min_value=0, value=0, step=1000, key="session_max_rows")
        save_after_build = st.checkbox("Save this session as cached dataset", value=True)
    else:
        save_after_build = False
        if dataset_exists():
            existing_name = DATASET_PATH if DATASET_PATH.exists() else DATASET_PKL_PATH
            st.success(f"Cached dataset found: `{existing_name}`")
        else:
            st.warning("No cached dataset found. Build one in Dataset Builder or choose file upload.")

    st.markdown("### Stop / target settings")
    colA, colB, colC, colD = st.columns(4)
    target_f1 = colA.number_input("Target F1; 0 = ignore", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    target_precision = colB.number_input("Target precision; 0 = ignore", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    target_recall = colC.number_input("Target recall; 0 = ignore", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    max_minutes = colD.number_input("Max training minutes; 0 = no time limit", min_value=0.0, max_value=1440.0, value=0.0, step=15.0)
    continue_training = st.checkbox("Continue from existing allocation_ai_model.keras", value=True)

    if st.button("Start advanced neural training", type="primary"):
        try:
            if source == "Use cached dataset":
                dataset = load_dataset()
            else:
                if not session_uploads:
                    raise ValueError("Upload at least one completed training file.")
                read_bar = st.progress(0, text="Starting multi-file dataset build...")
                dataset = build_dataset_from_uploads(
                    session_uploads,
                    sheet_name=train_sheet,
                    max_rows=None if max_rows == 0 else int(max_rows),
                    max_units=int(max_units),
                    progress=read_bar,
                )
                if save_after_build:
                    save_dataset(dataset)
            st.success(f"Training dataset ready: {len(dataset):,} rows · {int((dataset['__target_units'] > 0).sum()):,} positive allocation rows")
            with st.expander("Dataset by source file", expanded=True):
                if "__source_file" in dataset.columns:
                    st.dataframe(dataset.groupby("__source_file").agg(rows=("__target_units", "size"), positive_rows=("__target_units", lambda s: int((s > 0).sum())), max_units=("__target_units", "max")).reset_index(), use_container_width=True)

            progress = st.progress(0, text="Preparing model...")
            metric_box = st.empty()
            chart = st.empty()
            status = st.container()
            log_table = st.empty()
            with st.spinner("Training neural network..."):
                trained_meta = train_neural_network(
                    dataset=dataset,
                    model_dir=MODEL_DIR,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_mode=learning_mode,
                    max_units=int(max_units),
                    continue_training=continue_training,
                    learning_rate=float(learning_rate),
                    validation_split=float(validation_split),
                    validation_method=validation_method,
                    threshold_mode=threshold_mode,
                    target_f1=float(target_f1),
                    target_precision=float(target_precision),
                    target_recall=float(target_recall),
                    max_minutes=float(max_minutes),
                    progress=progress,
                    chart=chart,
                    status=status,
                    metric_box=metric_box,
                    log_table=log_table,
                )
            st.success("Training complete. Model artifacts saved in the app folder.")
            st.json(trained_meta)
            st.download_button("Download all model/training artifacts", artifact_zip_bytes(), "allocation_ai_training_artifacts.zip", mime="application/zip")
        except Exception as exc:
            st.error("Training failed.")
            st.exception(exc)

with t_dataset:
    st.subheader("Dataset Builder")
    st.write("Build a clean reusable Parquet dataset from multiple workbooks. This makes long training sessions faster because the app does not need to reread large Excel files each time.")
    uploads = st.file_uploader("Historical completed files", type=["xlsb", "xlsx", "csv"], accept_multiple_files=True, key="dataset_uploads")
    ds_sheet = st.text_input("Dataset sheet", value="3.3 Working Table", key="dataset_sheet")
    ds_max_rows = st.number_input("Dataset max rows per file; 0 = all rows", min_value=0, value=0, step=1000, key="ds_max_rows")
    if uploads and st.button("Build cached dataset", type="primary"):
        try:
            bar = st.progress(0, text="Reading files...")
            ds = build_dataset_from_uploads(uploads, sheet_name=ds_sheet, max_rows=None if ds_max_rows == 0 else int(ds_max_rows), max_units=int(max_units), progress=bar)
            save_dataset(ds)
            st.success(f"Cached dataset saved: {len(ds):,} rows · {int((ds['__target_units'] > 0).sum()):,} positive allocation rows")
            st.dataframe(ds.head(200), use_container_width=True)
            if DATASET_PATH.exists():
                st.download_button("Download cached dataset parquet", DATASET_PATH.read_bytes(), DATASET_PATH.name)
            if DATASET_PKL_PATH.exists():
                st.download_button("Download cached dataset pickle", DATASET_PKL_PATH.read_bytes(), DATASET_PKL_PATH.name)
            if DATASET_META_PATH.exists():
                st.download_button("Download cached dataset metadata CSV", DATASET_META_PATH.read_bytes(), DATASET_META_PATH.name, mime="text/csv")
        except Exception as exc:
            st.error("Dataset build failed.")
            st.exception(exc)
    if dataset_exists():
        ds = load_dataset()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cached rows", f"{len(ds):,}")
        c2.metric("Positive rows", f"{int((ds['__target_units'] > 0).sum()):,}")
        c3.metric("Positive rate", f"{(ds['__target_units'] > 0).mean():.1%}")
        c4.metric("Source files", f"{ds['__source_file'].nunique() if '__source_file' in ds else 1:,}")

with t_evaluate:
    st.subheader("Evaluate model on labeled allocation file")
    test_file = st.file_uploader("Labeled test file", type=["xlsb", "xlsx", "csv"], key="eval_upload")
    eval_sheet = st.text_input("Evaluation sheet", value="3.3 Working Table", key="eval_sheet")
    if test_file and st.button("Evaluate model", type="primary"):
        try:
            path = save_upload(test_file, suffix=Path(test_file.name).suffix)
            df = read_allocation_file(path, sheet_name=eval_sheet)
            y = build_targets(df, max_units=int(max_units))
            units, prob, _ = predict_arrays(df, MODEL_DIR)
            sweep = threshold_sweep(y["__target_alloc_binary"].values, prob, y["__target_units"].values, units)
            best_t = best_threshold_from_sweep(sweep, mode=threshold_mode)
            m = allocation_metrics(y["__target_alloc_binary"].values, prob, y["__target_units"].values, units, threshold=best_t)
            st.success(f"Best threshold: {best_t:.2f}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Precision", f"{m['precision']:.3f}")
            c2.metric("Recall", f"{m['recall']:.3f}")
            c3.metric("F1", f"{m['f1']:.3f}")
            c4.metric("Unit accuracy", f"{m['exact_unit_accuracy']:.3f}")
            c5.metric("Unit MAE", f"{m['unit_mae']:.3f}")
            st.line_chart(sweep.set_index("threshold")[["precision", "recall", "f1", "exact_unit_accuracy"]])
            st.dataframe(confusion_table(y["__target_alloc_binary"].values, prob, best_t), use_container_width=True)
            rows = pd.DataFrame({"actual_units": y["__target_units"], "actual_alloc": y["__target_alloc_binary"], "predicted_units_raw": units, "probability": prob})
            rows["predicted_units_thresholded"] = rows["predicted_units_raw"].where(rows["probability"] >= best_t, 0)
            st.download_button("Download threshold sweep CSV", dataframe_to_csv_bytes(sweep), "validation_threshold_sweep.csv", mime="text/csv")
            st.download_button("Download row-level evaluation CSV", dataframe_to_csv_bytes(rows), "row_level_evaluation.csv", mime="text/csv")
        except Exception as exc:
            st.error("Evaluation failed.")
            st.exception(exc)

with t_audit:
    st.subheader("Last prediction audit")
    audit = st.session_state.get("last_audit_df")
    if audit is None:
        st.info("Run a prediction first to populate the audit table.")
    else:
        st.dataframe(audit, use_container_width=True)
        st.download_button("Download last audit CSV", dataframe_to_csv_bytes(audit), "allocation_ai_audit.csv", mime="text/csv")

with t_model:
    st.subheader("Model Manager")
    st.write("Upload, download, back up, or replace model artifacts. Keep these files together: `.keras`, preprocessor, feature columns, and metadata.")
    c1, c2 = st.columns(2)
    with c1:
        up_model = st.file_uploader("Upload allocation_ai_model.keras", type=["keras"], key="model_upload")
        up_pre = st.file_uploader("Upload allocation_ai_preprocessor.joblib", type=["joblib"], key="pre_upload")
        up_cols = st.file_uploader("Upload allocation_ai_feature_columns.joblib", type=["joblib"], key="cols_upload")
        up_meta = st.file_uploader("Upload allocation_ai_metadata.json", type=["json"], key="meta_upload")
        if st.button("Install uploaded artifacts"):
            install_uploaded_artifact(up_model, "allocation_ai_model.keras")
            install_uploaded_artifact(up_pre, "allocation_ai_preprocessor.joblib")
            install_uploaded_artifact(up_cols, "allocation_ai_feature_columns.joblib")
            install_uploaded_artifact(up_meta, "allocation_ai_metadata.json")
            st.success("Installed uploaded artifacts.")
    with c2:
        st.download_button("Download all current artifacts", artifact_zip_bytes(), "allocation_ai_artifacts.zip", mime="application/zip")
        if Path("allocation_ai_base_sklearn_mlp.joblib").exists():
            st.success("Included base neural model found: allocation_ai_base_sklearn_mlp.joblib")
        if meta:
            st.json(meta)
        if st.button("Clear model artifacts", type="secondary"):
            for name in ["allocation_ai_model.keras", "allocation_ai_best_checkpoint.keras", "allocation_ai_preprocessor.joblib", "allocation_ai_feature_columns.joblib", "allocation_ai_metadata.json"]:
                p = Path(name)
                if p.exists():
                    p.unlink()
            st.warning("Model artifacts cleared.")

with t_inspect:
    st.subheader("Inspect file mapping")
    inspect_file = st.file_uploader("Upload a file to inspect", type=["xlsb", "xlsx", "csv"], key="inspect_upload")
    inspect_sheet = st.text_input("Inspect sheet", value="3.3 Working Table", key="inspect_sheet")
    if inspect_file and st.button("Inspect columns"):
        try:
            path = save_upload(inspect_file, suffix=Path(inspect_file.name).suffix)
            df = read_allocation_file(path, sheet_name=inspect_sheet)
            cmap = build_column_map(df)
            st.write(f"Rows: **{len(df):,}** · Columns: **{len(df.columns):,}**")
            st.dataframe(pd.DataFrame(ColumnDiagnostics(len(df), len(df.columns), cmap).as_rows()), use_container_width=True)
            st.dataframe(df.head(50), use_container_width=True)
        except Exception as exc:
            st.error("Inspection failed.")
            st.exception(exc)

with t_help:
    st.subheader("Recommended workflow")
    st.markdown(
        """
        1. **Build a cached dataset** from many completed historical allocation files.
        2. **Train the neural network** using `holdout_by_file` validation so the score reflects new-file performance.
        3. Use **Evaluate Model** on a recent completed file that was not part of training.
        4. Use **Predict Allocation** on a new file and download the completed CSV.
        5. When you complete more allocation files manually, add them to the next training session and continue training.

        The model predicts **integer FLM units**, then converts them into integer Final Alloc values. Blank/no-allocation rows remain blank.
        """
    )
