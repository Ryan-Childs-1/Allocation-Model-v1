# Allocation AI Predictor — Prediction-Only Streamlit App

This is the stripped-down Allocation AI app. It removes all training, dataset-builder, and model-lab pages. It is designed for hosted Streamlit prediction only.

## What it does

1. Upload a `.xlsb`, `.xlsx`, or `.csv` allocation file.
2. Optionally upload an updated Allocation AI model bundle (`.joblib` or `.pkl`).
3. Predict integer-only `Final Alloc.` values.
4. Preserve the original row order.
5. Return:
   - `completed_allocation.csv`
   - `allocation_audit.csv`
   - `prediction_summary.json` inside a zip

## Included model

The app includes:

```text
allocation_ai_base_sklearn_mlp.joblib
allocation_ai_metadata.json
```

The base model is a compressed sklearn neural-network prediction bundle. It is used automatically unless an updated model bundle is uploaded in the sidebar.

## Updated model bundle format

An uploaded model bundle must be a joblib/pickle dictionary with these keys:

```python
{
    "preprocessor": ...,      # fitted feature preprocessor
    "feature_columns": ...,   # list of model feature columns
    "unit_model": ...,        # predicts integer FLM-unit class
    "alloc_model": ...,       # predicts allocation probability
}
```

## Running on Streamlit

The app entry point is:

```text
app.py
```

Install dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Output is CSV, not `.xlsb`.
- `Final Alloc.` predictions are integers or blanks. No floats are written.
- `Left DC` is simulated sequentially by item.
- `Z - No Alloc.` rows can be allocated when the model and demand signals justify it, depending on sidebar settings.
- The app does not train or retrain models. Use a separate training environment to produce updated `.joblib` bundles, then upload them here.


## Uploading a trained artifact ZIP

This prediction-only app can now accept the full artifact ZIP exported by the Jupyter trainer. The ZIP may contain:

- PyTorch checkpoints such as `.pt` files
- Keras checkpoints such as `.keras` files
- training logs and threshold sweeps
- metadata JSON files
- the app-compatible model bundle

The app will automatically search the ZIP and load the best app-compatible `.joblib` or `.pkl` bundle, with priority given to files named like:

```text
allocation_ai_app_compatible_model.joblib
allocation_ai_prediction_model.joblib
allocation_ai_base_sklearn_mlp.joblib
```

The raw checkpoint files are intentionally ignored by this simplified app. It expects the app-compatible compressed prediction bundle produced by the trainer.
