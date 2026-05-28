# Allocation AI Predictor — Updated Advanced Model

This is the simplified prediction-only Streamlit app for Allocation AI.

## What this package includes

- `app.py` — Streamlit entry point
- `allocation_ai_base_sklearn_mlp.joblib` — included app-compatible model updated from the latest Jupyter training artifact
- `allocation_ai_metadata.json` — validation metrics and recommended threshold
- `allocation_ai_threshold_sweep.csv` — threshold sweep for the app-compatible model
- `allocation_ai_torch_metadata.json` — reference metrics from the PyTorch training checkpoint
- `allocation_ai_torch_threshold_sweep.csv` — threshold sweep from the PyTorch model

## Updated model summary

The included base model was updated from `allocation_ai_jupyter_trained_artifacts.zip`.

Validation findings for the included app-compatible model:

- Best threshold: `0.85`
- Precision: `0.9395`
- Recall: `0.9176`
- F1: `0.9284`
- Unit accuracy: `0.9880`
- Positive unit accuracy: `0.7504`
- Unit MAE: `0.0141`
- False positive rate: `0.0024`

The app uses the exported app-compatible `.joblib` model for hosted prediction. The PyTorch checkpoint in the artifact had slightly stronger metrics, but this hosted prediction app intentionally avoids requiring PyTorch.

## Key behavior

- Upload `.xlsb`, `.xlsx`, or `.csv` allocation files.
- Preserves original row order in the output CSV.
- Overwrites only `Final Alloc.` in memory.
- Final Alloc outputs are integers or blanks.
- Review rows are handled with three intentional passes.
- There is no max-FLM-add-per-pass cap.
- Z - No Alloc rows can be overridden when model/demand signals justify it.
- The app can still accept a newer `.zip`, `.joblib`, or `.pkl` model artifact in the sidebar.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

This package is flat and designed for Streamlit hosting. `runtime.txt` pins Python 3.11.
