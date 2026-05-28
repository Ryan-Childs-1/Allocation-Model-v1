# Allocation AI Predictor — Fixed New Model Version

This is the simplified prediction-only Streamlit app. It accepts allocation files and returns a completed CSV plus audit outputs.

## What this fixed

This version fixes crashes that happened when uploading the newer Jupyter-trained artifact zip.

The issue was caused by two things:

1. **scikit-learn pickle version mismatch** between the Jupyter training environment and Streamlit hosting.
2. **memory pressure during prediction** because the newer model uses a larger sklearn preprocessing matrix than the older base model.

The app now:

- Repairs known sklearn compatibility issues after loading uploaded model artifacts.
- Processes prediction in chunks to avoid dense-array memory crashes.
- Accepts `.zip`, `.joblib`, and `.pkl` model uploads.
- Uses the newer Model v2 app-compatible model as the included base model.
- Keeps outputs as CSV so row order is preserved and downloads are reliable.

## Files

All files are flat for GitHub upload:

```text
app.py
allocation_simulator.py
data_io.py
features.py
neural_model.py
predictor.py
schema.py
requirements.txt
runtime.txt
README.md
allocation_ai_base_sklearn_mlp.joblib
allocation_ai_metadata.json
```

## Deploy

Use Streamlit with `app.py` as the entry point. The included `runtime.txt` requests Python 3.11.

## Uploading newer trained artifacts

The sidebar accepts the full artifact zip from the Jupyter trainer. The app will search the zip for:

```text
allocation_ai_app_compatible_model.joblib
```

Raw `.pt` PyTorch checkpoints are training checkpoints and are not used directly by this prediction-only app.
