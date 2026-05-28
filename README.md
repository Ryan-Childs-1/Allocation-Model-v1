# Allocation AI Predictor — Artifact ZIP Compatible

Prediction-only Streamlit app for Allocation AI.

## What this version fixes

This version repairs scikit-learn pickle compatibility problems when uploading a Jupyter-trained artifact ZIP. In particular, it fixes errors like:

```text
AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'
```

That error happens when the Jupyter trainer serialized the app-compatible model under one scikit-learn version and Streamlit loads it under a newer version. The app now patches missing compatibility attributes after loading the uploaded model bundle.

## Run on Streamlit

Upload these flat files to GitHub and deploy `app.py`.

## Model upload

The sidebar accepts:

- `.zip` full artifact export from the Jupyter trainer
- `.joblib` direct app-compatible prediction bundle
- `.pkl` direct app-compatible prediction bundle

If a full artifact ZIP is uploaded, the app finds the app-compatible model inside it, such as:

```text
allocation_ai_app_compatible_model.joblib
```

It intentionally ignores raw PyTorch checkpoints such as `.pt` because the hosted prediction app uses the compressed app-compatible bundle.

## Output

The app returns:

- `completed_allocation.csv`
- `allocation_audit.csv`
- `prediction_summary.json`
- one combined output ZIP

Final Alloc values are integer-only or blank.
