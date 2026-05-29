# Allocation AI Predictor

Prediction-only Streamlit app for Allocation AI.

## Entry point

Use `app.py` as the Streamlit entry point.

## Included model

The included default model appears in the app as **Base NN Model** and is stored as:

- `allocation_ai_base_sklearn_mlp.joblib`
- `allocation_ai_metadata.json`
- `allocation_ai_threshold_sweep.csv`

## Supported prediction files

- `.xlsb`
- `.xlsx`
- `.csv`

## Outputs

The app returns:

- `completed_allocation.csv`
- `allocation_audit.csv`
- `prediction_summary.json`
- `model_feature_importance.csv`
- `prediction_feature_relationships.csv`

## Notes

This version fixes a deployment ImportError by importing `predictor.py` safely as a module and surfacing any file mismatch inside the Streamlit UI. It also lazily imports `pyxlsb`, so the app itself can load even if `.xlsb` support has an environment problem.
