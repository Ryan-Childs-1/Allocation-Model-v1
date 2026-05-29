# Allocation AI Streamlit Predictor

This is a flat, Streamlit-ready prediction app for Allocation AI.

## Built-in models

The AI selector includes two built-in models:

1. **Base NN Model** — the prior included neural/allocation model.
2. **Base Transfer Model** — the sequential transfer-trained Streamlit-compatible MLP model exported as `transfer_model`.

You can also upload additional `.zip`, `.joblib`, or `.pkl` model artifacts and choose between all available models in the sidebar.

## Inputs

The app accepts allocation files in:

- `.xlsb`
- `.xlsx`
- `.csv`

It preserves row order and writes predictions into the detected Final Alloc column in the downloadable CSV output.

## Outputs

After prediction, the app provides:

- `completed_allocation.csv`
- `allocation_audit.csv`
- `prediction_summary.json`
- `model_feature_importance.csv`
- `prediction_feature_relationships.csv`
- a combined output ZIP

## Model behavior

The simulator supports:

- integer Final Alloc output or blank
- three-pass Review logic
- no max-FLM-per-pass cap
- `Z - No Alloc.` override when justified
- partial remaining Left DC below one FLM
- sequential Left DC updates by item

## Deploying to Streamlit

Upload all files in this flat folder to GitHub and deploy `app.py` as the Streamlit entry point.


## File-size note

The built-in **Base Transfer Model** is stored as split files:

```text
allocation_ai_base_transfer_model.joblib.part01
allocation_ai_base_transfer_model.joblib.part02
allocation_ai_base_transfer_model.joblib.part03
```

This keeps every repository file below 25 MB for GitHub/Streamlit upload limits. The app reconstructs the model in memory automatically at runtime. Do not rename or delete these `.partXX` files.
