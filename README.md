# Allocation AI Predictor

Prediction-only Streamlit app for allocation files.

## Included model

- **Base NN Model** — built-in Allocation AI model bundled with the app.

You can still upload additional trained `.zip`, `.joblib`, or `.pkl` model artifacts from the sidebar and select them in the model dropdown.

## Workflow

1. Upload a `.xlsb`, `.xlsx`, or `.csv` allocation file.
2. Select the built-in Base NN Model or upload/select another compatible model.
3. Run prediction.
4. Download:
   - `completed_allocation.csv`
   - `allocation_audit.csv`
   - `allocation_ai_prediction_output.zip`

## Features

- Multi-model selector for uploaded custom models
- Base NN Model included by default
- Three-pass Review-row allocation logic
- No artificial max-FLM-per-pass cap
- Allows partial leftover Left DC below one FLM
- Supports justified `Z - No Alloc.` overrides
- CSV output preserving row order
- Prediction insights and model metrics pages

## Deployment

Upload all files in this flat folder to GitHub / Streamlit Cloud. The app entry point is:

```bash
streamlit run app.py
```


## Built-in models

- Base NN Model: original included allocation model.
- Base Camp Model: latest production Camp MLP model trained through the live-progress sklearn path.

Additional `.zip`, `.joblib`, and `.pkl` model artifacts can still be uploaded through the app sidebar.
