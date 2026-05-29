# Allocation AI Predictor

Prediction-only Streamlit app for Allocation AI.

## What is included

- **Base NN Model**: the latest Camp app-compatible neural model (`allocation_ai_base_sklearn_mlp.joblib`).
- Multi-model selector: upload additional `.zip`, `.joblib`, or `.pkl` model artifacts and select which model to use.
- Prediction output as CSV, audit CSV, and ZIP.
- AI process/metadata page with training metrics, threshold sweep, and model/prediction insight charts.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Inputs

- `.xlsb`, `.xlsx`, or `.csv` allocation file.
- Default Excel sheet: `3.3 Working Table`.

## Outputs

- `completed_allocation.csv`
- `allocation_audit.csv`
- `prediction_summary.json`
- `model_feature_importance.csv`
- `prediction_feature_relationships.csv`

## Notes

The app preserves row order and writes Final Alloc in memory for a CSV download. It does not edit the source Excel workbook directly.
