# Allocation AI Predictor — Model v2 Bundled

This is the simplified prediction-only Streamlit app with your updated **Model v2** included as the default AI.

## What this app does

- Upload a `.xlsb`, `.xlsx`, or `.csv` allocation file.
- Predict integer-only `Final Alloc.` values.
- Preserve the original row order in the output CSV.
- Simulate `Left DC` sequentially by item.
- Allow `Z - No Alloc.` rows to receive allocation when the model and demand signals justify it.
- Download:
  - `completed_allocation.csv`
  - `allocation_audit.csv`
  - `prediction_summary.json`
  - combined output zip

## Included AI

The bundled default model is your Jupyter-trained **Model v2** app-compatible model, stored as:

```text
allocation_ai_base_sklearn_mlp.joblib
```

The app metadata is stored in:

```text
allocation_ai_metadata.json
```

The included metadata reports approximately:

- Rows total: 106,950
- Training rows: 73,803
- Validation rows: 33,147
- Positive rows: 5,502
- Best threshold: 0.95
- Validation F1: about 0.901
- Validation precision: about 0.894
- Validation recall: about 0.909
- Unit accuracy: about 0.986

## Uploading newer models

The sidebar still accepts:

```text
.zip
.joblib
.pkl
```

If you upload a Jupyter training artifact zip, the app automatically searches inside it for:

```text
allocation_ai_app_compatible_model.joblib
```

and uses that model for the session.

## Streamlit deployment

The app entry point is:

```text
app.py
```

Recommended install:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

This version intentionally does not include training inside Streamlit. Training should be done in the Jupyter Lab trainer, then exported as a training artifact zip or app-compatible joblib bundle.
