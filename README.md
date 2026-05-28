# Allocation AI Advanced Neural Training Studio

Hosted Streamlit app for allocation training and prediction.

## What this version is designed to do

- Run through Streamlit on the web.
- Accept multiple `.xlsb`, `.xlsx`, or `.csv` historical allocation files in one training session.
- Train a single advanced Keras neural network using the PyTorch backend.
- Include a compressed base neural model so prediction works immediately after deployment.
- Predict integer FLM-unit classes, then convert them into integer `Final Alloc.` values.
- Preserve row order and download the completed result as CSV.
- Return blanks for no-allocation rows.
- Use `Left DC`, `Proj. Demand`, `Alloc. Rec.`, `Demand Check`, and `Helper` in feature engineering and allocation simulation.

## Included base model

This package includes:

- `allocation_ai_base_sklearn_mlp.joblib`
- `allocation_ai_metadata.json`
- `allocation_training_dataset_base.pkl`

The base neural model is a compressed sklearn MLP neural network trained from the provided allocation files that could be processed in this environment. It is intended as a starter model so the hosted app can make predictions immediately. The **Advanced Training Session** tab should be used to train a stronger `.keras` model for many epochs or hours.

## Streamlit Cloud / hosted setup

Upload all files in this flat folder to a GitHub repository. Then deploy the repo in Streamlit.

Use `app.py` as the entry point.

`runtime.txt` pins Python 3.11 to improve Torch/Keras wheel compatibility.

## Recommended workflow

1. Deploy to Streamlit.
2. Open the app.
3. Use **Predict Allocation** to test the included base model.
4. Use **Dataset Builder** to upload many completed allocation files and build a cached dataset.
5. Use **Advanced Training Session** to train the Keras/Torch neural network.
6. Use **Evaluate Model** on a held-out completed file.
7. Use **Predict Allocation** to download a completed CSV with integer Final Alloc values.

## Main files

```text
app.py
allocation_simulator.py
data_io.py
dataset_store.py
features.py
metrics.py
neural_model.py
predictor.py
schema.py
training.py
requirements.txt
runtime.txt
allocation_ai_base_sklearn_mlp.joblib
allocation_ai_metadata.json
allocation_training_dataset_base.pkl
```

No `run_app.bat` is included because this version is designed for hosted Streamlit deployment.

## Included continued-training model

This package includes an updated hosted-compatible base neural MLP artifact:

- `allocation_ai_base_sklearn_mlp.joblib`
- `allocation_ai_metadata.json`
- `continued_training_progress.csv`
- `continued_training_result.json`

The base model was continued on the cached multi-file allocation dataset and the app will use it automatically until a `.keras` model is trained in the Advanced Training Session tab.

## Z - No Alloc. override behavior

This version no longer treats `Z - No Alloc.` rows as permanently blocked.

During prediction, the simulator can allocate these rows when either:

- the model predicts a high enough allocation probability and a positive FLM-unit class, or
- the row has enough demand/need and `Alloc. Rec.` support to justify a conservative override.

The sidebar exposes these controls:

- **Allow Z - No Alloc. rows when model/demand justify it**
- **Z - No Alloc override probability**
- **Z - No Alloc minimum need / Alloc. Rec. units**

The audit CSV includes:

- `is_z_no_alloc`
- `z_no_alloc_override`
- `need_units`
- `alloc_rec_units`
- `reason`

Final outputs remain integer FLM multiples or blank.
