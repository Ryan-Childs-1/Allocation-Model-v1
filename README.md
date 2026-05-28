# Allocation AI Predictor — Three-Pass Review Version

This is the simplified prediction-only Streamlit app. It accepts allocation files and returns a completed CSV plus audit outputs.

## Key behavior

- Upload `.xlsb`, `.xlsx`, or `.csv` allocation files.
- Optionally upload a newer trained artifact `.zip`, `.joblib`, or `.pkl` model bundle.
- Predict integer-only `Final Alloc.` values.
- Preserve the original row order in the completed CSV.
- Return blank Final Alloc cells when no allocation is recommended.
- Simulate item-level `Left DC` as allocations are made.
- Allow `Z - No Alloc.` rows to receive allocation only when model/demand signals justify it.
- Process `Review` rows in up to **three separate passes**.
- Do **not** cap Review additions by a maximum number of FLMs per pass.

## Three-pass Review logic

Review rows are intentionally revisited after the normal allocation pass:

1. **Review pass 1 — zero/blank scan**
   - Looks for Review rows still at zero/blank.
   - Can seed allocation when demand, Alloc. Rec., or model confidence supports it.

2. **Review pass 2 — add justified remaining amount**
   - Can add the full remaining model/demand-supported amount when the row still has need and remaining Left DC.
   - There is no artificial max-FLM-per-pass cap.

3. **Review pass 3 — final top-up**
   - Highest-confidence final pass.
   - Can add the full remaining justified amount when the model and/or Alloc. Rec. strongly support it.
   - There is no artificial max-FLM-per-pass cap.

Each pass sees the updated remaining item-level Left DC from previous allocations. Final outputs are still constrained by Left DC, demand protection, Alloc. Rec. influence mode, probability thresholds, and integer FLM rounding.

## Audit output

The audit CSV includes review-specific fields:

```text
review_passes_attempted
review_pass_1_added
review_pass_2_added
review_pass_3_added
review_total_added
allocated_on_pass
```

These fields show exactly which pass added inventory to a Review row.

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
