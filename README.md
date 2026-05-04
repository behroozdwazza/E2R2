# E2R2 Data Analysis App

This local app recreates the E2R2 workflow from the manuscript:

1. Upload a raw tabular dataset and optional data dictionary.
2. Define the binary prediction goal, target column, positive outcome, and columns to ignore.
3. Train several off-the-shelf baseline classifiers and keep the best performer.
4. Build a case base from holdout predictions that were both correct and above the selected confidence threshold.
5. Attach top feature-level attribution scores and data-dictionary context to each precedent case.
6. Retrieve the nearest high-confidence cases for a holdout record using cosine similarity over normalized model inputs.
7. Prepare a context-rich LLM prompt and produce a local E2R2 prediction rationale.

## Run

From this folder:

```powershell
.\run_e2r2_app.ps1
```

The app opens at a local browser address such as `http://127.0.0.1:8765`.

## Streamlit Deployment

The deployable Streamlit entrypoint is:

```text
streamlit_app.py
```

It contains both the full E2R2 pipeline and the LLM Experiment Lab in one web interface. To run locally with Streamlit:

```powershell
.\run_streamlit_app.ps1
```

If Streamlit is not installed locally yet, install the deployable requirements first:

```powershell
C:\Users\davazdab\.conda\envs\knime_python\python.exe -m pip install -r requirements.txt
```

To deploy on Streamlit Community Cloud:

1. Push this folder to a GitHub repository.
2. Create a new Streamlit app from that repository.
3. Set the main file path to `streamlit_app.py`.
4. Make sure `requirements.txt` is included.
5. Add any shared OpenAI API key as a Streamlit secret only if you want server-managed credentials; otherwise users can paste their own key in the app.

For student or sensitive data, use a private deployment option approved by your institution. Avoid public demos with identifiable records.

## LLM Experiment Lab

After you export `e2r2_case_base.csv`, `e2r2_shap_scores.csv`, and `e2r2_holdout_dataset.csv`, you can test LLM predictions without rerunning model training or SHAP:

```powershell
.\run_llm_experiment_app.ps1
```

This opens a separate local app, usually at `http://127.0.0.1:8775`. Upload the three exported files, save your API key once for the local session, then run individual or batch LLM predictions. The lab reconstructs retrieval from the exported holdout features and case-base row indexes; it does not retrain the baseline model or recompute SHAP.

For retrieval that exactly matches the main app, also download and upload `e2r2_case_vectors.csv` and `e2r2_holdout_vectors.csv`. Those machine-readable vector files contain the normalized model-preprocessed vectors used by the main E2R2 cosine retrieval. If the vector files are not uploaded, the lab falls back to reconstructed retrieval from exported holdout features.

The script uses the stable local KNIME Python environment when it is available. If you prefer a separate environment, install `requirements.txt` into that environment and run:

```powershell
python app.py
```

## Install True SHAP

Actual SHAP mode requires the `shap` package in the same Python environment used by the app. Install it once with:

```powershell
.\install_shap.ps1
```

or directly:

```powershell
C:\Users\davazdab\.conda\envs\knime_python\python.exe -m pip install shap
```

## Data Dictionary Format

The app accepts CSV or Excel dictionaries. It looks for a feature-name column such as `variable`, `feature`, `field`, `column`, `column_name`, or `name`, and then combines available definition fields such as `definition`, `description`, `meaning`, `label`, `notes`, or `values`.

## LLM Use

The app always produces a local prediction and reasoning paragraph. If you enter an OpenAI API key and model name in the holdout section, it also sends the prepared E2R2 prompt to the OpenAI Responses API and displays the returned narrative.

The model selector includes current OpenAI text models such as GPT-5.1, GPT-5, GPT-5 mini, GPT-5 nano, GPT-5 pro, GPT-4.1, GPT-4.1 mini, and GPT-4.1 nano.

## Attribution Note

The default attribution method is now `Actual SHAP in probability units`. The app uses Tree SHAP in probability mode for supported tree ensembles and Kernel SHAP against `predict_proba` for other estimators. This avoids raw log-odds SHAP values, which can legitimately be greater than 1 but are hard to interpret in the E2R2 case table. The optional fast approximation remains available in the UI for quick testing; it replaces each feature with a training baseline and measures the change in predicted probability.

SHAP runtime depends on the selected baseline model and the number of case-base records, not simply the full training-set size. The app computes SHAP for the correctly predicted high-confidence cases that enter the E2R2 case base and uses a sampled background set for probability-space explanations. Tree SHAP can finish quickly even for thousands of records; Kernel SHAP is the slower path. The UI exposes `SHAP background cases` and `Kernel SHAP samples`; increasing either value makes probability-space Kernel SHAP more rigorous and slower.
