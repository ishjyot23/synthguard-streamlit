# SynthGuard — Synthetic-Data Generation & Quality-Evaluation Agent

**Big Data Analytics mini-project** | Streamlit + Python | cloud deployment friendly

SynthGuard demonstrates a complete workflow for generating synthetic tabular data and checking its schema, direct-PII indicators, statistical fidelity, downstream model utility, privacy-risk signals, and a group-bias proxy.

## Run it without installing anything locally

1. Create or sign in to a GitHub account.
2. Create a repository, for example `synthguard-streamlit`.
3. Upload these three files from this folder to the repository root: `app.py`, `synthguard_core.py`, and `requirements.txt` (keep this `README.md` too).
4. Open [Streamlit Community Cloud](https://share.streamlit.io/) and connect your GitHub account.
5. Choose **Create app**. Select the repository and branch, set **Main file path** to `app.py`, then deploy.
6. Wait for the hosted app to build. The public dataset is fetched from a URL at runtime; no local Python setup or dataset download is needed.

Official deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## What the app does

1. Loads the UCI Student Performance Math dataset from its public CSV URL, or lets the user upload a CSV.
2. Scans for likely direct identifiers and masks flagged columns in the preview.
3. Excludes flagged columns by default before synthesis; the selected prediction target is retained.
4. Drops rows with a missing target and median/mode-imputes other missing values.
5. Splits real data into training and held-out evaluation subsets.
6. Fits a lightweight Gaussian-copula-style sampler only on the real training subset and generates synthetic rows.
7. Audits schema, types, ranges, categorical domains, and missing values.
8. Compares marginal distributions and numerical correlation structure.
9. Evaluates downstream prediction utility against a real-trained baseline on the same held-out real records.
10. Screens exact-row overlap and nearest mixed-type distances.
11. Compares the pass-rate gap between groups for a numeric outcome.
12. Combines configurable pilot guardrails into PASS / REVIEW / BLOCK gates and provides CSV, JSON, and ZIP exports.

## Suggested live demo

- Use the default UCI Student Performance dataset.
- Keep `G3` as the target.
- Keep `sex` as the group column.
- Leave default PII exclusions selected.
- Click **Run SynthGuard analysis**.
- Show the Overview, Schema & PII, Fidelity & Utility, and Privacy & Bias tabs.
- Download the ZIP report bundle.

## Important scientific and privacy limitations

- This is a teaching implementation, not a production-grade synthetic data product.
- The sampler uses a simplified latent normal encoding for categorical features; nominal category order is arbitrary. The fidelity measurements should be used to discuss limitations.
- PII scanning is heuristic and can miss sensitive data or flag innocent fields. No findings do not prove anonymity.
- Exact-row overlap and nearest-distance checks are only screening indicators. They do not provide differential privacy and do not rule out re-identification.
- The bias comparison examines one outcome-rate gap; it is not a complete fairness, causal, or legal assessment.
- The default thresholds are illustrative decision guardrails, not industry standards.
- A publicly deployed app is not an appropriate place to process identifiable or sensitive real records. Use public/sanitized demo data and upload only data you are authorised to process.
- The app does not intentionally save uploaded records to disk, but the hosting service controls its runtime and operational environment. Do not treat this as a secure data-processing environment.

## Dataset reference

UCI Machine Learning Repository — Student Performance: https://archive.ics.uci.edu/dataset/320/student+performance

The default app uses the `student-mat.csv` math-subject subset. Its `address` field represents urban/rural category, not a street address.
