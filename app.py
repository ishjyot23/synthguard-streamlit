from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from synthguard_core import (
    GaussianCopulaTabularSampler,
    bias_comparison,
    build_decision,
    correlation_drift,
    evaluate_privacy,
    evaluate_utility,
    json_safe,
    marginal_fidelity,
    normalize_dataframe,
    prepare_analysis_frame,
    scan_pii,
    schema_audit,
)

UCI_MATH_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
UCI_MATH_FALLBACK = "https://huggingface.co/datasets/DElmazi/Student_Performance/resolve/main/student-mat.csv"

st.set_page_config(page_title="SynthGuard | Synthetic Data Quality Agent", page_icon="S", layout="wide")

st.markdown("""
<style>
:root { --sg-accent: #4f46e5; }
.block-container { padding-top: 1.7rem; padding-bottom: 2.6rem; }
.sg-kicker { text-transform: uppercase; letter-spacing: .14em; font-size: .75rem; font-weight: 700; color: #6366f1; }
.sg-subtitle { color: #64748b; font-size: 1.02rem; margin-top: -.3rem; }
.sg-card { border: 1px solid rgba(120,130,150,.28); border-radius: 14px; padding: 16px 18px; height: 100%; }
.sg-big { font-size: 1.8rem; font-weight: 750; line-height: 1.2; }
.sg-small { color: #64748b; font-size: .84rem; margin-top: .25rem; }
[data-testid="stMetric"] { border: 1px solid rgba(120,130,150,.25); padding: 12px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_uci_math() -> tuple[pd.DataFrame, str]:
    errors = []
    for url in [UCI_MATH_URL, UCI_MATH_FALLBACK]:
        try:
            frame = pd.read_csv(url, sep=";")
            if frame.shape[1] < 10:
                frame = pd.read_csv(url)
            if len(frame) < 100 or frame.shape[1] < 10:
                raise ValueError("The response did not look like the Student Performance table.")
            return normalize_dataframe(frame), url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not load the public dataset. " + " | ".join(errors))


def source_signature(frame: pd.DataFrame) -> str:
    try:
        hashed = pd.util.hash_pandas_object(frame.astype(str), index=True).to_numpy().tobytes()
        return hashlib.sha256(hashed).hexdigest()[:12]
    except Exception:
        return hashlib.sha256(f"{frame.shape}-{list(frame.columns)}".encode()).hexdigest()[:12]


def masked_preview(frame: pd.DataFrame, pii_report: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    preview = frame.head(n).copy()
    if not pii_report.empty:
        for col in pii_report["column"].tolist():
            if col in preview.columns:
                preview[col] = "[masked by SynthGuard]"
    return preview


def clean_report_markdown(result: dict) -> str:
    util = result["utility"]
    privacy = result["privacy"]
    bias = result["bias"]
    decision = result["overall_decision"]
    fidelity_mean = result["fidelity_mean"]
    lines = [
        "# SynthGuard Quality Evaluation Report", "",
        f"- Generated at (UTC): {result['generated_at_utc']}",
        f"- Dataset: {result['dataset_name']}",
        f"- Prepared real rows: {result['prepared_rows']}",
        f"- Synthetic rows: {result['synthetic_rows']}",
        f"- Target: `{result['target']}`",
        f"- Excluded columns: {', '.join(result['excluded_columns']) if result['excluded_columns'] else 'None'}", "",
        f"## Agent recommendation: {decision}", "",
        "## Core results", "",
        f"- Schema checks passed: {int(result['schema_report']['passed'].sum())}/{len(result['schema_report'])}",
        f"- Synthetic PII-indicator columns: {len(result['synthetic_pii_report'])}",
        f"- Average marginal distance: {fidelity_mean:.4f}",
        f"- Mean absolute numeric-correlation drift: {result['correlation_mae']:.4f}" if np.isfinite(result['correlation_mae']) else "- Mean absolute numeric-correlation drift: N/A",
        f"- Utility task: {util['task']} ({util['primary_metric']})",
        f"- Real-trained baseline: {util['baseline_primary']:.4f}",
        f"- Synthetic-trained model: {util['synthetic_primary']:.4f}",
        f"- Relative utility degradation (positive means worse): {util['relative_degradation_pct']:+.1f}%",
        f"- Exact full-row overlap: {privacy['exact_overlap_pct']:.2f}%",
        f"- Synthetic rows below near-distance cutoff: {privacy['synthetic_near_share_pct']:.1f}%",
    ]
    if bias.get("available"):
        lines += [
            f"- Group column: `{bias['group_column']}`; outcome threshold: `{bias['positive_outcome_threshold']}`",
            f"- Held-out real group pass-rate gap: {bias['real_gap_pp']:.1f} percentage points",
            f"- Synthetic group pass-rate gap: {bias['synthetic_gap_pp']:.1f} percentage points",
            f"- Absolute gap change: {bias['absolute_gap_change_pp']:.1f} percentage points",
        ]
    else:
        lines.append(f"- Bias comparison: not available ({bias.get('reason', 'not evaluated')})")
    lines += [
        "", "## Decision gates", "", result["decision_table"].to_markdown(index=False), "",
        "## Interpretation and limitations", "",
        "This report is an educational screening assessment. Statistical similarity does not prove privacy, and a low PII-indicator count does not prove anonymity. The generator is a lightweight Gaussian-copula-style sampler; its nominal-category encoding is simplified. The bias analysis measures one outcome-rate gap only and is not a complete fairness assessment. Do not use this output as the sole approval for production or high-stakes decisions.", "",
    ]
    return "\n".join(lines)


def build_export_zip(result: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("synthetic_data.csv", result["synthetic"].to_csv(index=False))
        zf.writestr("quality_report.json", json.dumps(json_safe(result["report_json"]), indent=2, ensure_ascii=False))
        zf.writestr("schema_audit.csv", result["schema_report"].to_csv(index=False))
        zf.writestr("marginal_fidelity.csv", result["fidelity"].to_csv(index=False))
        zf.writestr("correlation_drift.csv", result["correlation_detail"].to_csv(index=False))
        zf.writestr("decision_gates.csv", result["decision_table"].to_csv(index=False))
        zf.writestr("bias_comparison.csv", result["bias_table"].to_csv(index=False))
        zf.writestr("README_report.md", clean_report_markdown(result["report_json"] | {
            "synthetic": result["synthetic"], "schema_report": result["schema_report"],
            "fidelity": result["fidelity"], "correlation_detail": result["correlation_detail"],
            "decision_table": result["decision_table"], "bias_table": result["bias_table"],
        }))
    return buffer.getvalue()


# ---------------------------- Sidebar / source ----------------------------
st.sidebar.markdown("## SynthGuard")
st.sidebar.caption("Synthetic-data lab · Big Data Analytics")
source_choice = st.sidebar.radio("Data source", ["UCI Student Performance (recommended)", "Upload a CSV"])

source_name = ""
source_df = None
if source_choice == "UCI Student Performance (recommended)":
    with st.spinner("Loading public education dataset from UCI..."):
        try:
            source_df, dataset_url = load_uci_math()
            source_name = "UCI Student Performance — Math subset"
            st.sidebar.success(f"Loaded {len(source_df):,} rows")
        except Exception as exc:
            st.error(f"Unable to load the public dataset: {exc}")
            st.info("Try again later, or switch to Upload a CSV. The public dataset requires internet access.")
            st.stop()
else:
    uploaded = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    st.sidebar.caption("Only upload data you are authorised to process. Avoid real names, contact details, IDs, or sensitive records in a publicly deployed app.")
    if uploaded is None:
        st.markdown("# SynthGuard")
        st.markdown("### A synthetic-data generation and quality-evaluation agent")
        st.info("Choose the public UCI dataset or upload a CSV in the sidebar to begin.")
        st.stop()
    try:
        source_df = normalize_dataframe(pd.read_csv(uploaded, sep=None, engine="python"))
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()

if len(source_df) > 10000:
    st.warning("For this cloud-hosted teaching demo, only the first 10,000 rows are used. Select a smaller sample for predictable runtime.")
    source_df = source_df.head(10000).copy()

raw_pii_report = scan_pii(source_df)
columns = list(source_df.columns)
num_cols = source_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in columns if c not in num_cols]
def_target = "G3" if "G3" in columns else (num_cols[-1] if num_cols else columns[-1])

target = st.sidebar.selectbox("Target column for usefulness test", options=columns,
                              index=columns.index(def_target) if def_target in columns else len(columns) - 1)

group_options = [None] + [c for c in columns if c != target and source_df[c].nunique(dropna=True) >= 2]
def_group = "sex" if "sex" in group_options else next((c for c in group_options if c is not None and c in cat_cols), None)
group_col = st.sidebar.selectbox("Group column for bias comparison", options=group_options,
                                  index=group_options.index(def_group) if def_group in group_options else 0,
                                  format_func=lambda value: "Skip bias comparison" if value is None else str(value))
flagged_default = [c for c in (raw_pii_report["column"].tolist() if not raw_pii_report.empty else []) if c != target]
exclude_options = [c for c in columns if c != target]
excluded_columns = st.sidebar.multiselect("Exclude before synthesis (recommended for PII)", options=exclude_options,
                                          default=[c for c in flagged_default if c in exclude_options])
if target in (raw_pii_report["column"].tolist() if not raw_pii_report.empty else []):
    st.sidebar.warning("The selected target is flagged by the PII scanner. Choose a non-identifier target before analysis.")

max_synth = max(100, min(3000, len(source_df) * 2))
def_synth = max(100, min(len(source_df), 1000))
row_step = 50 if max_synth >= 500 else 10
max_synth = 100 + ((max_synth - 100) // row_step) * row_step
def_synth = 100 + round((def_synth - 100) / row_step) * row_step
def_synth = min(max_synth, max(100, def_synth))
n_synthetic = st.sidebar.slider("Synthetic records to generate", min_value=100, max_value=max_synth,
                                 value=def_synth, step=row_step)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
near_cutoff = st.sidebar.slider("Near-record distance cutoff", min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                 help="Illustrative Gower-style distance threshold. This is not a formal privacy standard.")

selected_for_run = [c for c in columns if c not in excluded_columns]
if target not in selected_for_run:
    selected_for_run.append(target)

numeric_target = pd.api.types.is_numeric_dtype(source_df[target])
if numeric_target:
    target_min = float(source_df[target].dropna().min())
    target_max = float(source_df[target].dropna().max())
    threshold_default = 10.0 if target_min <= 10 <= target_max else float(source_df[target].median())
    target_values = pd.to_numeric(source_df[target].dropna(), errors="coerce")
    threshold_step = 1.0 if np.allclose(target_values.to_numpy() % 1, 0) else max((target_max - target_min) / 100.0, 0.01)
    pass_threshold = st.sidebar.number_input("Positive-outcome threshold for bias check", min_value=target_min,
                                               max_value=target_max, value=float(threshold_default), step=float(threshold_step),
                                               help="For the education demo, G3 ≥ 10 means pass. For other numeric targets, define a meaningful threshold.")
else:
    pass_threshold = 0.0

with st.sidebar.expander("Decision guardrails", expanded=False):
    max_mean_distance = st.slider("Max average marginal distance", min_value=0.05, max_value=0.50, value=0.15, step=0.05)
    max_utility_degradation_pct = st.slider("Max utility degradation (%)", min_value=0, max_value=100, value=25, step=5)
    max_near_share_pct = st.slider("Max near-record share (%)", min_value=0.0, max_value=25.0, value=5.0, step=1.0)
    max_bias_gap_change_pp = st.slider("Max bias-gap change (pp)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)

st.sidebar.divider()
st.sidebar.caption("Cloud-hosted processing. This app does not intentionally save uploaded data to disk, but a public demo is not a secure environment for sensitive records.")

# ----------------------------- Main dashboard ----------------------------
st.markdown('<div class="sg-kicker">Big Data Analytics · Extra Experiment</div>', unsafe_allow_html=True)
st.title("SynthGuard")
st.markdown('<div class="sg-subtitle">Synthetic-data generation and quality-evaluation agent</div>', unsafe_allow_html=True)
st.write("Generate new tabular records, then audit their structure, statistical similarity, downstream usefulness, privacy-risk signals and group disparity—all in one cloud-hosted dashboard.")

summary_cols = st.columns(4)
summary_cols[0].metric("Real dataset", f"{len(source_df):,} rows")
summary_cols[1].metric("Columns", f"{len(source_df.columns)}")
summary_cols[2].metric("PII-indicator fields", f"{len(raw_pii_report)}")
summary_cols[3].metric("Target", target)

with st.expander("1 · Inspect source data and PII preflight", expanded=True):
    st.markdown("**Source preview** — columns flagged by the scanner are masked in this preview. The scanner reports field names and counts, never matching values.")
    st.dataframe(masked_preview(source_df, raw_pii_report), use_container_width=True, hide_index=True)
    if raw_pii_report.empty:
        st.success("No obvious direct-identifier indicators found by the current rules. This is not proof of anonymity.")
    else:
        st.warning("Potential direct-identifier indicators were found. Review these fields and keep them excluded from synthesis unless you have a justified, authorised use.")
        st.dataframe(raw_pii_report, use_container_width=True, hide_index=True)
    st.caption(f"Selected for generation: {len(selected_for_run)} columns. Excluded: {', '.join(excluded_columns) if excluded_columns else 'none'}. The target column is always retained for the usefulness test.")

current_signature = "|".join([source_signature(source_df), target, str(group_col), ",".join(sorted(excluded_columns)),
                               str(n_synthetic), str(seed), str(near_cutoff), str(pass_threshold), str(max_mean_distance),
                               str(max_utility_degradation_pct), str(max_near_share_pct), str(max_bias_gap_change_pp)])
if st.button("Run SynthGuard analysis", type="primary", use_container_width=True):
    if target in (raw_pii_report["column"].tolist() if not raw_pii_report.empty else []):
        st.error("The selected target is flagged as a direct-identifier field. Choose a non-identifier target first.")
    elif len(selected_for_run) < 2:
        st.error("At least two columns must remain after exclusions: one target and at least one feature.")
    else:
        try:
            with st.spinner("Preparing data, generating synthetic records, and running quality gates..."):
                prepared, dropped_target_rows, imputed_cells = prepare_analysis_frame(source_df[selected_for_run], target)
                train_real, test_real = train_test_split(prepared, test_size=0.25, random_state=int(seed))
                train_real = train_real.reset_index(drop=True)
                test_real = test_real.reset_index(drop=True)
                sampler = GaussianCopulaTabularSampler(int(seed)).fit(train_real)
                synthetic = sampler.sample(int(n_synthetic))
                schema_report = schema_audit(synthetic, prepared)
                synthetic_pii_report = scan_pii(synthetic)
                fidelity = marginal_fidelity(test_real, synthetic)
                mean_marginal_distance = float(fidelity["distance"].mean()) if not fidelity.empty else float("nan")
                corr_mae, correlation_detail = correlation_drift(test_real, synthetic)
                utility = evaluate_utility(train_real, test_real, synthetic, target, int(seed))
                privacy = evaluate_privacy(train_real, test_real, synthetic, float(near_cutoff), random_state=int(seed))
                bias_table, bias = bias_comparison(test_real, synthetic, target, group_col, float(pass_threshold))
                thresholds = {"max_mean_distance": float(max_mean_distance),
                              "max_utility_degradation_pct": float(max_utility_degradation_pct),
                              "max_near_share_pct": float(max_near_share_pct),
                              "max_bias_gap_change_pp": float(max_bias_gap_change_pp)}
                overall_decision, decision_table = build_decision(schema_report, synthetic_pii_report, fidelity,
                                                                   utility, privacy, bias, thresholds)
                report_json = {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_name": source_name, "source_rows": len(source_df), "source_columns": len(source_df.columns),
                    "prepared_rows": len(prepared), "train_rows": len(train_real), "heldout_rows": len(test_real),
                    "synthetic_rows": len(synthetic), "target": target, "group_column": group_col,
                    "positive_outcome_threshold": pass_threshold if numeric_target else None,
                    "excluded_columns": excluded_columns, "dropped_rows_missing_target": dropped_target_rows,
                    "missing_cells_imputed": imputed_cells, "schema_report": schema_report,
                    "source_pii_report": raw_pii_report, "synthetic_pii_report": synthetic_pii_report,
                    "fidelity": fidelity, "fidelity_mean": mean_marginal_distance,
                    "correlation_mae": corr_mae, "utility": utility, "privacy": privacy,
                    "bias": bias, "bias_table": bias_table, "thresholds": thresholds,
                    "overall_decision": overall_decision, "decision_table": decision_table,
                    "limitations": ["Educational Gaussian-copula-style generator; nominal category ordering is simplified.",
                                     "PII scanning catches indicators, not every identity leak.",
                                     "Nearest-distance and exact-row checks do not provide differential privacy.",
                                     "Bias comparison examines one group outcome-rate gap only."],
                }
                st.session_state["synthguard_result"] = {
                    "signature": current_signature, "synthetic": synthetic, "prepared": prepared,
                    "train_real": train_real, "test_real": test_real, "schema_report": schema_report,
                    "synthetic_pii_report": synthetic_pii_report, "fidelity": fidelity,
                    "fidelity_mean": mean_marginal_distance, "correlation_mae": corr_mae,
                    "correlation_detail": correlation_detail, "utility": utility, "privacy": privacy,
                    "bias_table": bias_table, "bias": bias, "decision_table": decision_table,
                    "overall_decision": overall_decision, "thresholds": thresholds,
                    "report_json": report_json, "source_pii_report": raw_pii_report,
                    "dropped_target_rows": dropped_target_rows, "imputed_cells": imputed_cells,
                    "excluded_columns": excluded_columns, "dataset_name": source_name,
                    "target": target, "group_col": group_col,
                }
        except Exception as exc:
            st.exception(exc)

result = st.session_state.get("synthguard_result")
if result is None:
    st.markdown("### Analysis workflow")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("**01 · Generate**")
        st.write("Fit a lightweight tabular synthesizer to the training split and produce new feature combinations.")
    with w2:
        st.markdown("**02 · Evaluate**")
        st.write("Check schema, marginal distributions, correlations, model utility, PII signals and near-record similarity.")
    with w3:
        st.markdown("**03 · Decide**")
        st.write("Convert each measurement into an explicit PASS, REVIEW or BLOCK gate.")
    st.info("Configure the dataset and target in the sidebar, then select **Run SynthGuard analysis**. For the default education data, use `G3` as the target and `sex` for the group comparison.")
    st.stop()
elif result["signature"] != current_signature:
    st.info("The data source or analysis settings changed. Run SynthGuard again to update the results.")
    st.stop()

# ------------------------------ Result tabs ------------------------------
tab_overview, tab_schema, tab_quality, tab_privacy, tab_export = st.tabs([
    "Overview", "Schema & PII", "Fidelity & Utility", "Privacy & Bias", "Export report"
])

with tab_overview:
    st.subheader("Agent recommendation")
    status = result["overall_decision"]
    if status == "PASS":
        st.success("PASS · All applicable configured demonstration gates passed. This is not production certification.")
    elif status == "REVIEW":
        st.warning("REVIEW · At least one quality or risk indicator is outside its configured guardrail. Inspect the details before using the data.")
    else:
        st.error("BLOCK · A schema or direct-PII gate failed. Do not treat this output as ready for use.")
    met = st.columns(4)
    met[0].metric("Synthetic records", f"{len(result['synthetic']):,}")
    schema_passed = int(result["schema_report"]["passed"].sum())
    met[1].metric("Schema checks", f"{schema_passed}/{len(result['schema_report'])}")
    met[2].metric("Mean marginal distance", f"{result['fidelity_mean']:.3f}")
    met[3].metric("Exact-row overlap", f"{result['privacy']['exact_overlap_pct']:.2f}%")
    st.markdown("#### Decision gates")
    st.dataframe(result["decision_table"], use_container_width=True, hide_index=True,
                 column_config={"status": st.column_config.TextColumn("Status", help="PASS / REVIEW / BLOCK gate")})
    st.markdown("#### Synthetic sample")
    safe_preview = result["synthetic"].head(12).copy()
    st.dataframe(safe_preview, use_container_width=True, hide_index=True)
    st.caption(f"Preprocessing: {result['dropped_target_rows']} row(s) without target dropped; {result['imputed_cells']} missing cell(s) imputed before splitting.")

with tab_schema:
    st.subheader("Schema validation")
    st.write("Schema checks test the generated records against the prepared source schema. Schema validity alone does not establish statistical quality or privacy.")
    st.dataframe(result["schema_report"], use_container_width=True, hide_index=True)
    st.subheader("PII screening")
    st.markdown("SynthGuard looks for common identifier-related column names and direct value patterns. It does not display any matched values.")
    if result["source_pii_report"].empty:
        st.success("No obvious direct-PII indicators detected in source columns by the current rules.")
    else:
        st.markdown("**Source indicators**")
        st.dataframe(result["source_pii_report"], use_container_width=True, hide_index=True)
    if result["synthetic_pii_report"].empty:
        st.success("No obvious direct-PII indicators detected in the synthetic output.")
    else:
        st.error("Potential direct-PII indicators remain in the synthetic output. Review these columns before use.")
        st.dataframe(result["synthetic_pii_report"], use_container_width=True, hide_index=True)
    with st.expander("PII detector self-test (fake canary values)"):
        fake_canaries = pd.DataFrame({
            "email": ["demo.student@example.com", "no-email"],
            "phone": ["+1-202-555-0198", "not a phone"],
            "student_id": ["12345678901", "sample-id"],
        })
        st.caption("These are fabricated test strings, not taken from the loaded dataset. The test validates common pattern detection only.")
        st.dataframe(scan_pii(fake_canaries), use_container_width=True, hide_index=True)
    st.info("For the public UCI example, `address` means urban/rural category and is deliberately not treated as a street address. PII rules are heuristic and may produce false positives or false negatives.")

with tab_quality:
    st.subheader("Distribution fidelity")
    q1, q2 = st.columns(2)
    q1.metric("Average marginal distance", f"{result['fidelity_mean']:.4f}", help="Lower is closer. Average is a dashboard summary, not a certified overall quality score.")
    q2.metric("Numeric correlation drift", f"{result['correlation_mae']:.4f}" if np.isfinite(result["correlation_mae"]) else "N/A",
              help="Mean absolute difference between numeric-feature correlations; lower is closer.")
    st.dataframe(result["fidelity"], use_container_width=True, hide_index=True)
    if not result["correlation_detail"].empty:
        with st.expander("Inspect numeric relationship drift"):
            st.dataframe(result["correlation_detail"].sort_values("absolute_difference", ascending=False).head(20),
                         use_container_width=True, hide_index=True)
    st.markdown("#### Compare one feature")
    feature = st.selectbox("Feature to compare", options=list(result["prepared"].columns), key="feature_compare")
    real_compare = result["test_real"][feature]
    synth_compare = result["synthetic"][feature]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if pd.api.types.is_numeric_dtype(result["prepared"][feature]):
        lo = min(float(real_compare.min()), float(synth_compare.min()))
        hi = max(float(real_compare.max()), float(synth_compare.max()))
        bins = min(20, max(5, int(np.sqrt(len(real_compare)))))
        ax.hist(real_compare, bins=bins, range=(lo, hi) if lo < hi else None, alpha=0.55, density=True, label="Held-out real")
        ax.hist(synth_compare, bins=bins, range=(lo, hi) if lo < hi else None, alpha=0.55, density=True, label="Synthetic")
        ax.set_ylabel("Density")
    else:
        levels = list(dict.fromkeys(list(real_compare.astype(str).unique()) + list(synth_compare.astype(str).unique())))
        if len(levels) > 15:
            levels = list(real_compare.astype(str).value_counts().head(12).index)
            ax.set_title("Top held-out real categories (other synthetic categories omitted from this view)")
        real_pct = real_compare.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0) * 100
        synth_pct = synth_compare.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0) * 100
        pos = np.arange(len(levels))
        width = 0.38
        ax.bar(pos - width/2, real_pct.values, width, label="Held-out real")
        ax.bar(pos + width/2, synth_pct.values, width, label="Synthetic")
        ax.set_xticks(pos)
        ax.set_xticklabels(levels, rotation=45, ha="right")
        ax.set_ylabel("Share (%)")
    ax.set_title(f"{feature}: held-out real vs synthetic")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Downstream model usefulness")
    utility = result["utility"]
    st.write(f"Task: **{utility['task']}** — predict `{result['target']}` from the remaining selected columns. Both models are tested on the same held-out real records.")
    ux = st.columns(3)
    ux[0].metric(f"Real-trained {utility['primary_metric']}", f"{utility['baseline_primary']:.4f}", help=utility["interpretation"])
    ux[1].metric(f"Synthetic-trained {utility['primary_metric']}", f"{utility['synthetic_primary']:.4f}", help=utility["interpretation"])
    ux[2].metric("Relative degradation", f"{utility['relative_degradation_pct']:+.1f}%", help="Positive means synthetic-trained model performed worse on the primary metric; negative means better.")
    util_rows = []
    for metric_name, value in utility["baseline_secondary"].items():
        util_rows.append({"metric": metric_name, "real-trained baseline": value,
                          "synthetic-trained": utility["synthetic_secondary"].get(metric_name)})
    st.dataframe(pd.DataFrame(util_rows), use_container_width=True, hide_index=True)

with tab_privacy:
    st.subheader("Privacy-risk screening")
    p1, p2, p3 = st.columns(3)
    p1.metric("Exact full-row overlap", f"{result['privacy']['exact_overlap_pct']:.2f}%")
    p2.metric("Synthetic near-record share", f"{result['privacy']['synthetic_near_share_pct']:.1f}%")
    p3.metric("Near-distance cutoff", f"{result['privacy']['near_cutoff']:.2f}")
    st.write(f"Median nearest distance — synthetic: **{result['privacy']['synthetic_median_nearest_distance']:.3f}**; held-out real: **{result['privacy']['heldout_median_nearest_distance']:.3f}**.")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Synthetic-to-training nearest distance**")
        counts, edges = np.histogram(result["privacy"]["synthetic_nearest_distances"], bins=10)
        labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(counts))]
        st.bar_chart(pd.DataFrame({"Distance band": labels, "Count": counts}).set_index("Distance band"), use_container_width=True)
    with s2:
        st.markdown("**Held-out-real-to-training nearest distance**")
        counts, edges = np.histogram(result["privacy"]["heldout_nearest_distances"], bins=10)
        labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(counts))]
        st.bar_chart(pd.DataFrame({"Distance band": labels, "Count": counts}).set_index("Distance band"), use_container_width=True)
    st.warning(result["privacy"]["warning"])
    st.caption(f"Distance comparisons were capped at a sample of {result['privacy']['comparison_rows_sampled']} records to keep the cloud app responsive.")

    st.subheader("Bias comparison")
    bias = result["bias"]
    if bias.get("available"):
        b1, b2, b3 = st.columns(3)
        b1.metric("Held-out real pass-rate gap", f"{bias['real_gap_pp']:.1f} pp")
        b2.metric("Synthetic pass-rate gap", f"{bias['synthetic_gap_pp']:.1f} pp")
        b3.metric("Absolute gap change", f"{bias['absolute_gap_change_pp']:.1f} pp")
        st.dataframe(result["bias_table"], use_container_width=True, hide_index=True)
        st.caption(f"Positive outcome: `{result['target']} ≥ {bias['positive_outcome_threshold']}`. Group attribute: `{bias['group_column']}`.")
        st.warning(bias["warning"])
    else:
        st.info(bias.get("reason", "Bias comparison was not available for this dataset configuration."))

with tab_export:
    st.subheader("Export artifacts")
    st.write("Download the generated data and machine-readable metrics for your lab report. Inspect the dataset and privacy findings before sharing any file.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button("Download synthetic CSV", data=result["synthetic"].to_csv(index=False).encode("utf-8"),
                           file_name="synthguard_synthetic_data.csv", mime="text/csv", use_container_width=True)
    with col_b:
        safe_json = json.dumps(json_safe(result["report_json"]), indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button("Download metrics JSON", data=safe_json, file_name="synthguard_quality_report.json",
                           mime="application/json", use_container_width=True)
    with col_c:
        zip_bytes = build_export_zip(result)
        st.download_button("Download complete report ZIP", data=zip_bytes, file_name="synthguard_report_bundle.zip",
                           mime="application/zip", use_container_width=True, type="primary")
    st.markdown("#### Report-ready conclusion")
    util = result["utility"]
    bias = result["bias"]
    bias_sentence = (f"The group pass-rate gap changed by {bias['absolute_gap_change_pp']:.1f} percentage points." if bias.get("available") else "A group pass-rate gap was not calculated for this configuration.")
    conclusion = (
        f"SynthGuard generated {len(result['synthetic'])} synthetic records from {result['prepared'].shape[0]} prepared records in the {result['dataset_name']} dataset. "
        f"The schema audit passed {int(result['schema_report']['passed'].sum())} of {len(result['schema_report'])} checks, and the direct-PII scanner flagged {len(result['synthetic_pii_report'])} synthetic-output column(s). "
        f"The average marginal distance was {result['fidelity_mean']:.4f}; mean absolute numerical-correlation drift was {result['correlation_mae']:.4f} "
        f"if numeric correlations were available. On held-out real records, the real-trained model's {util['primary_metric']} was {util['baseline_primary']:.4f}, compared with {util['synthetic_primary']:.4f} for the synthetic-trained model, "
        f"giving a relative degradation of {util['relative_degradation_pct']:+.1f}%. Exact full-row overlap was {result['privacy']['exact_overlap_pct']:.2f}%, and {result['privacy']['synthetic_near_share_pct']:.1f}% of sampled synthetic records fell below the configured nearest-distance cutoff. {bias_sentence} "
        f"The agent recommendation was {result['overall_decision']}. These results are educational screening indicators, not a formal guarantee of anonymity, differential privacy, or fairness."
    )
    st.text_area("Copy this into the report after verifying the result", value=conclusion, height=200)
    st.markdown("#### Methodology notes")
    st.markdown("- The generator is a simplified Gaussian-copula-style sampler fitted only on the real training split.\n- Both utility models are evaluated on the same held-out real test split.\n- Privacy checks use exact-row overlap and mixed-type nearest-record distances.\n- Bias extension compares one group outcome-rate gap across the selected group column.\n- PASS means configured demonstration gates passed; it is not certification for production use.")

st.divider()
st.caption("SynthGuard · Big Data Analytics mini-project · Use synthetic output only after reviewing the gate results and limitations.")
