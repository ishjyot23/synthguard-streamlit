"""Core logic for SynthGuard: synthetic-data generation and quality evaluation.

This is an educational Gaussian-copula-style sampler and an audit dashboard,
not a certified anonymisation or differential-privacy system.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

# Conservative name-based rules for common direct identifiers.
_PII_NAME_PATTERNS = [
    ("Email field name", re.compile(r"(^|_)(e.?mail)(_|$)|email", re.I)),
    ("Phone field name", re.compile(r"phone|mobile|telephone|contact.?number", re.I)),
    ("Person name field", re.compile(r"(^|_)(name|full.?name|first.?name|last.?name|person.?name)(_|$)", re.I)),
    ("Government identifier field", re.compile(r"ssn|social.?security|passport|national.?id|tax.?id", re.I)),
    ("Account/customer/student identifier field", re.compile(r"customer.?id|student.?id|user.?id|account.?number|(^|_)id($|_)", re.I)),
    ("Birth-date field", re.compile(r"date.?of.?birth|birth.?date|(^|_)dob($|_)", re.I)),
    ("Detailed address field", re.compile(r"street|postal.?code|zip.?code|full.?address|home.?address|ip.?address", re.I)),
]
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){9,15}(?!\d)")
_LONG_DIGIT_RE = re.compile(r"^\d{9,}$")


def normalize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize labels, remove empty columns, and reject ambiguous duplicate names."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    result = frame.copy()
    result.columns = [str(c).strip() for c in result.columns]
    if any(not c for c in result.columns):
        raise ValueError("One or more columns have blank names. Rename them before continuing.")
    if pd.Index(result.columns).duplicated().any():
        duplicates = pd.Index(result.columns)[pd.Index(result.columns).duplicated()].tolist()
        raise ValueError(f"Duplicate column names are not supported: {duplicates}")
    result = result.loc[:, ~result.columns.str.match(r"^Unnamed(:\s*\d+)?$", case=False)]
    result = result.dropna(axis=1, how="all")
    if result.empty or len(result.columns) < 2:
        raise ValueError("The dataset must contain at least two non-empty columns.")
    return result.replace([np.inf, -np.inf], np.nan)


def _is_uci_address_category(series: pd.Series) -> bool:
    """The UCI Student Performance 'address' feature means urban/rural, not street address."""
    vals = set(series.dropna().astype(str).str.strip().str.lower().unique().tolist())
    return bool(vals) and vals.issubset({"u", "r", "urban", "rural"})


def scan_pii(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only detection summaries; never return or display matched values."""
    rows: list[dict[str, Any]] = []
    for col in frame.columns:
        name = str(col)
        indicators: list[str] = []
        for label, pattern in _PII_NAME_PATTERNS:
            if pattern.search(name):
                indicators.append(label)
        if name.strip().lower() == "address" and not _is_uci_address_category(frame[col]):
            indicators.append("Address-like field name")

        values = frame[col].dropna()
        text_values = values.astype(str)
        email_count = int(text_values.map(lambda x: bool(_EMAIL_RE.search(x))).sum())
        # Phone-pattern matching is applied to text columns to reduce false positives
        # from ordinary numeric attributes such as grades and ages.
        phone_count = 0
        if not pd.api.types.is_numeric_dtype(frame[col]):
            phone_count = int(text_values.map(lambda x: bool(_PHONE_RE.search(x))).sum())
        long_digit_count = int(text_values.map(lambda x: bool(_LONG_DIGIT_RE.fullmatch(x.strip()))).sum())

        if email_count:
            indicators.append("Email-like value")
        if phone_count:
            indicators.append("Phone-like value")
        if long_digit_count:
            indicators.append("Long numeric ID-like value")
        indicators = list(dict.fromkeys(indicators))
        if indicators:
            high = any("field name" in s.lower() or "identifier field" in s.lower() or "government identifier" in s.lower() for s in indicators)
            rows.append({
                "column": name,
                "indicators": "; ".join(indicators),
                "matching_values_scanned": max(email_count, phone_count, long_digit_count),
                "severity": "High" if high else "Review",
            })
    return pd.DataFrame(rows, columns=["column", "indicators", "matching_values_scanned", "severity"])


def prepare_analysis_frame(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int, int]:
    """Drop rows without a target; median/mode-impute feature fields for the demo."""
    data = normalize_dataframe(frame)
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} is not present.")
    before_rows = len(data)
    data = data.dropna(subset=[target]).reset_index(drop=True)
    dropped_target_rows = before_rows - len(data)
    if len(data) < 20:
        raise ValueError("At least 20 records with a non-missing target are required for a train/test evaluation.")
    missing_cells = int(data.isna().sum().sum())
    for col in data.columns:
        if not data[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(data[col]):
            median = data[col].median()
            data[col] = data[col].fillna(0 if pd.isna(median) else median)
        else:
            modes = data[col].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "Unknown"
            data[col] = data[col].fillna(fill_value)
    return data.reset_index(drop=True), dropped_target_rows, missing_cells


class GaussianCopulaTabularSampler:
    """A lightweight educational Gaussian-copula-style tabular sampler.

    Categorical values are mapped to latent normal scores ordered by frequency.
    That preserves approximate marginals and some dependencies, but imposes an
    arbitrary latent ordering on nominal categories. Inspect the fidelity metrics.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = int(random_state)
        self.fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "GaussianCopulaTabularSampler":
        if frame.empty or len(frame.columns) < 1:
            raise ValueError("Cannot fit a synthesizer to an empty dataset.")
        if frame.isna().any().any():
            raise ValueError("Missing values remain. Impute or remove them before synthesis.")
        self.columns_ = list(frame.columns)
        self.meta_: dict[str, dict[str, Any]] = {}
        latent_columns: list[np.ndarray] = []
        n = len(frame)
        for col in self.columns_:
            series = frame[col]
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series).to_numpy(dtype=float)
                sorted_values = np.sort(values)
                ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
                u = np.clip((ranks - 0.5) / n, 1e-5, 1 - 1e-5)
                z = norm.ppf(u)
                self.meta_[col] = {
                    "kind": "numeric",
                    "sorted_values": sorted_values,
                    "dtype": series.dtype,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
                latent_columns.append(z)
            else:
                raw = series.astype(str)
                original_values: dict[str, Any] = {}
                for original in series.tolist():
                    original_values.setdefault(str(original), original)
                counts = raw.value_counts(normalize=True, sort=True)
                categories = counts.index.tolist()
                probabilities = counts.to_numpy(dtype=float)
                cumulative = np.cumsum(probabilities)
                lower = np.concatenate(([0.0], cumulative[:-1]))
                midpoints = (lower + cumulative) / 2
                category_z = norm.ppf(np.clip(midpoints, 1e-5, 1 - 1e-5))
                mapping = {cat: float(score) for cat, score in zip(categories, category_z)}
                z = raw.map(mapping).to_numpy(dtype=float)
                self.meta_[col] = {
                    "kind": "categorical",
                    "categories": [original_values[c] for c in categories],
                    "category_strings": categories,
                    "cumulative": cumulative,
                    "dtype": series.dtype,
                }
                latent_columns.append(z)

        latent_matrix = np.column_stack(latent_columns)
        corr = np.corrcoef(latent_matrix.T)
        corr = np.atleast_2d(np.asarray(corr, dtype=float))
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        # Project the estimate to a positive-semidefinite correlation matrix.
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.clip(eigvals, 1e-6, None)
        corr = (eigvecs * eigvals) @ eigvecs.T
        scale = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
        corr = corr / np.outer(scale, scale)
        np.fill_diagonal(corr, 1.0)
        self.correlation_ = corr
        self.fitted_ = True
        return self

    def sample(self, n_rows: int) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() before sample().")
        if int(n_rows) < 1:
            raise ValueError("Synthetic row count must be at least 1.")
        rng = np.random.default_rng(self.random_state)
        latent = rng.multivariate_normal(
            mean=np.zeros(len(self.columns_)), cov=self.correlation_, size=int(n_rows), check_valid="warn"
        )
        if latent.ndim == 1:
            latent = latent.reshape(-1, len(self.columns_))
        out: dict[str, Any] = {}
        for j, col in enumerate(self.columns_):
            meta = self.meta_[col]
            u = np.clip(norm.cdf(latent[:, j]), 0.0, 1.0 - 1e-12)
            if meta["kind"] == "numeric":
                vals = meta["sorted_values"]
                idx = np.clip(np.floor(u * len(vals)).astype(int), 0, len(vals) - 1)
                sampled = vals[idx]
                sampled = np.clip(sampled, meta["minimum"], meta["maximum"])
                try:
                    sampled = pd.Series(sampled).astype(meta["dtype"]).to_numpy()
                except (TypeError, ValueError):
                    pass
                out[col] = sampled
            else:
                idx = np.searchsorted(meta["cumulative"], u, side="right")
                idx = np.clip(idx, 0, len(meta["categories"]) - 1)
                sampled = [meta["categories"][i] for i in idx]
                try:
                    sampled = pd.Series(sampled).astype(meta["dtype"]).to_numpy()
                except (TypeError, ValueError):
                    pass
                out[col] = sampled
        return pd.DataFrame(out, columns=self.columns_)


def schema_audit(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected, actual = set(reference.columns), set(candidate.columns)
    rows.append({"check": "Column set", "passed": expected == actual,
                 "detail": f"Missing={sorted(expected-actual)}; extra={sorted(actual-expected)}"})
    rows.append({"check": "Non-empty output", "passed": len(candidate) > 0 and len(candidate.columns) > 0,
                 "detail": f"{len(candidate)} rows × {len(candidate.columns)} columns"})
    rows.append({"check": "No missing values", "passed": not candidate.isna().any().any(),
                 "detail": f"{int(candidate.isna().sum().sum())} missing cell(s)"})
    for col in reference.columns:
        if col not in candidate.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference[col]):
            is_numeric = pd.api.types.is_numeric_dtype(candidate[col])
            rows.append({"check": f"Numeric type: {col}", "passed": bool(is_numeric),
                         "detail": f"candidate dtype={candidate[col].dtype}; expected numeric"})
            if is_numeric and len(candidate):
                lo, hi = float(reference[col].min()), float(reference[col].max())
                c_lo, c_hi = float(candidate[col].min()), float(candidate[col].max())
                ok = c_lo >= lo - 1e-9 and c_hi <= hi + 1e-9
                rows.append({"check": f"Range: {col}", "passed": bool(ok),
                             "detail": f"candidate [{c_lo:g}, {c_hi:g}] vs reference [{lo:g}, {hi:g}]"})
        else:
            allowed = set(reference[col].astype(str).unique())
            observed = set(candidate[col].astype(str).unique())
            invalid = sorted(observed - allowed)
            rows.append({"check": f"Category domain: {col}", "passed": len(invalid) == 0,
                         "detail": "All values in reference domain" if not invalid else f"Unexpected={invalid[:10]}"})
    return pd.DataFrame(rows, columns=["check", "passed", "detail"])


def marginal_fidelity(reference: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in reference.columns:
        real_s, synth_s = reference[col], synthetic[col]
        if pd.api.types.is_numeric_dtype(reference[col]):
            x = pd.to_numeric(real_s).to_numpy(dtype=float)
            y = pd.to_numeric(synth_s).to_numpy(dtype=float)
            scale = max(float(np.nanmax(x) - np.nanmin(x)), 1.0)
            distance = float(wasserstein_distance(x, y) / scale)
            metric = "Normalized Wasserstein distance"
        else:
            levels = sorted(set(real_s.astype(str)) | set(synth_s.astype(str)))
            p = real_s.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0)
            q = synth_s.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0)
            distance = float(0.5 * np.abs(p.to_numpy() - q.to_numpy()).sum())
            metric = "Total variation distance"
        rows.append({"column": col, "type": "Numeric" if pd.api.types.is_numeric_dtype(reference[col]) else "Categorical",
                     "metric": metric, "distance": distance})
    return pd.DataFrame(rows).sort_values("distance", ascending=False).reset_index(drop=True)


def correlation_drift(reference: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    numeric = [c for c in reference.columns if pd.api.types.is_numeric_dtype(reference[c]) and c in synthetic.columns]
    if len(numeric) < 2:
        return float("nan"), pd.DataFrame(columns=["feature_a", "feature_b", "real_corr", "synthetic_corr", "absolute_difference"])
    real_corr = reference[numeric].corr().fillna(0.0)
    synth_corr = synthetic[numeric].corr().fillna(0.0)
    diffs = (real_corr - synth_corr).abs()
    upper_mask = np.triu(np.ones(diffs.shape, dtype=bool), k=1)
    rows = []
    for i, a in enumerate(numeric):
        for j, b in enumerate(numeric):
            if upper_mask[i, j]:
                rows.append({"feature_a": a, "feature_b": b, "real_corr": float(real_corr.iloc[i, j]),
                             "synthetic_corr": float(synth_corr.iloc[i, j]),
                             "absolute_difference": float(diffs.iloc[i, j])})
    detail = pd.DataFrame(rows)
    return (float(detail["absolute_difference"].mean()) if not detail.empty else float("nan"), detail)


def _build_model(X: pd.DataFrame, task: str, seed: int) -> Pipeline:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if not transformers:
        raise ValueError("No usable feature columns remain after excluding the target.")
    prep = ColumnTransformer(transformers, remainder="drop")
    if task == "regression":
        model = RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=80, min_samples_leaf=2, random_state=seed,
                                       n_jobs=-1, class_weight="balanced_subsample")
    return Pipeline([("prep", prep), ("model", model)])


def evaluate_utility(real_train: pd.DataFrame, real_test: pd.DataFrame, synthetic: pd.DataFrame,
                     target: str, random_state: int = 42) -> dict[str, Any]:
    feature_cols = [c for c in real_train.columns if c != target]
    if not feature_cols:
        raise ValueError("The target is the only available column; at least one feature is required.")
    y_train, y_test, y_syn = real_train[target], real_test[target], synthetic[target]
    is_regression = pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 10
    task = "regression" if is_regression else "classification"
    x_train, x_test, x_syn = real_train[feature_cols], real_test[feature_cols], synthetic[feature_cols]
    real_model = _build_model(x_train, task, random_state)
    synth_model = _build_model(x_syn, task, random_state)
    real_model.fit(x_train, y_train)
    synth_model.fit(x_syn, y_syn)
    pred_real = real_model.predict(x_test)
    pred_synth = synth_model.predict(x_test)
    if task == "regression":
        base_mae = float(mean_absolute_error(y_test, pred_real))
        synth_mae = float(mean_absolute_error(y_test, pred_synth))
        base_rmse = float(np.sqrt(mean_squared_error(y_test, pred_real)))
        synth_rmse = float(np.sqrt(mean_squared_error(y_test, pred_synth)))
        base_r2 = float(r2_score(y_test, pred_real)) if len(y_test) >= 2 else float("nan")
        synth_r2 = float(r2_score(y_test, pred_synth)) if len(y_test) >= 2 else float("nan")
        degradation = ((synth_mae - base_mae) / max(base_mae, 1e-9)) * 100.0
        return {"task": task, "primary_metric": "MAE", "baseline_primary": base_mae,
                "synthetic_primary": synth_mae, "relative_degradation_pct": float(degradation),
                "baseline_secondary": {"RMSE": base_rmse, "R2": base_r2},
                "synthetic_secondary": {"RMSE": synth_rmse, "R2": synth_r2},
                "interpretation": "Lower MAE/RMSE is better; higher R² is better."}
    base_f1 = float(f1_score(y_test, pred_real, average="macro", zero_division=0))
    synth_f1 = float(f1_score(y_test, pred_synth, average="macro", zero_division=0))
    degradation = ((base_f1 - synth_f1) / max(base_f1, 1e-9)) * 100.0
    return {"task": task, "primary_metric": "Macro-F1", "baseline_primary": base_f1,
            "synthetic_primary": synth_f1, "relative_degradation_pct": float(degradation),
            "baseline_secondary": {"Accuracy": float(accuracy_score(y_test, pred_real)),
                                    "Balanced accuracy": float(balanced_accuracy_score(y_test, pred_real))},
            "synthetic_secondary": {"Accuracy": float(accuracy_score(y_test, pred_synth)),
                                    "Balanced accuracy": float(balanced_accuracy_score(y_test, pred_synth))},
            "interpretation": "Higher Macro-F1/accuracy/balanced accuracy is better."}


def _row_key(row: tuple[Any, ...]) -> tuple[str, ...]:
    output = []
    for value in row:
        if pd.isna(value):
            output.append("<NA>")
        elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
            output.append(format(float(value), ".12g"))
        else:
            output.append(str(value))
    return tuple(output)


def mixed_distance_matrix(query: pd.DataFrame, reference: pd.DataFrame, scale_reference: pd.DataFrame) -> np.ndarray:
    """Vectorized Gower-style distance matrix for numeric and categorical fields."""
    cols = [c for c in reference.columns if c in query.columns and c in scale_reference.columns]
    if not cols:
        return np.zeros((len(query), len(reference)), dtype=float)
    distances = np.zeros((len(query), len(reference)), dtype=float)
    for col in cols:
        q, r = query[col], reference[col]
        if pd.api.types.is_numeric_dtype(scale_reference[col]):
            qv = pd.to_numeric(q).to_numpy(dtype=float)[:, None]
            rv = pd.to_numeric(r).to_numpy(dtype=float)[None, :]
            span = float(pd.to_numeric(scale_reference[col]).max() - pd.to_numeric(scale_reference[col]).min())
            if span <= 0:
                component = np.zeros((len(query), len(reference)), dtype=float)
            else:
                component = np.abs(qv - rv) / span
        else:
            qv = q.astype(str).to_numpy()[:, None]
            rv = r.astype(str).to_numpy()[None, :]
            component = (qv != rv).astype(float)
        distances += np.clip(component, 0.0, 1.0)
    return distances / len(cols)


def evaluate_privacy(real_train: pd.DataFrame, real_test: pd.DataFrame, synthetic: pd.DataFrame,
                     near_cutoff: float = 0.05, max_rows: int = 500, random_state: int = 42) -> dict[str, Any]:
    train_keys = {_row_key(row) for row in real_train.itertuples(index=False, name=None)}
    synth_keys = [_row_key(row) for row in synthetic.itertuples(index=False, name=None)]
    exact_count = sum(key in train_keys for key in synth_keys)
    rng = np.random.default_rng(random_state)

    def sample_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) > max_rows:
            positions = rng.choice(len(frame), size=max_rows, replace=False)
            return frame.iloc[positions].reset_index(drop=True)
        return frame.reset_index(drop=True)

    train_sample = sample_rows(real_train)
    synth_sample = sample_rows(synthetic)
    test_sample = sample_rows(real_test)
    synth_nn = mixed_distance_matrix(synth_sample, train_sample, real_train).min(axis=1)
    test_nn = mixed_distance_matrix(test_sample, train_sample, real_train).min(axis=1)
    return {
        "exact_overlap_count": int(exact_count),
        "exact_overlap_pct": float(100 * exact_count / max(len(synthetic), 1)),
        "near_cutoff": float(near_cutoff),
        "synthetic_near_share_pct": float(100 * np.mean(synth_nn < near_cutoff)) if len(synth_nn) else 0.0,
        "synthetic_median_nearest_distance": float(np.median(synth_nn)) if len(synth_nn) else float("nan"),
        "heldout_median_nearest_distance": float(np.median(test_nn)) if len(test_nn) else float("nan"),
        "synthetic_nearest_distances": synth_nn,
        "heldout_nearest_distances": test_nn,
        "comparison_rows_sampled": {"synthetic": len(synth_sample), "real_train": len(train_sample), "heldout_real": len(test_sample)},
        "warning": "Similarity screening only; this is not a re-identification attack or a formal privacy guarantee.",
    }


def bias_comparison(real_test: pd.DataFrame, synthetic: pd.DataFrame, target: str,
                    group_column: str | None, threshold: float = 10.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not group_column or group_column not in real_test.columns or group_column not in synthetic.columns:
        return pd.DataFrame(columns=["dataset", "group", "n", "positive_rate_pct", "mean_outcome"]), {
            "available": False, "reason": "Choose a group column to evaluate disparity."
        }
    if not pd.api.types.is_numeric_dtype(real_test[target]) or not pd.api.types.is_numeric_dtype(synthetic[target]):
        return pd.DataFrame(columns=["dataset", "group", "n", "positive_rate_pct", "mean_outcome"]), {
            "available": False, "reason": "Bias pass-rate comparison requires a numeric outcome."
        }
    def one(frame: pd.DataFrame, label: str) -> list[dict[str, Any]]:
        rows = []
        for value, group in frame.groupby(group_column, dropna=False):
            rows.append({"dataset": label, "group": str(value), "n": len(group),
                         "positive_rate_pct": float(100 * (pd.to_numeric(group[target]) >= threshold).mean()),
                         "mean_outcome": float(pd.to_numeric(group[target]).mean())})
        return rows
    table = pd.DataFrame(one(real_test, "Held-out real") + one(synthetic, "Synthetic"))
    def gap(label: str) -> float:
        vals = table.loc[table["dataset"] == label, "positive_rate_pct"].dropna().to_numpy(dtype=float)
        return float(vals.max() - vals.min()) if len(vals) >= 2 else float("nan")
    real_gap, synth_gap = gap("Held-out real"), gap("Synthetic")
    change = abs(synth_gap - real_gap) if np.isfinite(real_gap) and np.isfinite(synth_gap) else float("nan")
    return table, {"available": bool(np.isfinite(change)), "target": target, "group_column": group_column,
                   "positive_outcome_threshold": float(threshold), "real_gap_pp": real_gap,
                   "synthetic_gap_pp": synth_gap, "absolute_gap_change_pp": change,
                   "warning": "This is a narrow group-disparity proxy, not a complete fairness or causal assessment."}


def build_decision(schema_report: pd.DataFrame, synthetic_pii_report: pd.DataFrame,
                   fidelity: pd.DataFrame, utility: dict[str, Any], privacy: dict[str, Any],
                   bias: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    schema_ok = bool(schema_report["passed"].all()) if not schema_report.empty else False
    pii_findings = 0 if synthetic_pii_report.empty else len(synthetic_pii_report)
    mean_distance = float(fidelity["distance"].mean()) if not fidelity.empty else float("nan")
    near_share = float(privacy["synthetic_near_share_pct"])
    utility_degradation = float(utility["relative_degradation_pct"])

    def add(name: str, observed: str, guardrail: str, status: str) -> None:
        rows.append({"check": name, "observed": observed, "guardrail": guardrail, "status": status})

    add("Schema and domain validity", "All passed" if schema_ok else "One or more failed",
        "All checks pass", "PASS" if schema_ok else "BLOCK")
    add("Direct PII indicators", f"{pii_findings} flagged column(s)", "Zero flagged columns",
        "PASS" if pii_findings == 0 else "BLOCK")
    add("Average marginal distance", f"{mean_distance:.4f}" if np.isfinite(mean_distance) else "N/A",
        f"≤ {thresholds['max_mean_distance']:.2f}",
        "PASS" if np.isfinite(mean_distance) and mean_distance <= thresholds["max_mean_distance"] else "REVIEW")
    add("Downstream utility degradation", f"{utility_degradation:+.1f}%" if np.isfinite(utility_degradation) else "N/A",
        f"≤ {thresholds['max_utility_degradation_pct']:.0f}% degradation",
        "PASS" if np.isfinite(utility_degradation) and utility_degradation <= thresholds["max_utility_degradation_pct"] else "REVIEW")
    add("Near-record share", f"{near_share:.1f}%", f"≤ {thresholds['max_near_share_pct']:.1f}%",
        "PASS" if near_share <= thresholds["max_near_share_pct"] else "REVIEW")
    bias_change = float(bias.get("absolute_gap_change_pp", float("nan")))
    if bias.get("available") and np.isfinite(bias_change):
        add("Group pass-rate gap change", f"{bias_change:.1f} percentage points",
            f"≤ {thresholds['max_bias_gap_change_pp']:.1f} pp",
            "PASS" if bias_change <= thresholds["max_bias_gap_change_pp"] else "REVIEW")
    else:
        add("Group pass-rate gap change", "Not evaluated", "Choose a suitable group and numeric outcome", "N/A")
    table = pd.DataFrame(rows)
    if (table["status"] == "BLOCK").any():
        overall = "BLOCK"
    elif (table["status"] == "REVIEW").any():
        overall = "REVIEW"
    else:
        overall = "PASS"
    return overall, table


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy/Pandas values to JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_safe(value.tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value
