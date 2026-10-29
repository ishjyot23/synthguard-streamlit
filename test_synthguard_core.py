"""Small deterministic smoke tests for SynthGuard's core workflow."""
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from synthguard_core import (
    GaussianCopulaTabularSampler,
    bias_comparison,
    correlation_drift,
    evaluate_privacy,
    evaluate_utility,
    marginal_fidelity,
    prepare_analysis_frame,
    scan_pii,
    schema_audit,
)


class SynthGuardSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(19)
        n = 240
        df = pd.DataFrame({
            "age": rng.integers(15, 22, size=n),
            "studytime": rng.integers(1, 5, size=n),
            "sex": rng.choice(["F", "M"], size=n),
            "address": rng.choice(["U", "R"], size=n),
            "G1": rng.integers(0, 21, size=n),
            "G2": rng.integers(0, 21, size=n),
        })
        df["G3"] = np.clip(np.rint(0.4 * df["G1"] + 0.4 * df["G2"] + rng.normal(2, 3, n)), 0, 20).astype(int)
        cls.data = df

    def test_pii_scanner_detects_fake_canaries_and_not_uci_address(self):
        test = pd.DataFrame({
            "email": ["student@example.com"],
            "phone": ["+1-202-555-0198"],
            "student_id": ["12345678901"],
            "address": ["U"],
        })
        found = scan_pii(test)
        self.assertIn("email", set(found["column"]))
        self.assertIn("phone", set(found["column"]))
        self.assertIn("student_id", set(found["column"]))
        self.assertNotIn("address", set(found["column"]))

    def test_end_to_end_pipeline(self):
        prepared, dropped, imputed = prepare_analysis_frame(self.data, "G3")
        train, test = train_test_split(prepared, test_size=0.25, random_state=42)
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)
        synthetic = GaussianCopulaTabularSampler(42).fit(train).sample(120)
        self.assertEqual(list(synthetic.columns), list(prepared.columns))
        schema = schema_audit(synthetic, prepared)
        self.assertTrue(schema["passed"].all(), schema.loc[~schema["passed"]].to_dict("records"))
        fidelity = marginal_fidelity(test, synthetic)
        self.assertGreaterEqual(float(fidelity["distance"].mean()), 0)
        corr, _ = correlation_drift(test, synthetic)
        self.assertTrue(np.isfinite(corr))
        utility = evaluate_utility(train, test, synthetic, "G3")
        self.assertIn(utility["task"], {"regression", "classification"})
        privacy = evaluate_privacy(train, test, synthetic)
        self.assertGreaterEqual(privacy["exact_overlap_pct"], 0)
        bias_table, bias = bias_comparison(test, synthetic, "G3", "sex", 10)
        self.assertTrue(bias["available"])
        self.assertFalse(bias_table.empty)


if __name__ == "__main__":
    unittest.main()
