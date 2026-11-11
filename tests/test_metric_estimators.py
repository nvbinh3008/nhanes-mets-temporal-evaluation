"""Synthetic tests; no participant records or serialized models are opened."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phaseb_metrics import Metric, NAMES, replicate_weights


class SyntheticChecks(unittest.TestCase):
    def test_weighted_ties_and_zero_replicate_weights(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        p = np.array([.2, .2, .5, .8, .8, .9])
        w = np.array([2, 3, 0, 2, 4, 0.])
        result = Metric(y, p, .5)(w)
        self.assertAlmostEqual(result[NAMES.index("roc_auc")], roc_auc_score(y, p, sample_weight=w), 12)
        self.assertAlmostEqual(result[NAMES.index("average_precision")], average_precision_score(y, p, sample_weight=w), 12)

    def test_rescale_and_singleton_guard(self):
        data = pd.DataFrame({"stratum": ["a", "a", "b", "b"], "psu": [1, 2, 1, 2], "analysis_weight": [1., 1., 1., 1.]})
        samples = list(replicate_weights(data, B=8))
        self.assertEqual(len(samples), 8)
        for w in samples:
            self.assertEqual(sorted(w[:2]), [0., 2.])
            self.assertEqual(sorted(w[2:]), [0., 2.])
        with self.assertRaisesRegex(RuntimeError, "Singleton"):
            list(replicate_weights(data.iloc[:3], B=1))

    def test_missing_authorized_archive_is_actionable(self):
        env = {k: v for k, v in os.environ.items() if k not in ("NHANES_SOURCE_ROOT", "NHANES_OUTPUT_ROOT")}
        r = subprocess.run([sys.executable, str(ROOT / "scripts/phaseb_metrics.py")], env=env, capture_output=True, text=True, cwd=ROOT)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Set NHANES_SOURCE_ROOT", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
