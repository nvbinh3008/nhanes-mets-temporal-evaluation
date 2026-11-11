"""Check the corrected PSU draw and the installed CLI using synthetic inputs."""
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phaseb_metrics import B, SEED, replicate_weights


class BootstrapSafetyChecks(unittest.TestCase):
    def test_rescaled_draw_count_and_cluster_weights(self):
        # Two participants per PSU, unequal weights and PSU labels reused across strata.
        data = pd.DataFrame(
            [(s, psu, w) for s, count in [("a", 3), ("b", 4)]
             for psu in range(count) for w in [1.0, 3.0]],
            columns=["stratum", "psu", "analysis_weight"],
        )
        self.assertEqual((B, SEED), (2000, 20261108))
        rng = np.random.default_rng(SEED)
        saw_repeat = False
        for weights in replicate_weights(data, B=32):
            ratios = weights / data["analysis_weight"].to_numpy()
            for stratum, n in [("a", 3), ("b", 4)]:
                indices = data["stratum"].eq(stratum)
                observed = ratios[indices].reshape(n, 2)
                draws = rng.integers(0, n, n - 1)
                counts = np.bincount(draws, minlength=n)
                expected = counts * n / (n - 1)
                np.testing.assert_allclose(observed, np.repeat(expected[:, None], 2, axis=1))
                self.assertEqual(int(counts.sum()), n - 1)
                saw_repeat |= bool(np.any(counts > 1))
        self.assertTrue(saw_repeat)

    def test_modeling_cli_refuses_legacy_before_config_or_output(self):
        exe = Path(sysconfig.get_path("scripts")) / ("nhanes-model.exe" if sys.platform == "win32" else "nhanes-model")
        invocations = [
            ["temporal-evaluate"], ["temporal-evaluate", "--resume-interrupted"],
            ["nested-validate"], ["nested-validate", "--no-bootstrap"],
            ["nested-validate", "--smoke"], ["run-development"], ["fit-final"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            for args in invocations:
                with self.subTest(args=args):
                    result = subprocess.run(
                        [str(exe), *args, "--config", "missing.yaml"],
                        cwd=directory, capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("HISTORICAL / SUPERSEDED 500-replicate unrescaled PSU", result.stderr)
                    self.assertIn("2,000-replicate rescaled Phase B", result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(list(Path(directory).iterdir()), [])
