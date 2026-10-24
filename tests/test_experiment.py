"""Integration checks for the trained model and experiment metrics."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class ExperimentTests(unittest.TestCase):
    def test_experiment_is_reproducible_and_checkpoint_is_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, str(ROOT / "train.py")],
                           cwd=folder, check=True, capture_output=True, text=True,
                           timeout=180)
            artifacts = Path(folder) / "artifacts"
            data = json.loads((artifacts / "metrics.json").read_text())
            self.assertEqual(data["best_hidden_width"], min(
                data["candidates"], key=lambda x: x["validation_mse"])["hidden_width"])
            self.assertEqual(len(data["candidates"]), 3)
            self.assertGreaterEqual(data["test_mse"], 0)
            import numpy as np
            with np.load(artifacts / "best_weights.npz") as weights:
                self.assertEqual(weights["W1"].shape,
                                 (2, data["best_hidden_width"]))
                self.assertTrue(all(np.isfinite(weights[k]).all()
                                    for k in ("W1", "b1", "W2", "b2")))

if __name__ == "__main__":
    unittest.main()
