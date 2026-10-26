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
            self.assertEqual(data["python_version"], __import__("platform").python_version())
            self.assertEqual(data["numpy_version"], __import__("numpy").__version__)
            self.assertEqual(data["dataset_sizes"],
                             {"train": 1200, "validation": 300, "test": 300})
            self.assertEqual(data["best_hidden_width"], min(
                data["candidates"], key=lambda x: x["validation_mse"])["hidden_width"])
            import numpy as np
            self.assertEqual(len(data["candidates"]), 9)
            self.assertEqual(
                {(item["generation"], item["hidden_width"])
                 for item in data["candidates"]},
                {(generation, width)
                 for generation in range(3) for width in (8, 16, 32)},
            )
            self.assertEqual(
                data["best_validation_mse"],
                min(item["validation_mse"] for item in data["candidates"]),
            )
            self.assertTrue(all(
                np.isfinite(item["validation_mse"]) and item["validation_mse"] >= 0
                for item in data["candidates"]
            ))
            self.assertTrue(all(
                "test_mse" not in item for item in data["candidates"]
            ), "Held-out test scores must not be used for candidate selection")
            self.assertEqual(len(json.loads((artifacts / "history.json").read_text())), 9)
            self.assertEqual(data["best_generation"], min(
                data["candidates"], key=lambda x: x["validation_mse"])["generation"])
            self.assertGreaterEqual(data["test_mse"], 0)
            with np.load(artifacts / "best_weights.npz") as weights:
                self.assertEqual(weights["W1"].shape,
                                 (2, data["best_hidden_width"]))
                self.assertTrue(all(np.isfinite(weights[k]).all()
                                    for k in ("W1", "b1", "W2", "b2")))
                rng = np.random.default_rng(data["seed"])
                inputs = rng.uniform(-1, 1, (1800, 2))
                test_inputs = inputs[1500:]
                test_targets = test_inputs[:, :1] * test_inputs[:, 1:2]
                predictions = (np.tanh(test_inputs @ weights["W1"] + weights["b1"])
                               @ weights["W2"] + weights["b2"])
                checkpoint_mse = float(np.mean((predictions - test_targets) ** 2))
                self.assertAlmostEqual(checkpoint_mse, data["test_mse"], places=12)
                validation_inputs = inputs[1200:1500]
                validation_targets = validation_inputs[:, :1] * validation_inputs[:, 1:2]
                validation_predictions = (
                    np.tanh(validation_inputs @ weights["W1"] + weights["b1"])
                    @ weights["W2"] + weights["b2"]
                )
                checkpoint_validation_mse = float(
                    np.mean((validation_predictions - validation_targets) ** 2)
                )
                self.assertAlmostEqual(
                    checkpoint_validation_mse, data["best_validation_mse"], places=12
                )
                self.assertEqual(
                    json.loads((artifacts / "history.json").read_text()),
                    data["candidates"],
                )


    def test_repeated_runs_are_reproducible(self):
        """Identical seeds and code must reproduce metrics and model weights."""
        import numpy as np

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for folder in (first, second):
                subprocess.run(
                    [sys.executable, str(ROOT / "train.py")],
                    cwd=folder, check=True, capture_output=True, text=True,
                    timeout=180,
                )
            left = Path(first) / "artifacts"
            right = Path(second) / "artifacts"
            self.assertEqual(
                json.loads((left / "metrics.json").read_text()),
                json.loads((right / "metrics.json").read_text()),
            )
            self.assertEqual(
                json.loads((left / "history.json").read_text()),
                json.loads((right / "history.json").read_text()),
            )
            with np.load(left / "best_weights.npz") as a, np.load(right / "best_weights.npz") as b:
                self.assertEqual(set(a.files), set(b.files))
                for name in a.files:
                    with self.subTest(weight=name):
                        np.testing.assert_array_equal(a[name], b[name])

if __name__ == "__main__":
    unittest.main()
