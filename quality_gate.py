"""Validate model selection quality without using held-out test error as a gate."""
import json
import math
from pathlib import Path


def check_metrics(metrics, limit=0.02):
    """Return (validation_mse, test_mse) or raise ValueError.

    Validation data controls acceptance; the held-out test metric is
    checked for validity but never used to accept/reject model quality.
    """
    validation = float(metrics["best_validation_mse"])
    test = float(metrics["test_mse"])
    if not all(math.isfinite(v) and v >= 0 for v in (validation, test)):
        raise ValueError("Error metrics must be finite and nonnegative")
    if validation > limit:
        raise ValueError(f"Validation MSE {validation:.6f} exceeds {limit:.6f}")
    return validation, test


if __name__ == "__main__":
    metrics = json.loads(Path("artifacts/metrics.json").read_text())
    validation, test = check_metrics(metrics)
    print(f"Validation MSE: {validation:.6f}; held-out test MSE (report only): {test:.6f}")
    print("Validation quality gate passed")
