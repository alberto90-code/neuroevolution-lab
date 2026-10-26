"""Validate model selection quality without using held-out test error as a gate."""
import json
import math
from pathlib import Path


def check_metrics(metrics, limit=0.02):
    """Return (validation_mse, test_mse) or raise ValueError.

    Validation data controls acceptance; the held-out test metric is
    checked for validity but never used to accept/reject model quality.
    """
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        raise ValueError("Quality threshold must be a finite nonnegative number")
    try:
        limit = float(limit)
    except OverflowError as exc:
        raise ValueError("Quality threshold must be a finite nonnegative number") from exc
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("Quality threshold must be a finite nonnegative number")
    raw_validation = metrics["best_validation_mse"]
    raw_test = metrics["test_mse"]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           for v in (raw_validation, raw_test)):
        raise ValueError("Error metrics must be JSON numbers (not booleans or strings)")
    try:
        validation, test = float(raw_validation), float(raw_test)
    except OverflowError as exc:
        raise ValueError("Error metrics must be finite and nonnegative") from exc
    if not all(math.isfinite(v) and v >= 0 for v in (validation, test)):
        raise ValueError("Error metrics must be finite and nonnegative")
    if validation > limit:
        raise ValueError(f"Validation MSE {validation!r} exceeds {limit!r}")
    return validation, test


if __name__ == "__main__":
    metrics = json.loads(Path("artifacts/metrics.json").read_text())
    validation, test = check_metrics(metrics)
    print(f"Validation MSE: {validation!r}; held-out test MSE (report only): {test!r}")
    print("Validation quality gate passed")
