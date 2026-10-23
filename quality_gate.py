"""Fail CI when held-out test error exceeds the agreed quality limit."""
import json
from pathlib import Path

metrics = json.loads(Path("artifacts/metrics.json").read_text())
limit = 0.02
test = float(metrics["test_mse"])
validation = float(metrics["best_validation_mse"])
print(f"Validation MSE: {validation:.6f}; test MSE: {test:.6f}; limit: {limit:.6f}")
if not (0 <= test <= limit):
    raise SystemExit("Model quality gate failed")
print("Model quality gate passed")
