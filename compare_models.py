"""Compare candidate and baseline metrics without consulting held-out test MSE.

Usage: python compare_models.py baseline.json candidate.json
Exit status 0 means a strictly better candidate; 1 means no improvement.
Malformed input exits with an error.
"""
import argparse
import json
import math
from pathlib import Path

from quality_gate import check_metrics


def is_improvement(baseline, candidate, min_relative_gain=0.01):
    """Require a meaningful validation gain; never select using test MSE."""
    if isinstance(min_relative_gain, bool) or not isinstance(min_relative_gain, (int, float)):
        raise ValueError("Gain must be a finite number in [0, 1)")
    try:
        gain = float(min_relative_gain)
    except OverflowError as exc:
        raise ValueError("Gain must be a finite number in [0, 1)") from exc
    if not math.isfinite(gain) or not 0 <= gain < 1:
        raise ValueError("Gain must be a finite number in [0, 1)")
    base_val, _ = _validated(baseline)
    candidate_val, _ = _validated(candidate)
    # A zero-error baseline cannot be improved.
    return base_val > 0 and candidate_val < base_val and (base_val - candidate_val) / base_val >= gain


def _validated(metrics):
    """Validate both reported MSE values without applying an acceptance threshold."""
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be an object")
    # Use the existing gate with a finite maximum float threshold.
    return check_metrics(metrics, limit=1.7976931348623157e308)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--min-relative-gain", type=float, default=0.01)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    improved = is_improvement(baseline, candidate, args.min_relative_gain)
    print("Validation improvement confirmed" if improved else "No validated improvement")
    return 0 if improved else 1


if __name__ == "__main__":
    raise SystemExit(main())
