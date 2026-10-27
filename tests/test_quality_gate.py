"""Unit tests for validation-only model quality gating."""
import unittest
from quality_gate import check_metrics


class QualityGateTests(unittest.TestCase):
    def test_rejects_malformed_metric_documents(self):
        for metrics in (None, [], "metrics", 42, True, {},
                        {"best_validation_mse": 0.01}, {"test_mse": 0.5}):
            with self.subTest(metrics=metrics):
                with self.assertRaisesRegex(ValueError, "must be an object containing"):
                    check_metrics(metrics)

    def test_accepts_valid_metrics(self):
        self.assertEqual(
            check_metrics({"best_validation_mse": 0.01, "test_mse": 0.5}),
            (0.01, 0.5),
        )

    def test_rejects_bad_validation_error(self):
        with self.assertRaises(ValueError):
            check_metrics({"best_validation_mse": 0.03, "test_mse": 0.001})

    def test_rejects_nonfinite_or_negative_errors(self):
        for validation, test in [(float("nan"), 0.01),
                                 (0.01, float("inf")),
                                 (-0.1, 0.01),
                                 (0.01, -0.1)]:
            with self.subTest(validation=validation, test=test):
                with self.assertRaises(ValueError):
                    check_metrics({"best_validation_mse": validation, "test_mse": test})


    def test_rejects_invalid_quality_thresholds(self):
        metrics = {"best_validation_mse": 0.01, "test_mse": 0.5}
        for limit in (float("nan"), float("inf"), float("-inf"), -0.01,
                      False, True, "0.02", None):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    check_metrics(metrics, limit=limit)

    def test_zero_quality_threshold_is_valid(self):
        self.assertEqual(
            check_metrics({"best_validation_mse": 0.0, "test_mse": 0.5}, limit=0),
            (0.0, 0.5),
        )

    def test_rejects_non_numeric_json_values(self):
        for key in ("best_validation_mse", "test_mse"):
            for value in (False, True, "0.01", None, [], {}):
                with self.subTest(metric=key, value=value):
                    metrics = {"best_validation_mse": 0.01, "test_mse": 0.5}
                    metrics[key] = value
                    with self.assertRaises(ValueError):
                        check_metrics(metrics)


    def test_rejects_oversized_integer_values(self):
        """JSON integers too large for float must fail closed with ValueError."""
        oversized = 10 ** 400
        metrics = {"best_validation_mse": 0.01, "test_mse": 0.5}
        with self.assertRaises(ValueError):
            check_metrics(metrics, limit=oversized)
        for key in metrics:
            with self.subTest(metric=key):
                invalid = dict(metrics)
                invalid[key] = oversized
                with self.assertRaises(ValueError):
                    check_metrics(invalid)


    def test_adjacent_float_above_limit_reports_exact_value(self):
        """A just-failing metric must not look equal to its threshold."""
        import math
        limit = 0.02
        validation = math.nextafter(limit, math.inf)
        with self.assertRaises(ValueError) as caught:
            check_metrics({"best_validation_mse": validation, "test_mse": 0.001},
                          limit=limit)
        self.assertIn(repr(validation), str(caught.exception))
        self.assertIn(repr(limit), str(caught.exception))

    def test_near_threshold_failure_shows_actual_values(self):
        with self.assertRaisesRegex(ValueError, "0[.]0200001 exceeds 0[.]02"):
            check_metrics({"best_validation_mse": 0.0200001, "test_mse": 0.001},
                          limit=0.02)


if __name__ == "__main__":
    unittest.main()
