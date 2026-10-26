"""Unit tests for validation-only model quality gating."""
import unittest
from quality_gate import check_metrics


class QualityGateTests(unittest.TestCase):
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


    def test_rejects_non_numeric_json_values(self):
        for key in ("best_validation_mse", "test_mse"):
            for value in (False, True, "0.01", None, [], {}):
                with self.subTest(metric=key, value=value):
                    metrics = {"best_validation_mse": 0.01, "test_mse": 0.5}
                    metrics[key] = value
                    with self.assertRaises(ValueError):
                        check_metrics(metrics)


if __name__ == "__main__":
    unittest.main()
