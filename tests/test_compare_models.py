"""Regression tests for validation-only comparison."""
import unittest

from compare_models import is_improvement


def metrics(validation, test=0.1):
    return {"best_validation_mse": validation, "test_mse": test}


class ComparisonTests(unittest.TestCase):
    def test_meaningful_improvement(self):
        self.assertTrue(is_improvement(metrics(0.01, 0.001), metrics(0.009, 100)))
        self.assertFalse(is_improvement(metrics(0.01, 100), metrics(0.00995, 0)))

    def test_test_error_never_selects_model(self):
        self.assertFalse(is_improvement(metrics(0.01, 100), metrics(0.02, 0)))
        self.assertTrue(is_improvement(metrics(0.01, 0), metrics(0.008, 100)))

    def test_zero_baseline_and_equal_scores(self):
        self.assertFalse(is_improvement(metrics(0), metrics(0)))
        self.assertFalse(is_improvement(metrics(0.01), metrics(0.01)))

    def test_invalid_metrics_and_gain_fail_closed(self):
        for candidate in (metrics(-1), metrics(float("nan")), metrics(0.01, float("inf")), {}):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                is_improvement(metrics(0.01), candidate)
        for gain in (-0.1, 1, float("nan"), True, "0.01"):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                is_improvement(metrics(0.01), metrics(0.009), gain)


if __name__ == "__main__":
    unittest.main()
