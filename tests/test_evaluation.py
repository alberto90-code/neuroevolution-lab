"""Tests for validation-only, deterministic model selection."""
import unittest
from neuroevolution_ai.evaluation import Candidate, select_candidate


class EvaluationTests(unittest.TestCase):
    def test_validation_only(self):
        a = Candidate("a", 0.8, 0.1, 10)
        b = Candidate("b", 0.7, 1.0, 1)
        self.assertEqual(select_candidate([a, b]), a)

    def test_ties_independent_of_order(self):
        a = Candidate("a", 0.8, 0.2, 10)
        b = Candidate("b", 0.8, 0.9, 1)
        self.assertEqual(select_candidate([b, a]), a)

    def test_reject_invalid_metrics(self):
        for invalid in (float('nan'), float('inf'), -1, True, '0.9'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    select_candidate([Candidate('a', invalid, 0.5, 1)])

    def test_empty_and_duplicate(self):
        with self.assertRaises(ValueError):
            select_candidate([])
        with self.assertRaises(ValueError):
            select_candidate([Candidate('a', 1, 1, 1), Candidate('a', 0, 0, 0)])

    def test_reject_invalid_candidate_names(self):
        for invalid in (None, 42, " ", ""):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    select_candidate([Candidate(invalid, 0.8, 0.1, 1)])

    def test_reject_huge_integer_metrics(self):
        huge = 10 ** 1000
        for position in range(3):
            values = [0.8, 0.1, 1]
            values[position] = huge
            with self.subTest(position=position):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    select_candidate([Candidate("a", *values)])
