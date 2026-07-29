import math
import unittest

from lfm1b_protocol.metrics import evaluate_rankings


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics_ties_coverage_and_novelty(self):
        scores = {1: [(2, 0.5), (1, 0.5), (3, 0.1)],
                  2: [(2, 1.0), (3, 0.0)]}
        result = evaluate_rankings(scores, {1: (1, 3), 2: (2,)}, 2,
                                   (1, 2, 3), {1: 3, 2: 1, 3: 0})
        self.assertAlmostEqual(result.recall, 0.75)
        user_one_ndcg = 1.0 / (1.0 + 1.0 / math.log2(3.0))
        self.assertAlmostEqual(result.ndcg, (user_one_ndcg + 1.0) / 2.0)
        self.assertAlmostEqual(result.mrr, 1.0)
        self.assertAlmostEqual(result.mean_average_precision, 0.75)
        self.assertAlmostEqual(result.catalog_coverage, 1.0)
        expected = (-math.log2(4.0 / 7.0) - math.log2(2.0 / 7.0)
                    - math.log2(2.0 / 7.0) - math.log2(1.0 / 7.0)) / 4.0
        self.assertAlmostEqual(result.novelty, expected)

    def test_k_larger_than_candidates_and_invalid_empty(self):
        result = evaluate_rankings({1: [(1, 0.0)]}, {1: (1,)}, 10,
                                   (1,), {})
        self.assertEqual(result.recall, 1.0)
        with self.assertRaises(ValueError):
            evaluate_rankings({}, {}, 1, (1,), {})
        for score in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                evaluate_rankings({1: [(1, score)]}, {1: (1,)}, 1, (1,), {})
        with self.assertRaisesRegex(ValueError, "catalog"):
            evaluate_rankings({1: [(1, 1.0), (999, -1.0)]}, {1: (1,)}, 1,
                              (1,), {})


if __name__ == "__main__":
    unittest.main()
