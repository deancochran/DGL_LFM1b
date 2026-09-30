import math
import sys
import unittest

from lfm1b_protocol.metrics import (aggregate_ranking_contributions,
                                    evaluate_ranking_contributions,
                                    evaluate_rankings)


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics_ties_coverage_and_novelty(self):
        scores = {1: [(2, 0.5), (1, 0.5), (3, 0.1)],
                  2: [(2, 1.0), (3, 0.0)]}
        result = evaluate_rankings(scores, {1: (1, 3), 2: (2,)}, {1: (1, 2, 3), 2: (2, 3)}, 2,
                                   (1, 2, 3), {1: 3, 2: 1, 3: 0})
        self.assertAlmostEqual(result.recall, 0.75)
        user_one_ndcg = 1.0 / (1.0 + 1.0 / math.log2(3.0))
        self.assertAlmostEqual(result.ndcg, (user_one_ndcg + 1.0) / 2.0)
        self.assertAlmostEqual(result.mrr, 1.0)
        self.assertAlmostEqual(result.mean_average_precision, 0.75)
        self.assertAlmostEqual(result.precision, 0.5)
        self.assertAlmostEqual(result.hit_rate, 1.0)
        self.assertAlmostEqual(result.catalog_coverage, 1.0)
        expected = (-math.log2(4.0 / 7.0) - math.log2(2.0 / 7.0)
                    - math.log2(2.0 / 7.0) - math.log2(1.0 / 7.0)) / 4.0
        self.assertAlmostEqual(result.novelty, expected)
        contributions = evaluate_ranking_contributions(
            scores, {1: (1, 3), 2: (2,)}, {1: (1, 2, 3), 2: (2, 3)}, 2,
            (1, 2, 3), {1: 3, 2: 1, 3: 0})
        self.assertEqual(tuple(value.user_id for value in contributions), (1, 2))
        self.assertTrue(all(value.k == 2 for value in contributions))
        self.assertEqual(contributions[0].recommended_item_ids, (1, 2))
        self.assertEqual(contributions[0].relevant_items, 2)
        self.assertEqual(contributions[0].candidate_items, 3)
        self.assertEqual(contributions[0].recommended_items, 2)
        self.assertEqual(contributions[0].hit_items, 1)
        self.assertEqual(contributions[0].hit_ranks, (1,))
        self.assertAlmostEqual(
            result.recall,
            sum(value.recall for value in contributions) / len(contributions))
        self.assertAlmostEqual(
            result.ndcg,
            sum(value.ndcg for value in contributions) / len(contributions))
        self.assertEqual(
            aggregate_ranking_contributions(
                contributions, 2, (1, 2, 3), {1: 3, 2: 1, 3: 0}), result)

    def test_k_larger_than_candidates_and_invalid_empty(self):
        result = evaluate_rankings({1: [(1, 0.0)]}, {1: (1,)}, {1: (1,)}, 10,
                                   (1,), {})
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 0.1)
        with self.assertRaises(ValueError):
            evaluate_rankings({}, {}, {}, 1, (1,), {})
        for score in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                evaluate_rankings({1: [(1, score)]}, {1: (1,)}, {1: (1,)}, 1, (1,), {})
        with self.assertRaisesRegex(ValueError, "catalog"):
            evaluate_rankings({1: [(1, 1.0), (999, -1.0)]}, {1: (1,)}, {1: (1, 999)}, 1,
                              (1,), {})
        with self.assertRaisesRegex(ValueError, "signed 63-bit"):
            evaluate_rankings({1: [(1, 1.0)]}, {1: (1,)}, {1: (1,)},
                              1 << 80, (1,), {})
        with self.assertRaisesRegex(ValueError, "finite novelty denominator"):
            evaluate_rankings({1: [(1, 1.0)]}, {1: (1,)}, {1: (1,)}, 1,
                              (1,), {1: 10 ** 400})
        with self.assertRaisesRegex(ValueError, "finite novelty denominator"):
            evaluate_ranking_contributions(
                {1: [(1, 1.0), (2, 0.0)]}, {1: (1,)}, {1: (1, 2)}, 1,
                (1, 2), {},
                smoothing=sys.float_info.max)
        with self.assertRaisesRegex(ValueError, "smoothing"):
            evaluate_rankings({1: [(1, 1.0)]}, {1: (1,)}, {1: (1,)},
                              1, (1,), {}, smoothing=10 ** 400)
        with self.assertRaisesRegex(ValueError, "smoothing"):
            aggregate_ranking_contributions((), 1, (1,), {},
                                            smoothing=10 ** 400)


if __name__ == "__main__":
    unittest.main()
