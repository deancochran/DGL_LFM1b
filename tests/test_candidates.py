import unittest

from lfm1b_protocol.candidates import build_candidates, candidate_digest


class CandidateTests(unittest.TestCase):
    def test_filtering_determinism_and_no_replacement(self):
        known = ((1, 1), (1, 2), (1, 3), (2, 4))
        first = build_candidates(1, "artist", "test", (3,), range(1, 11),
                                 known, "fixed_sampled", negatives=4, seed=17)
        second = build_candidates(1, "artist", "test", (3,), reversed(range(1, 11)),
                                  reversed(known), "fixed_sampled", negatives=4, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first.items), len(set(first.items)))
        self.assertNotIn(1, first.items)
        self.assertNotIn(2, first.items)
        self.assertIn(3, first.items)
        self.assertEqual(first.candidate_hash, candidate_digest(first.items))

    def test_full_catalog_excludes_other_known_positives(self):
        result = build_candidates(1, "track", "validation", (2,), (1, 2, 3),
                                  ((1, 1), (1, 2)), "full_catalog")
        self.assertEqual(result.items, (2, 3))

    def test_preindexed_sampling_matches_unindexed_semantics(self):
        catalog = tuple(range(100))
        known = tuple((1, item) for item in (1, 2, 3, 4))
        ordinary = build_candidates(
            1, "artist", "test", (4,), catalog, known,
            "fixed_sampled", negatives=10, seed=31)
        indexed = build_candidates(
            1, "artist", "test", (4,), catalog, known,
            "fixed_sampled", negatives=10, seed=31,
            known_positive_index={1: {1, 2, 3, 4}},
            sorted_catalog=catalog, catalog_index=set(catalog))
        self.assertEqual(ordinary, indexed)


if __name__ == "__main__":
    unittest.main()
