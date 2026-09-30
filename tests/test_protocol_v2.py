import copy
import hashlib
import os
import tempfile
import unittest

from lfm1b_protocol.artifacts import (ArtifactIntegrityError, candidate_rows_for_policy, graph_input_from_protocol,
                                      prepare_protocol_artifact, validate_protocol_artifact)
from lfm1b_protocol.candidates import build_candidates
from lfm1b_protocol.models import Event
from lfm1b_protocol.split import temporal_split
from lfm1b_protocol.io import read_listening_events_with_provenance


class ProtocolV2Tests(unittest.TestCase):
    def test_global_cutoffs_boundaries_and_missing_optional_target(self):
        rows = [Event(1, 1, None, None, 1), Event(1, None, 2, None, 2),
                Event(1, 3, None, 3, 3), Event(2, 1, 1, 1, 1),
                Event(2, 2, 2, 2, 2), Event(2, 3, 3, 3, 3)]
        artifact = prepare_protocol_artifact(rows, split_strategy="global_time_cutoffs",
                                             validation_cutoff=2, test_cutoff=3)
        self.assertEqual(artifact["statistics"]["eligible_users"], 2)
        validate_protocol_artifact(artifact)
        with self.assertRaises(ValueError):
            temporal_split(rows, strategy="global_time_cutoffs", validation_cutoff=3, test_cutoff=3)
        with self.assertRaises(ValueError):
            temporal_split(rows, strategy="per_user_last_timestamp_groups", validation_cutoff=2, test_cutoff=3)

    def test_repeat_statistics_do_not_become_cold_statistics(self):
        rows = [Event(1, 1, 1, 1, 1), Event(1, 1, 1, 1, 2), Event(1, 1, 1, 1, 3),
                Event(2, 2, 2, 2, 1), Event(2, 3, 3, 3, 2), Event(2, 4, 4, 4, 3)]
        artifact = prepare_protocol_artifact(rows, split_strategy="per_user_last_timestamp_groups")
        self.assertEqual(artifact["statistics"]["repeated_validation_pairs_excluded"], 3)
        cold = artifact["statistics"]["targets"]["artist"]["cold_start_exclusions"]
        self.assertEqual(cold["validation"]["excluded_positive_pairs"], 1)
        validate_protocol_artifact(artifact)
        repeated = prepare_protocol_artifact(rows, split_strategy="per_user_last_timestamp_groups", repeat_policy="repeat_allowed")
        self.assertEqual(repeated["statistics"]["repeated_validation_pairs_excluded"], 0)
        self.assertTrue(repeated["targets"]["artist"]["splits"]["validation"])

    def test_as_of_filter_does_not_hide_future_only_item(self):
        rows = [Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3), Event(1, 1, 1, 1, 3),
                Event(2, 3, 3, 3, 1), Event(2, 1, 1, 1, 2), Event(2, 2, 2, 2, 3)]
        as_of = prepare_protocol_artifact(rows, catalog_policy="all_mapped", sampled_negatives=10, split_strategy="per_user_last_timestamp_groups")
        all_seen = prepare_protocol_artifact(rows, catalog_policy="all_mapped", sampled_negatives=10,
                                               positive_filter_horizon="all_observed", split_strategy="per_user_last_timestamp_groups")
        user = next(row for row in as_of["targets"]["artist"]["candidates"]["validation"] if row["user_id"] == 1)
        future = next(row for row in all_seen["targets"]["artist"]["candidates"]["validation"] if row["user_id"] == 1)
        self.assertIn(3, user["candidate_item_ids"])
        self.assertNotIn(3, future["candidate_item_ids"])

    def test_sampler_excludes_direct_positive_with_incomplete_known_snapshot(self):
        candidate = build_candidates(1, "artist", "test", (2,), (1, 2, 3), (), "full_catalog")
        self.assertEqual(candidate.items, (1, 2, 3))
        self.assertEqual(candidate.positives, (2,))

    def test_rehashed_fabricated_snapshot_pair_is_rejected(self):
        artifact = prepare_protocol_artifact([Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3)], split_strategy="per_user_last_timestamp_groups")
        bad = copy.deepcopy(artifact)
        bad["targets"]["artist"]["splits"]["validation"] = ({"user_id": 1, "item_id": 1, "play_count": 1, "first_timestamp": 2, "last_timestamp": 2},)
        from lfm1b_protocol.canonical import sha256
        bad["protocol_hash"] = sha256({key: value for key, value in bad.items() if key not in ("protocol_hash", "provenance", "created_at", "created_utc")})
        with self.assertRaises(ArtifactIntegrityError):
            validate_protocol_artifact(bad)

    def test_graph_edges_are_only_train_edges(self):
        artifact = prepare_protocol_artifact([Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3)], catalog_policy="all_mapped", split_strategy="per_user_last_timestamp_groups")
        graph = graph_input_from_protocol(artifact)
        self.assertEqual(len(graph["train_interaction_edges"]["artist"]), 1)
        self.assertEqual(graph["train_interaction_edges"]["artist"][0]["item_id"],
                         graph["node_mappings"]["artist"]["1"])

    def test_global_cutoffs_keep_all_users_and_are_causal(self):
        rows = [Event(1, 1, 1, 1, 1), Event(2, 2, 2, 2, 1),
                Event(2, 3, 3, 3, 2), Event(2, 4, 4, 4, 3)]
        result = temporal_split(rows, strategy="global_time_cutoffs", validation_cutoff=2,
                                test_cutoff=3)
        self.assertEqual((result.statistics.total_users, result.statistics.eligible_users,
                          result.statistics.insufficient_users), (2, 2, ()))
        self.assertEqual({r.item_id for r in result.raw_split.train if r.user_id == 1}, {1})
        with_future = temporal_split(rows + [Event(1, 9, 9, 9, 4)],
                                     strategy="global_time_cutoffs", validation_cutoff=2,
                                     test_cutoff=3)
        self.assertEqual([r for r in result.raw_split.train if r.user_id == 1],
                         [r for r in with_future.raw_split.train if r.user_id == 1])

    def test_target_warmth_reasons_are_relation_specific(self):
        rows = [Event(1, 1, 10, 100, 1), Event(1, 2, 11, 101, 2), Event(1, 3, 12, 102, 3),
                Event(2, None, 20, 200, 1), Event(2, 1, 21, 201, 2), Event(2, 9, 22, 202, 3),
                Event(3, 2, 30, 300, 1), Event(3, 2, 31, 301, 2), Event(3, 3, 32, 302, 3)]
        artifact = prepare_protocol_artifact(rows, split_strategy="global_time_cutoffs",
                                             validation_cutoff=2, test_cutoff=3)
        cold = artifact["statistics"]["targets"]["artist"]["cold_start_exclusions"]["validation"]
        self.assertEqual(cold["no_same_target_training_history_only"],
                         {"excluded_positive_pairs": 1, "excluded_user_ids": (2,), "excluded_users": 1})
        self.assertEqual(artifact["targets"]["artist"]["splits"]["validation"],
                         ({"user_id": 1, "item_id": 2, "play_count": 1, "first_timestamp": 2, "last_timestamp": 2},))
        transductive = prepare_protocol_artifact(rows, split_strategy="global_time_cutoffs",
                                                  validation_cutoff=2, test_cutoff=3,
                                                  catalog_policy="all_mapped")
        self.assertIn(2, {r["user_id"] for r in transductive["targets"]["artist"]["splits"]["validation"]})

    def test_exact_candidate_exclusion_matrix(self):
        rows = [Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3), Event(1, 1, 1, 1, 3),
                Event(2, 4, 4, 4, 1), Event(2, 5, 5, 5, 2), Event(2, 6, 6, 6, 3)]
        expected = {("novel_only", "as_of_split"): (2, 3, 4, 5, 6),
                    ("novel_only", "all_observed"): (2, 4, 5, 6),
                    ("repeat_allowed", "as_of_split"): (1, 2, 3, 4, 5, 6),
                    ("repeat_allowed", "all_observed"): (1, 2, 4, 5, 6)}
        for (repeat, horizon), items in expected.items():
            artifact = prepare_protocol_artifact(rows, split_strategy="global_time_cutoffs",
                                                 validation_cutoff=2, test_cutoff=3,
                                                 catalog_policy="all_mapped", repeat_policy=repeat,
                                                 positive_filter_horizon=horizon, sampled_negatives=20)
            fixed = next(r for r in artifact["targets"]["artist"]["candidates"]["validation"] if r["user_id"] == 1)
            full = next(r for r in candidate_rows_for_policy(artifact, "artist", "validation", "full_catalog") if r["user_id"] == 1)
            self.assertEqual(fixed["candidate_item_ids"], items)
            self.assertEqual(full["candidate_item_ids"], items)
        for horizon in ("as_of_split", "all_observed"):
            artifact = prepare_protocol_artifact(rows, split_strategy="global_time_cutoffs",
                                                 validation_cutoff=2, test_cutoff=3,
                                                 catalog_policy="all_mapped", repeat_policy="repeat_allowed",
                                                 positive_filter_horizon=horizon, sampled_negatives=20)
            fixed = next(r for r in artifact["targets"]["artist"]["candidates"]["test"] if r["user_id"] == 1)
            full = next(r for r in candidate_rows_for_policy(artifact, "artist", "test", "full_catalog") if r["user_id"] == 1)
            self.assertEqual(fixed["candidate_item_ids"], (1, 2, 3, 4, 5, 6))
            self.assertEqual(full["candidate_item_ids"], fixed["candidate_item_ids"])

    def test_temporal_metadata_cross_product_and_parser_binding(self):
        rows = [Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3)]
        for strategy in ("per_user_last_timestamp_groups", "global_time_cutoffs"):
            for catalog in ("train_observed", "all_mapped"):
                for horizon in ("as_of_split", "all_observed"):
                    kwargs = {"split_strategy": strategy, "catalog_policy": catalog,
                              "positive_filter_horizon": horizon}
                    if strategy == "global_time_cutoffs": kwargs.update(validation_cutoff=2, test_cutoff=3)
                    config = prepare_protocol_artifact(rows, **kwargs)["config"]
                    self.assertEqual(config["globally_time_causal"], strategy == "global_time_cutoffs" and catalog == "train_observed" and horizon == "as_of_split")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "events.dat")
            content = b"1\t1\t1\t1\t1\n1\t2\t2\t2\t2\n1\t3\t3\t3\t3\n"
            with open(path, "wb") as stream: stream.write(content)
            artifact = prepare_protocol_artifact(read_listening_events_with_provenance(path, False),
                                                 split_strategy="global_time_cutoffs", validation_cutoff=2, test_cutoff=3)
            bound = artifact["sources"][0]
            manual = prepare_protocol_artifact(rows, split_strategy="global_time_cutoffs", validation_cutoff=2, test_cutoff=3,
                sources=({"source_kind": "listening_events", "parser_version": "lfm_listening_events_tsv_v1", "parsed_row_count": 3, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "has_header": False, "column_order": ("user_id", "artist_id", "album_id", "track_id", "timestamp")},))
            self.assertEqual(bound["provenance_binding"], "parser_bound")
            self.assertEqual(manual["sources"][0]["provenance_binding"], "self_attested")
            self.assertNotEqual(artifact["protocol_hash"], manual["protocol_hash"])


if __name__ == "__main__":
    unittest.main()
