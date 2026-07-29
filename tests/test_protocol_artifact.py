import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

from lfm1b_protocol import (candidate_rows_for_policy,
                            graph_input_from_protocol,
                            load_protocol_artifact,
                            prepare_protocol_artifact,
                            save_protocol_artifact,
                            validate_protocol_artifact)
from lfm1b_protocol.artifacts import ArtifactIntegrityError, save_bundle
from lfm1b_protocol.canonical import sha256
from lfm1b_protocol.models import Event


def events():
    rows = []
    for user_id in range(1, 7):
        for timestamp in (1, 2, 3):
            item = 10 + ((user_id + timestamp - 2) % 6)
            rows.append(Event(user_id, item, item + 100, item + 1000, timestamp))
    return rows


def rehash(artifact):
    identity = {key: value for key, value in artifact.items()
                if key not in ("protocol_hash", "provenance", "created_at",
                               "created_utc")}
    artifact["protocol_hash"] = sha256(identity)


def rehash_config(artifact):
    artifact["config_hash"] = sha256(artifact["config"])
    rehash(artifact)


class ProtocolArtifactTests(unittest.TestCase):
    def test_schema_hashes_targets_and_both_candidate_policies(self):
        artifact = prepare_protocol_artifact(
            events(), sampled_negatives=2, seed=7,
            sources=({"bytes": 123, "sha256": "a" * 64},))
        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(set(artifact["targets"]), {"artist", "album", "track"})
        self.assertEqual(artifact["config"]["catalog_policy"], "train_observed")
        self.assertEqual(artifact["config"]["sampled_negatives"], 2)
        self.assertEqual(artifact["config"]["seed"], 7)
        target = artifact["targets"]["artist"]
        self.assertEqual(set(target["splits"]), {"train", "validation", "test"})
        self.assertEqual(
            set(target["splits"]["train"][0]),
            {"user_id", "item_id", "play_count", "first_timestamp",
             "last_timestamp"})
        fixed = candidate_rows_for_policy(
            artifact, "artist", "test", "fixed_sampled")
        full = candidate_rows_for_policy(
            artifact, "artist", "test", "full_catalog")
        self.assertTrue(all("candidate_hash" in row for row in fixed))
        self.assertTrue(all(len(row["candidate_item_ids"]) == 3 for row in fixed))
        self.assertTrue(all(len(row["candidate_item_ids"]) > 3 for row in full))

    def test_warm_catalog_excludes_and_records_cold_heldout(self):
        cold_events = [Event(1, 10, 100, 1000, 1),
                       Event(1, 11, 101, 1001, 2),
                       Event(1, 12, 102, 1002, 3)]
        warm = prepare_protocol_artifact(cold_events)
        artist = warm["targets"]["artist"]
        self.assertEqual(artist["catalog"], (10,))
        self.assertEqual(artist["splits"]["validation"], ())
        stats = warm["statistics"]["targets"]["artist"]["cold_start_exclusions"]
        self.assertEqual(stats["validation"]["excluded_positive_pairs"], 1)
        self.assertEqual(stats["test"]["excluded_user_ids"], (1,))
        transductive = prepare_protocol_artifact(
            cold_events, catalog_policy="all_mapped")
        self.assertEqual(transductive["targets"]["artist"]["catalog"],
                         (10, 11, 12))
        self.assertTrue(transductive["config"]["catalog_uses_future_information"])

    def test_fixed_candidates_persist_and_are_order_deterministic(self):
        first = prepare_protocol_artifact(events(), sampled_negatives=2, seed=9)
        second = prepare_protocol_artifact(reversed(events()), sampled_negatives=2,
                                           seed=9)
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            save_protocol_artifact(directory, first)
            loaded = load_protocol_artifact(directory)
            self.assertEqual(
                loaded["targets"]["track"]["candidates"],
                json.loads(json.dumps(first["targets"]["track"]["candidates"])))

    def test_source_and_candidate_tampering_change_or_invalidate_identity(self):
        first = prepare_protocol_artifact(events(), sources=({"sha256": "a" * 64},))
        second = prepare_protocol_artifact(events(), sources=({"sha256": "b" * 64},))
        self.assertNotEqual(first["protocol_hash"], second["protocol_hash"])
        tampered = copy.deepcopy(first)
        row = tampered["targets"]["artist"]["candidates"]["test"][0]
        row["candidate_item_ids"] = tuple(row["candidate_item_ids"]) + (999,)
        with tempfile.TemporaryDirectory() as directory:
            save_bundle(directory, {"protocol": tampered})
            with self.assertRaises(ArtifactIntegrityError):
                load_protocol_artifact(directory)

    def test_source_path_is_not_identity(self):
        first = prepare_protocol_artifact(
            events(), sources=({"path": "/one/events.dat", "bytes": 7,
                                "sha256": "a" * 64},))
        second = prepare_protocol_artifact(
            events(), sources=({"path": "/two/events.dat", "bytes": 7,
                                "sha256": "a" * 64},))
        self.assertEqual(first["protocol_hash"], second["protocol_hash"])

    def test_semantic_tampering_is_rejected_after_rehash(self):
        artifact = prepare_protocol_artifact(events(), sampled_negatives=2)
        bad_count = copy.deepcopy(artifact)
        bad_count["targets"]["artist"]["splits"]["train"][0]["play_count"] = 0
        rehash(bad_count)
        with self.assertRaises(ArtifactIntegrityError):
            validate_protocol_artifact(bad_count)

        known_negative = copy.deepcopy(artifact)
        target = known_negative["targets"]["artist"]
        row = target["candidates"]["test"][0]
        known = next(value["item_id"] for value in target["known_positives"]
                     if value["user_id"] == row["user_id"] and
                     value["item_id"] not in row["positive_item_ids"])
        row["candidate_item_ids"] = tuple(sorted(
            set(row["candidate_item_ids"]).union((known,))))
        row["candidate_hash"] = sha256(row["candidate_item_ids"])
        rehash(known_negative)
        with self.assertRaises(ArtifactIntegrityError):
            validate_protocol_artifact(known_negative)

    def test_global_timestamp_config_and_statistics_tampering_is_rejected(self):
        spaced = [Event(row.user_id, row.artist_id, row.album_id, row.track_id,
                        row.timestamp * 10) for row in events()]
        artifact = prepare_protocol_artifact(spaced, sampled_negatives=2)
        boundary = copy.deepcopy(artifact)
        row = boundary["targets"]["album"]["splits"]["validation"][0]
        row["first_timestamp"] += 1
        row["last_timestamp"] += 1
        rehash(boundary)
        with self.assertRaisesRegex(ArtifactIntegrityError, "timestamp boundaries"):
            validate_protocol_artifact(boundary)

        config = copy.deepcopy(artifact)
        config["config"]["candidate_policies"]["fixed_sampled"]["seed"] += 1
        rehash_config(config)
        with self.assertRaisesRegex(ArtifactIntegrityError, "policy config"):
            validate_protocol_artifact(config)

        statistics = copy.deepcopy(artifact)
        cold = statistics["statistics"]["targets"]["artist"][
            "cold_start_exclusions"]["test"]
        cold["excluded_users"] = 1
        cold["excluded_positive_pairs"] = 1
        rehash(statistics)
        with self.assertRaisesRegex(ArtifactIntegrityError, "cold-start statistics"):
            validate_protocol_artifact(statistics)

    def test_graph_input_covers_known_ids_with_contiguous_mappings(self):
        artifact = prepare_protocol_artifact(events(), sampled_negatives=2)
        graph = graph_input_from_protocol(artifact)
        self.assertEqual(graph["static_edges"], [])
        for node_type, mapping in graph["node_mappings"].items():
            self.assertEqual(sorted(mapping.values()),
                             list(range(graph["node_counts"][node_type])))
        known_artists = {str(row["item_id"]) for row in
                         artifact["targets"]["artist"]["known_positives"]}
        self.assertTrue(known_artists.issubset(graph["node_mappings"]["artist"]))

    def test_warm_graph_input_excludes_insufficient_users_and_future_items(self):
        rows = [
            Event(1, 10, 100, 1000, 1), Event(1, 11, 101, 1001, 2),
            Event(1, 12, 102, 1002, 3), Event(2, 11, 101, 1001, 1),
            Event(2, 12, 102, 1002, 2), Event(2, 10, 100, 1000, 3),
            Event(9, 99, 199, 1999, 1), Event(9, 98, 198, 1998, 2),
        ]
        warm = graph_input_from_protocol(prepare_protocol_artifact(rows))
        self.assertEqual(set(warm["node_mappings"]["user"]), {"1", "2"})
        self.assertEqual(set(warm["node_mappings"]["artist"]), {"10", "11"})
        self.assertNotIn("12", warm["node_mappings"]["artist"])
        self.assertNotIn("99", warm["node_mappings"]["artist"])
        transductive = graph_input_from_protocol(prepare_protocol_artifact(
            rows, catalog_policy="all_mapped"))
        self.assertIn("9", transductive["node_mappings"]["user"])
        self.assertIn("12", transductive["node_mappings"]["artist"])
        self.assertIn("99", transductive["node_mappings"]["artist"])

    def test_cli_prepare_verify_and_both_baselines(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fixture = os.path.join(root, "examples", "tiny_events.dat")
        with tempfile.TemporaryDirectory() as directory:
            prepare = subprocess.run(
                [sys.executable, "-m", "lfm1b_protocol.cli", "prepare", fixture,
                 directory, "--sampled-negatives", "2", "--seed", "11"],
                cwd=root, check=True, capture_output=True, text=True)
            self.assertIn("protocol_hash", json.loads(prepare.stdout))
            verify = subprocess.run(
                [sys.executable, "-m", "lfm1b_protocol.cli", "verify", directory],
                cwd=root, check=True, capture_output=True, text=True)
            self.assertIn("verified protocol", verify.stdout)
            prepared = load_protocol_artifact(directory)
            pinned = subprocess.run(
                [sys.executable, "-m", "lfm1b_protocol.cli", "verify", directory,
                 "--expected-protocol-hash", prepared["protocol_hash"],
                 "--expected-config-hash", prepared["config_hash"]],
                cwd=root, check=True, capture_output=True, text=True)
            self.assertIn("verified protocol", pinned.stdout)
            rejected = subprocess.run(
                [sys.executable, "-m", "lfm1b_protocol.cli", "verify", directory,
                 "--expected-protocol-hash", "0" * 64],
                cwd=root, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            for policy in ("fixed_sampled", "full_catalog"):
                baseline = subprocess.run(
                    [sys.executable, "-m", "lfm1b_protocol.cli", "baseline",
                     directory, "--item-type", "artist", "--policy", policy],
                    cwd=root, check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(baseline.stdout)["policy"], policy)
            graph_path = os.path.join(directory, "graph-input.json")
            subprocess.run(
                [sys.executable, "-m", "lfm1b_protocol.cli", "graph-input",
                 directory, graph_path], cwd=root, check=True,
                capture_output=True, text=True)
            with open(graph_path, "r") as source:
                graph = json.load(source)
            self.assertEqual(graph["static_edges"], [])


if __name__ == "__main__":
    unittest.main()
