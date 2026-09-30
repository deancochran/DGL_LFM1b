import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from lfm1b_protocol.artifacts import (ArtifactIntegrityError, load_bundle, save_bundle,
                                      load_protocol_artifact, prepare_protocol_artifact,
                                      validate_graph_input, validate_protocol_artifact)
from lfm1b_protocol.candidates import build_candidates
from lfm1b_protocol.canonical import canonical_json, sha256
from lfm1b_protocol.metrics import evaluate_rankings
from lfm1b_protocol.models import Event
from lfm1b_protocol.io import read_listening_events_with_provenance
from lfm1b_protocol.split import temporal_split


class HardeningV2Tests(unittest.TestCase):
    def prepare(self, rows, **kwargs):
        kwargs.setdefault("split_strategy", "per_user_last_timestamp_groups")
        return prepare_protocol_artifact(rows, **kwargs)

    def _manifest(self, directory, files):
        identity = {"schema_version": 2, "files": files}
        value = dict(identity, identity_sha256=sha256(identity))
        with open(os.path.join(directory, "manifest.json"), "wb") as stream:
            stream.write(canonical_json(value))

    def test_manifest_rejects_path_and_extra_key(self):
        with tempfile.TemporaryDirectory() as directory:
            self._manifest(directory, {"../x.json": "0" * 64})
            with self.assertRaises(ArtifactIntegrityError): load_bundle(directory)
            self._manifest(directory, {"x.json": "0" * 64})
            with open(os.path.join(directory, "manifest.json"), "r+") as stream:
                value = json.load(stream); value["extra"] = 1; stream.seek(0); json.dump(value, stream); stream.truncate()
            with self.assertRaises(ArtifactIntegrityError): load_bundle(directory)

    def test_manifest_rejects_float_schema_bad_digest_and_files_type(self):
        with tempfile.TemporaryDirectory() as directory:
            for manifest in ({"schema_version": 2.0, "files": {}, "identity_sha256": "0" * 64},
                             {"schema_version": 2, "files": [], "identity_sha256": "0" * 64},
                             {"schema_version": 2, "files": {"x.json": "bad"}, "identity_sha256": "0" * 64}):
                with open(os.path.join(directory, "manifest.json"), "w") as stream: json.dump(manifest, stream)
                with self.assertRaises(ArtifactIntegrityError): load_bundle(directory)

    def test_symlink_fifo_and_size_limit_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "target.json")
            with open(target, "w") as stream: stream.write("{}")
            digest = hashlib.sha256(b"{}").hexdigest()
            self._manifest(directory, {"x.json": digest})
            os.symlink(target, os.path.join(directory, "x.json"))
            with self.assertRaises(ArtifactIntegrityError): load_bundle(directory)
        with tempfile.TemporaryDirectory() as directory:
            save_bundle(directory, {"x": {"a": 1}})
            with self.assertRaises(ArtifactIntegrityError): load_bundle(directory, max_artifact_bytes=1)
        if hasattr(os, "mkfifo"):
            with tempfile.TemporaryDirectory() as directory:
                self._manifest(directory, {"x.json": "0" * 64})
                os.mkfifo(os.path.join(directory, "x.json"))
                with self.assertRaises(ArtifactIntegrityError): load_bundle(directory)

    def test_bundle_limits_cover_stale_fstat_count_and_aggregate_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            save_bundle(directory, {"x": {"value": "0123456789"}})
            original_fstat = os.fstat
            def stale_fstat(descriptor):
                value = original_fstat(descriptor)
                return SimpleNamespace(st_mode=value.st_mode, st_size=0)
            with mock.patch("lfm1b_protocol.artifacts.os.fstat", side_effect=stale_fstat):
                with self.assertRaises(ArtifactIntegrityError):
                    load_bundle(directory, max_artifact_bytes=3)
        with tempfile.TemporaryDirectory() as directory:
            save_bundle(directory, {"first": {"a": 1}, "second": {"b": 2}})
            with self.assertRaises(ArtifactIntegrityError):
                load_bundle(directory, max_artifact_count=1)
            original_read = os.read
            original_fstat = os.fstat
            read_sizes = []
            def bounded_read(descriptor, size):
                read_sizes.append(size)
                return original_read(descriptor, size)
            def stale_fstat(descriptor):
                value = original_fstat(descriptor)
                return SimpleNamespace(st_mode=value.st_mode, st_size=0)
            with mock.patch("lfm1b_protocol.artifacts.os.fstat", side_effect=stale_fstat), \
                    mock.patch("lfm1b_protocol.artifacts.os.read", side_effect=bounded_read):
                with self.assertRaisesRegex(ArtifactIntegrityError, "total artifact size"):
                    load_bundle(directory, max_total_artifact_bytes=10)
            # Each artifact is seven bytes; only three bytes remain for the
            # second artifact, so its fourth byte is the sole detection read.
            self.assertEqual(read_sizes[-1], 4)
            for value in (True, -1):
                with self.assertRaises(ArtifactIntegrityError):
                    load_bundle(directory, max_total_artifact_bytes=value)

    def test_protocol_bundle_rejects_extra_files_before_loading(self):
        artifact = self.prepare([
            Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3)])
        with tempfile.TemporaryDirectory() as directory:
            save_bundle(directory, {"protocol": artifact, "extra": {"ignored": True}})
            with self.assertRaisesRegex(ArtifactIntegrityError, "count limit"):
                load_protocol_artifact(directory)

    def test_sampler_golden_and_fixed_incomplete_known(self):
        candidate = build_candidates(7, "artist", "test", (9,), tuple(range(1, 11)), (),
                                     "fixed_sampled", negatives=3, seed=4)
        self.assertEqual(candidate.items, (5, 7, 8, 9))
        self.assertEqual(len(candidate.items) - 1, 3)

    def test_metrics_reject_candidate_and_popularity_types(self):
        good = {1: [(1, 1.0), (2, 0.0)]}
        for expected in ({1: (1,)}, {1: (1, 1)}, {1: (1, "2")}):
            with self.assertRaises(ValueError):
                evaluate_rankings(good, {1: (1,)}, expected, 1, (1, 2), {})
        with self.assertRaises(ValueError):
            evaluate_rankings({1: [(True, 1.0)]}, {1: (1,)}, {1: (1,)}, 1, (1,), {})
        with self.assertRaises(ValueError):
            evaluate_rankings({1: [(1, 1.0)]}, {1: (1,)}, {1: (1,)}, 1, (1,), {1: -1})

    def test_graph_validator_rejects_counts_mapping_and_duplicate_edges(self):
        graph = {"schema_version": 2, "protocol_hash": "a" * 64,
                 "node_counts": {"user": 1, "artist": 1, "album": 0, "track": 0},
                 "node_mappings": {"user": {"1": 0}, "artist": {"2": 0}, "album": {}, "track": {}},
                 "train_interaction_edges": {"artist": [{"user_id": 0, "item_id": 0, "play_count": 1, "first_timestamp": 1, "last_timestamp": 1}], "album": [], "track": []}, "static_edges": []}
        validate_graph_input(graph)
        for mutate in (lambda g: g["node_counts"].update(user=2),
                       lambda g: g["node_mappings"]["user"].update({"01": 0}),
                       lambda g: g["train_interaction_edges"]["artist"].append(dict(g["train_interaction_edges"]["artist"][0]))):
            import copy
            bad = copy.deepcopy(graph); mutate(bad)
            with self.assertRaises(ArtifactIntegrityError): validate_graph_input(bad)

    def test_cli_requires_strategy_and_records_one_pass_provenance(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        content = b"1\t1\t1\t1\t1\n1\t2\t2\t2\t2\n1\t3\t3\t3\t3\n"
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "events.dat")
            with open(source, "wb") as stream: stream.write(content)
            missing = subprocess.run([sys.executable, "-m", "lfm1b_protocol.cli", "prepare", source, os.path.join(directory, "bad")], cwd=root)
            self.assertNotEqual(missing.returncode, 0)
            output = os.path.join(directory, "out")
            subprocess.run([sys.executable, "-m", "lfm1b_protocol.cli", "prepare", source, output, "--split-strategy", "global_time_cutoffs", "--validation-cutoff", "2", "--test-cutoff", "3"], cwd=root, check=True)
            from lfm1b_protocol.artifacts import load_protocol_artifact
            record = load_protocol_artifact(output)["sources"][0]
            self.assertEqual(record["bytes"], len(content)); self.assertEqual(record["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(tuple(record["column_order"]), ("user_id", "artist_id", "album_id", "track_id", "timestamp"))

    def test_absolute_manifest_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self._manifest(directory, {"/tmp/x.json": "0" * 64})
            with self.assertRaises(ArtifactIntegrityError): load_bundle(directory)

    def test_metrics_rejects_missing_extra_and_duplicate_scored_ids(self):
        for scores in ({1: [(1, 1.0)]}, {1: [(1, 1.0), (2, 0.0), (3, -1.0)]}, {1: [(1, 1.0), (1, 0.0)]}):
            with self.assertRaises(ValueError):
                evaluate_rankings(scores, {1: (1,)}, {1: (1, 2)}, 1, (1, 2, 3), {})

    def test_manifest_path_provenance_is_nonidentity(self):
        from lfm1b_protocol.artifacts import prepare_protocol_artifact
        events = [Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3)]
        source = {"source_kind": "listening_events", "parser_version": "lfm_listening_events_tsv_v1", "parsed_row_count": 3, "bytes": 1, "sha256": "a" * 64, "has_header": False,
                  "column_order": ("user_id", "artist_id", "album_id", "track_id", "timestamp")}
        self.assertEqual(self.prepare(events, sources=(dict(source, path="/a"),))["protocol_hash"],
                         self.prepare(events, sources=(dict(source, path="/b"),))["protocol_hash"])

    def test_provenance_requires_complete_records_and_aligned_paths(self):
        events = [Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3)]
        with self.assertRaises(ValueError):
            self.prepare(events, sources=({"sha256": "a" * 64},))
        source = {"bytes": 1, "sha256": "a" * 64, "has_header": False,
                  "column_order": ("user_id", "artist_id", "album_id", "track_id", "timestamp")}
        source.update({"source_kind": "listening_events", "parser_version": "lfm_listening_events_tsv_v1", "parsed_row_count": 3})
        artifact = self.prepare(events, sources=(source, dict(source, path="/input")))
        self.assertEqual(artifact["provenance"]["source_paths"], (None, "/input"))
        bad = dict(artifact)
        bad["provenance"] = {"source_paths": (None,)}
        with self.assertRaisesRegex(ArtifactIntegrityError, "path provenance"):
            validate_protocol_artifact(bad)
        self.assertEqual(validate_protocol_artifact(self.prepare(events))["sources"], ())

    def test_config_rejects_contradictory_candidate_and_global_metadata(self):
        from lfm1b_protocol.models import Event
        events = [Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3)]
        global_artifact = prepare_protocol_artifact(events, split_strategy="global_time_cutoffs",
                                                    validation_cutoff=2, test_cutoff=3)
        global_artifact["config"]["minimum_timestamp_groups"] = 3
        global_artifact["config_hash"] = sha256(global_artifact["config"])
        global_artifact["protocol_hash"] = sha256({key: value for key, value in global_artifact.items()
                                                    if key not in ("protocol_hash", "provenance", "created_at", "created_utc")})
        with self.assertRaises(ArtifactIntegrityError): validate_protocol_artifact(global_artifact)
        artifact = self.prepare(events)
        artifact["config"]["candidate_policies"]["full_catalog"] = {"materialized": True}
        artifact["config_hash"] = sha256(artifact["config"])
        artifact["protocol_hash"] = sha256({key: value for key, value in artifact.items()
                                             if key not in ("protocol_hash", "provenance", "created_at", "created_utc")})
        with self.assertRaisesRegex(ArtifactIntegrityError, "candidate policy"):
            validate_protocol_artifact(artifact)

    def test_graph_rejects_float_schema(self):
        with self.assertRaises(ArtifactIntegrityError):
            validate_graph_input({"schema_version": 2.0})

    def test_partial_parse_does_not_complete_provenance(self):
        from lfm1b_protocol.io import read_listening_events_with_provenance
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "broken.dat")
            with open(path, "wb") as stream:
                stream.write(b"1\t1\t1\t1\t1\nbroken\n")
            stream = read_listening_events_with_provenance(path, False)
            with self.assertRaises(ValueError): list(stream)
            with self.assertRaises(ValueError): stream.completed_record()

    def test_parser_stream_is_one_shot_and_generic_claims_are_self_attested(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "events.dat")
            with open(path, "w") as output:
                output.write("1\t1\t1\t1\t1\n1\t2\t2\t2\t2\n1\t3\t3\t3\t3\n")
            stream = read_listening_events_with_provenance(path, False)
            next(iter(stream))
            with self.assertRaises(ValueError):
                self.prepare(stream)
            record = {"provenance_binding": "parser_bound", "source_kind": "listening_events",
                      "parser_version": "lfm_listening_events_tsv_v1", "parsed_row_count": 3,
                      "bytes": 30, "sha256": "a" * 64, "has_header": False,
                      "column_order": ("user_id", "artist_id", "album_id", "track_id", "timestamp")}
            with self.assertRaises(ValueError): self.prepare([], sources=(record,))
            with self.assertRaises(ValueError):
                self.prepare(read_listening_events_with_provenance(path, False), sources=())

    def test_public_primitives_and_candidate_index_are_authoritative(self):
        from lfm1b_protocol.split import temporal_split
        from lfm1b_protocol.io import read_listening_events
        for event in (object(), Event(True, 1, 1, 1, 1), Event(1, True, 1, 1, 1),
                      Event(1, 1, 1, 1, True), Event(1, 1, 1, 1, 1, 0)):
            with self.assertRaises(ValueError): temporal_split([event], strategy="per_user_last_timestamp_groups")
        with self.assertRaises(ValueError): list(read_listening_events("unused", has_header=1))
        with self.assertRaises(ValueError): read_listening_events_with_provenance("unused", 0)
        with self.assertRaises(ValueError):
            build_candidates(1, "artist", "test", (2,), (1, 2), ((1, 1),),
                             known_positive_index={1: set()})

    def test_required_strategy_global_statistics_and_boolean_metadata_reject_rehash(self):
        rows = [Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3)]
        with self.assertRaises(TypeError): temporal_split(rows)
        with self.assertRaises(TypeError): prepare_protocol_artifact(rows)
        artifact = prepare_protocol_artifact(rows, split_strategy="global_time_cutoffs", validation_cutoff=2, test_cutoff=3)
        for mutate in (lambda a: a["statistics"].update(insufficient_users=(1,), eligible_users=0),
                       lambda a: a["config"].update(catalog_uses_future_information=0),
                       lambda a: a["config"].update(candidate_filter_uses_future_information=0),
                       lambda a: a["config"].update(globally_time_causal=1)):
            bad = __import__("copy").deepcopy(artifact); mutate(bad)
            bad["config_hash"] = sha256(bad["config"])
            bad["protocol_hash"] = sha256({k: v for k, v in bad.items() if k not in ("protocol_hash", "provenance", "created_at", "created_utc")})
            with self.assertRaises(ArtifactIntegrityError): validate_protocol_artifact(bad)

    def test_raw_rows_and_cold_summary_are_reconstructed(self):
        rows = [Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3),
                Event(2, 1, 4, 4, 1), Event(2, 2, 5, 5, 2), Event(2, 3, 6, 6, 2), Event(2, 4, 7, 7, 3)]
        artifact = prepare_protocol_artifact(rows, split_strategy="global_time_cutoffs", validation_cutoff=2, test_cutoff=3)
        cold = artifact["statistics"]["targets"]["artist"]["cold_start_exclusions"]["validation"]
        self.assertEqual(cold["unseen_training_catalog_item_only"]["excluded_user_ids"], (1, 2))
        # One user has two exclusions; summaries must remain unique and validator-derived.
        self.assertEqual(cold["unseen_training_catalog_item_only"]["excluded_users"], 2)
        self.assertEqual(cold["unseen_training_catalog_item_only"]["excluded_positive_pairs"], 3)
        for mutate in (lambda a: a["targets"]["artist"]["splits"]["train"][0].update(play_count=99),
                       lambda a: a["targets"]["artist"]["splits"]["train"][0].update(first_timestamp=0),
                       lambda a: a["statistics"]["targets"]["artist"]["cold_start_exclusions"]["validation"].update(excluded_user_ids=(1, 1)),
                       lambda a: a["statistics"]["targets"]["artist"]["cold_start_exclusions"]["validation"]["unseen_training_catalog_item_only"].update(excluded_user_ids=(1, 1))):
            bad = __import__("copy").deepcopy(artifact); mutate(bad)
            bad["protocol_hash"] = sha256({k: v for k, v in bad.items() if k not in ("protocol_hash", "provenance", "created_at", "created_utc")})
            with self.assertRaises(ArtifactIntegrityError): validate_protocol_artifact(bad)

    def test_per_user_raw_boundaries_cover_filtered_rows(self):
        rows = [Event(1, 1, 10, 100, 1), Event(1, 1, 10, 100, 2),
                Event(1, 2, 11, 101, 3)]
        artifact = self.prepare(rows)
        self.assertFalse(artifact["targets"]["artist"]["splits"]["validation"])
        for mutate in (
                lambda a: a["targets"]["artist"]["raw_splits"]["validation"][0].update(last_timestamp=3),
                lambda a: a["targets"]["album"]["raw_splits"]["validation"][0].update(first_timestamp=3, last_timestamp=3)):
            bad = __import__("copy").deepcopy(artifact); mutate(bad)
            bad["protocol_hash"] = sha256({k: v for k, v in bad.items()
                                            if k not in ("protocol_hash", "provenance", "created_at", "created_utc")})
            with self.assertRaises(ArtifactIntegrityError): validate_protocol_artifact(bad)

    def test_insufficient_user_ids_cannot_name_a_split_user(self):
        artifact = self.prepare([
            Event(1, 1, 1, 1, 1), Event(1, 2, 2, 2, 2), Event(1, 3, 3, 3, 3),
            Event(2, 4, 4, 4, 1), Event(2, 5, 5, 5, 2)])
        self.assertEqual(artifact["statistics"]["insufficient_users"], [2])
        artifact["statistics"]["insufficient_users"] = [1]
        artifact["protocol_hash"] = sha256({
            key: value for key, value in artifact.items()
            if key not in ("protocol_hash", "provenance", "created_at", "created_utc")})
        with self.assertRaisesRegex(ArtifactIntegrityError, "cannot appear"):
            validate_protocol_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
