import copy
import contextlib
import io
import hashlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from listenbrainz_protocol import (ListenBrainzIntegrityError, evaluate_baseline, graph_input,
    load, load_graph, prepare, save, save_graph, validate, validate_graph)
from listenbrainz_protocol.cli import main as cli_main
from lfm1b_protocol.canonical import sha256

ROOT = os.path.dirname(os.path.dirname(__file__))
FIXTURE = os.path.join(ROOT, "examples", "listenbrainz_synthetic.listens")
ARCHIVE = "a" * 64

class ListenBrainzProtocolTests(unittest.TestCase):
    def make(self, **changes):
        args = dict(dump_id="synthetic", dump_type="full", archive_sha256=ARCHIVE,
            member_path="synthetic/member.listens", track_identity_policy="server_mapped_musicbrainz",
            album_entity="release_group", split_strategy="global_time_cutoffs",
            validation_cutoff=20, test_cutoff=30, catalog_policy="all_mapped", repeat_policy="repeat_allowed",
            sampled_negatives=3)
        args.update(changes)
        return prepare(FIXTURE, **args)

    def rehash_wrapper(self, wrapper, config_changed=False):
        if config_changed:
            wrapper["config_hash"] = sha256(wrapper["config"])
        wrapper["artifact_hash"] = sha256(
            {key: value for key, value in wrapper.items()
             if key not in ("artifact_hash", "provenance")})
        return wrapper

    def rehash_embedded_protocol(self, wrapper):
        protocol = wrapper["protocol"]
        protocol["protocol_hash"] = sha256({
            key: value for key, value in protocol.items()
            if key not in ("protocol_hash", "provenance", "created_at", "created_utc")})
        return self.rehash_wrapper(wrapper)

    def metric_result(self, users=1):
        return SimpleNamespace(
            k=1, users=users, recall=0.0, ndcg=0.0, mrr=0.0,
            mean_average_precision=0.0, precision=0.0, hit_rate=0.0,
            catalog_coverage=0.0, novelty=0.0)

    def test_prepare_identity_and_normalization(self):
        wrapper = self.make()
        self.assertEqual(wrapper["source"]["member_bytes"], os.path.getsize(FIXTURE))
        with open(FIXTURE,"rb") as fixture:
            self.assertEqual(wrapper["source"]["member_sha256"], hashlib.sha256(fixture.read()).hexdigest())
        self.assertNotIn("Alice", repr(wrapper))
        self.assertEqual(wrapper["statistics"]["dropped_missing_track_identity_rows"], 1)
        self.assertEqual(wrapper["statistics"]["multi_artist_rows"], 1)
        self.assertEqual(wrapper["statistics"]["duplicate_event_key_rows"], 1)
        self.assertEqual(wrapper["statistics"]["conflicting_duplicate_event_keys"], 1)
        self.assertGreater(wrapper["statistics"]["missing_artist_rows"], 0)
        self.assertGreater(wrapper["statistics"]["missing_album_rows"], 0)
        self.assertEqual(wrapper["provenance"], {})

    def test_msid_and_reordering_determinism(self):
        mapped = self.make()
        msid = self.make(track_identity_policy="recording_msid")
        self.assertEqual(msid["statistics"]["emitted_events"], 10)
        with open(FIXTURE, "rb") as f: lines = f.readlines()
        fd, path = tempfile.mkstemp(); os.write(fd, b"".join(reversed(lines))); os.close(fd)
        try:
            reordered = prepare(path, dump_id="synthetic", dump_type="full", archive_sha256=ARCHIVE,
                member_path="synthetic/member.listens", track_identity_policy="server_mapped_musicbrainz",
                album_entity="release_group", split_strategy="global_time_cutoffs", validation_cutoff=20,
                test_cutoff=30, catalog_policy="all_mapped", repeat_policy="repeat_allowed", sampled_negatives=3)
        finally: os.unlink(path)
        self.assertEqual(mapped["mappings"], reordered["mappings"])
        self.assertEqual(mapped["statistics"], reordered["statistics"])
        self.assertEqual(mapped["protocol"], reordered["protocol"])
        self.assertNotEqual(mapped["artifact_hash"], reordered["artifact_hash"])

    def test_tampering_bundle_baseline_graph(self):
        wrapper = self.make()
        evil = copy.deepcopy(wrapper)
        evil["mappings"]["tracks"].append(
            "musicbrainz:recording:00000000-0000-0000-0000-000000000010")
        evil["statistics"]["track_count"] += 1
        self.rehash_wrapper(evil)
        with self.assertRaisesRegex(ListenBrainzIntegrityError, "represented"):
            validate(evil)
        with tempfile.TemporaryDirectory() as d:
            save(d, wrapper); self.assertEqual(load(d)["artifact_hash"], wrapper["artifact_hash"])
            gd=os.path.join(d,"graph"); save_graph(gd,wrapper); graph=load_graph(gd,wrapper)
            self.assertEqual(graph,graph_input(wrapper)); bad=copy.deepcopy(graph); bad["mapping_hash"]="0"*64
            bad["graph_hash"] = sha256(
                {key: value for key, value in bad.items() if key != "graph_hash"})
            with self.assertRaisesRegex(ListenBrainzIntegrityError, "not bound"):
                validate_graph(wrapper,bad)
            self.assertEqual(load_graph(gd, wrapper, graph["graph_hash"]), graph)
            with self.assertRaisesRegex(ListenBrainzIntegrityError, "expected graph hash"):
                load_graph(gd, wrapper, "0" * 64)
        for model in ("random","popularity","itemknn"):
            for policy in ("fixed_sampled","full_catalog"):
                out=evaluate_baseline(wrapper,model=model,target="track",split="test",candidate_policy=policy,k=1)
                self.assertEqual(out["model"],model)
                self.assertIn("algorithm", out["model_config"])

    def test_per_user_split_retains_insufficient_user_mappings(self):
        wrapper = self.make(
            split_strategy="per_user_last_timestamp_groups",
            validation_cutoff=None,
            test_cutoff=None)
        self.assertEqual(wrapper["protocol"]["statistics"]["total_users"], 3)
        self.assertEqual(len(wrapper["protocol"]["statistics"]["insufficient_users"]), 1)
        raw_plays = sum(
            row["play_count"]
            for name in ("train", "validation", "test")
            for row in wrapper["protocol"]["targets"]["track"]["raw_splits"][name])
        self.assertLess(raw_plays, wrapper["statistics"]["emitted_events"])
        self.assertEqual(len(wrapper["mappings"]["users"]), 3)
        validate(wrapper)

        missing_diagnostic = copy.deepcopy(wrapper)
        missing_diagnostic["protocol"]["statistics"]["insufficient_users"] = []
        missing_diagnostic["protocol"]["statistics"]["eligible_users"] = 3
        self.rehash_embedded_protocol(missing_diagnostic)
        with self.assertRaisesRegex(ListenBrainzIntegrityError,
                                    "insufficient-user identities"):
            validate(missing_diagnostic)

    def test_rehashed_schema_mapping_provenance_and_statistics_tampering(self):
        wrapper = self.make()

        malformed = copy.deepcopy(wrapper)
        malformed["mappings"]["tracks"][-1] = (
            "musicbrainz:recording:00000000-0000-0000-0000-00000000000g")
        self.rehash_wrapper(malformed)
        with self.assertRaisesRegex(ListenBrainzIntegrityError, "canonically namespaced"):
            validate(malformed)

        boolean_schema = copy.deepcopy(wrapper)
        boolean_schema["schema_version"] = True
        self.rehash_wrapper(boolean_schema)
        with self.assertRaisesRegex(ListenBrainzIntegrityError, "schema"):
            validate(boolean_schema)

        boolean_mapping_version = copy.deepcopy(wrapper)
        boolean_mapping_version["config"]["mapping_version"] = True
        self.rehash_wrapper(boolean_mapping_version, config_changed=True)
        with self.assertRaisesRegex(ListenBrainzIntegrityError, "config"):
            validate(boolean_mapping_version)

        bad_path = copy.deepcopy(wrapper)
        bad_path["provenance"] = {"local_path": 0}
        with self.assertRaisesRegex(ListenBrainzIntegrityError, "provenance"):
            validate(bad_path)

        impossible_stats = copy.deepcopy(wrapper)
        impossible_stats["statistics"]["conflicting_duplicate_event_keys"] = 2
        self.rehash_wrapper(impossible_stats)
        with self.assertRaisesRegex(ListenBrainzIntegrityError, "feasible bounds"):
            validate(impossible_stats)

        negative_size = copy.deepcopy(wrapper)
        negative_size["source"]["member_bytes"] = -1
        self.rehash_wrapper(negative_size)
        with self.assertRaisesRegex(ListenBrainzIntegrityError, "non-negative"):
            validate(negative_size)

    def test_random_baseline_preserves_integer_priorities(self):
        wrapper = self.make()
        with mock.patch("listenbrainz_protocol.core.evaluate_rankings",
                        return_value=self.metric_result()) as patched:
            output = evaluate_baseline(
                wrapper, model="random", target="track", split="test",
                candidate_policy="full_catalog", k=1, random_seed=7)
        scores = patched.call_args[0][0]
        self.assertTrue(all(isinstance(score, int)
                            for rows in scores.values() for unused_item, score in rows))
        self.assertEqual(output["model_config"],
                         {"algorithm": "sha256_priority_v1", "random_seed": 7})

    def test_popularity_and_itemknn_scores_are_train_only_and_hand_calculated(self):
        def recording(number):
            return "00000000-0000-0000-0000-%012d" % number

        raw_rows = []
        for user, timestamp, track in (
                ("A", 1, 1), ("A", 1, 2),
                ("B", 1, 1), ("B", 1, 3),
                ("C", 1, 2), ("C", 2, 4), ("C", 3, 3)):
            raw_rows.append(json.dumps({
                "user_name": user,
                "listened_at": timestamp,
                "recording_msid": recording(track + 100),
                "track_metadata": {"mbid_mapping": {
                    "recording_mbid": recording(track)}}}, sort_keys=True))
        descriptor, path = tempfile.mkstemp()
        try:
            os.write(descriptor, ("\n".join(raw_rows) + "\n").encode("utf-8"))
            os.close(descriptor)
            wrapper = prepare(
                path, dump_id="baseline-test", dump_type="full",
                archive_sha256=ARCHIVE, member_path="baseline/member.listens",
                track_identity_policy="server_mapped_musicbrainz",
                album_entity="release_group", split_strategy="global_time_cutoffs",
                validation_cutoff=2, test_cutoff=3, repeat_policy="repeat_allowed",
                catalog_policy="all_mapped")
        finally:
            if os.path.exists(path):
                os.unlink(path)

        user_digest = hashlib.sha256(b"listenbrainz:user:v1\0C").hexdigest()
        user_id = wrapper["mappings"]["users"].index(user_digest)
        captured = []

        def inspect_scores(scores, unused_positives, unused_candidates, unused_k,
                           unused_catalog, unused_popularity):
            captured.append(scores)
            return self.metric_result(len(scores))

        with mock.patch("listenbrainz_protocol.core.evaluate_rankings",
                        side_effect=inspect_scores):
            evaluate_baseline(wrapper, model="popularity", target="track", split="test")
            evaluate_baseline(wrapper, model="itemknn", target="track", split="test")

        self.assertEqual(dict(captured[0][user_id]), {0: 2, 1: 2, 2: 1, 3: 0})
        self.assertEqual(dict(captured[1][user_id]), {0: 0.5, 1: 0, 2: 0, 3: 0})

    def test_local_path_is_nonidentity_provenance(self):
        without_path = self.make()
        with_path = self.make(include_local_path=True)
        self.assertEqual(without_path["artifact_hash"], with_path["artifact_hash"])
        self.assertEqual(with_path["provenance"], {"local_path": os.path.abspath(FIXTURE)})

    def test_hard_parse_failures(self):
        bads=[b"\n",b'{"user_name":"x","user_name":"y"}\n',b'{"user_name":"x","listened_at":NaN,"track_metadata":{}}\n',b'{"user_name":"x","listened_at":true,"track_metadata":{}}\n',b'\xff\n']
        for content in bads:
            fd,path=tempfile.mkstemp();os.write(fd,content);os.close(fd)
            try:
                with self.assertRaises(ListenBrainzIntegrityError): self.make_path(path)
            finally: os.unlink(path)
    def make_path(self,path):
        return prepare(path,dump_id="x",dump_type="full",archive_sha256=ARCHIVE,member_path="x.listens",track_identity_policy="recording_msid",album_entity="release_group",split_strategy="global_time_cutoffs",validation_cutoff=2,test_cutoff=3)

    def test_cli_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle=os.path.join(directory,"bundle"); output=io.StringIO()
            command=["prepare",FIXTURE,bundle,"--dump-id","synthetic","--dump-type","full","--archive-sha256",ARCHIVE,"--member-path","synthetic/member.listens","--track-identity-policy","server_mapped_musicbrainz","--album-entity","release_group","--split-strategy","global_time_cutoffs","--validation-cutoff","20","--test-cutoff","30","--catalog-policy","all_mapped","--repeat-policy","repeat_allowed"]
            with contextlib.redirect_stdout(output): self.assertEqual(cli_main(command),0)
            with contextlib.redirect_stdout(output): self.assertEqual(cli_main(["verify",bundle]),0)
            with contextlib.redirect_stdout(output): self.assertEqual(cli_main(["baseline",bundle,"--model","random","--target","track"]),0)
            graph=os.path.join(directory,"graph")
            with contextlib.redirect_stdout(output): self.assertEqual(cli_main(["graph-input",bundle,graph]),0)
            with contextlib.redirect_stdout(output): self.assertEqual(cli_main(["graph-verify",bundle,graph]),0)
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            for record in (records[0], records[1], records[3], records[4]):
                self.assertTrue({"artifact_hash", "protocol_hash", "config_hash",
                                 "protocol_config_hash"}.issubset(record))
            self.assertIn("graph_hash", records[3])
            self.assertIn("graph_hash", records[4])

if __name__ == "__main__": unittest.main()
