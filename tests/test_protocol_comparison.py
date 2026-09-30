import contextlib
import copy
import io
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lfm1b_protocol.artifacts import (prepare_protocol_artifact,
                                       save_protocol_artifact,
                                       validate_protocol_artifact)
from lfm1b_protocol.canonical import sha256
from lfm1b_protocol.io import read_listening_events_with_provenance
from lfm1b_protocol.models import Event
from listenbrainz_protocol import (evaluate_baseline as listenbrainz_baseline,
                                   prepare as prepare_listenbrainz,
                                   save as save_listenbrainz)
from protocol_comparison import (ComparisonError, case_from_artifact,
                                 contrast_pair, create_contrast_analysis,
                                 create_plan, load_contrast_analysis, load_plan,
                                 load_report, run_plan, save_contrast_analysis,
                                 save_plan, save_report,
                                 validate_contrast_analysis, validate_report)
from protocol_comparison.cli import main as comparison_cli


ROOT = Path(__file__).resolve().parent.parent
LFM_FIXTURE = ROOT / "examples" / "tiny_events.dat"
LISTENBRAINZ_FIXTURE = ROOT / "examples" / "listenbrainz_synthetic.listens"


class ProtocolComparisonTests(unittest.TestCase):
    def make_artifacts(self, root, horizon="as_of_split",
                       catalog_policy="all_mapped"):
        lfm = prepare_protocol_artifact(
            read_listening_events_with_provenance(str(LFM_FIXTURE), False),
            sampled_negatives=3, seed=0, catalog_policy=catalog_policy,
            split_strategy="global_time_cutoffs", validation_cutoff=2,
            test_cutoff=3, repeat_policy="repeat_allowed",
            positive_filter_horizon=horizon)
        suffix = horizon + "-" + catalog_policy
        lfm_path = Path(root) / ("lfm-" + suffix)
        save_protocol_artifact(str(lfm_path), lfm)

        listenbrainz = prepare_listenbrainz(
            str(LISTENBRAINZ_FIXTURE), dump_id="synthetic", dump_type="full",
            archive_sha256="a" * 64, member_path="synthetic/member.listens",
            track_identity_policy="server_mapped_musicbrainz",
            album_entity="release_group", split_strategy="global_time_cutoffs",
            validation_cutoff=20, test_cutoff=30,
            repeat_policy="repeat_allowed", positive_filter_horizon=horizon,
            catalog_policy=catalog_policy, sampled_negatives=3, seed=0)
        listenbrainz_path = Path(root) / ("listenbrainz-" + suffix)
        save_listenbrainz(str(listenbrainz_path), listenbrainz)
        return lfm, lfm_path, listenbrainz, listenbrainz_path

    def select_result(self, report, **expected):
        matches = [value for value in report["results"]
                   if all(value[key] == item for key, item in expected.items())]
        self.assertEqual(len(matches), 1, expected)
        return matches[0]

    def make_plan(self, root):
        unused_lfm, lfm_path, unused_wrapper, listenbrainz_path = self.make_artifacts(root)
        options = dict(
            targets=("track",), splits=("test",),
            candidate_policies=("fixed_sampled", "full_catalog"),
            models=("random", "popularity", "itemknn"), k_values=(1, 2),
            random_seeds=(0, 7), neighbor_limit=2)
        cases = (
            case_from_artifact("lfm", "lfm1b", "lfm1b", str(lfm_path), **options),
            case_from_artifact("listenbrainz", "listenbrainz-synthetic",
                               "listenbrainz", str(listenbrainz_path), **options),
        )
        return create_plan(cases, max_candidate_scores=1000,
                           max_itemknn_pairs=1000)

    def test_cross_adapter_plan_and_report_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_plan(directory)
            first = run_plan(plan, plan["plan_hash"])
            second = run_plan(plan, plan["plan_hash"])
            self.assertEqual(first, second)
            self.assertEqual(len(first["results"]), 32)
            self.assertEqual({value["adapter"] for value in first["results"]},
                             {"lfm1b", "listenbrainz"})
            self.assertEqual({value["model"] for value in first["results"]},
                             {"random", "popularity", "itemknn"})
            self.assertTrue(all(value["contributions"]
                                for value in first["results"]))
            self.assertTrue(all(len(value["result_hash"]) == 64
                                for value in first["results"]))

            plan_path = Path(directory) / "plan.json"
            report_path = Path(directory) / "report.json"
            save_plan(plan_path, plan)
            save_report(report_path, first)
            self.assertEqual(load_plan(plan_path, plan["plan_hash"]), plan)
            loaded = load_report(report_path, first["report_hash"])
            self.assertEqual(loaded, json.loads(report_path.read_text(encoding="utf-8")))

    def test_plan_identity_excludes_local_paths_but_requires_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_plan(directory)
            relocated_cases = copy.deepcopy(plan["cases"])
            for index, case in enumerate(relocated_cases):
                case["artifact_path"] = "/relocated/artifact-%d" % index
            relocated = create_plan(
                relocated_cases, plan["limits"]["max_candidate_scores"],
                plan["limits"]["max_itemknn_pairs"])
            self.assertEqual(relocated["plan_hash"], plan["plan_hash"])

            report = run_plan(plan)
            relocated_report = copy.deepcopy(report)
            relocated_report["provenance"]["artifact_paths"] = {
                key: "/relocated/" + key
                for key in relocated_report["provenance"]["artifact_paths"]}
            self.assertEqual(validate_report(
                relocated_report, report["report_hash"])["report_hash"],
                report["report_hash"])

            bad = copy.deepcopy(plan)
            bad["cases"][0]["expected_hashes"]["protocol_hash"] = "0" * 64
            bad["plan_hash"] = sha256({
                "schema_version": bad["schema_version"],
                "cases": [{key: value for key, value in case.items()
                           if key != "artifact_path"} for case in bad["cases"]],
                "limits": bad["limits"],
            })
            with self.assertRaisesRegex(ValueError, "expected protocol hash"):
                run_plan(bad, bad["plan_hash"])

    def test_shared_baselines_match_existing_listenbrainz_algorithms(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_plan(directory)
            report = run_plan(plan)
            wrapper = prepare_listenbrainz(
                str(LISTENBRAINZ_FIXTURE), dump_id="synthetic", dump_type="full",
                archive_sha256="a" * 64, member_path="synthetic/member.listens",
                track_identity_policy="server_mapped_musicbrainz",
                album_entity="release_group", split_strategy="global_time_cutoffs",
                validation_cutoff=20, test_cutoff=30,
                repeat_policy="repeat_allowed", catalog_policy="all_mapped",
                sampled_negatives=3, seed=0)
            for model in ("popularity", "itemknn"):
                existing = listenbrainz_baseline(
                    wrapper, model=model, target="track", split="test",
                    candidate_policy="full_catalog", k=1, neighbor_limit=2)
                result = next(value for value in report["results"]
                              if value["case_name"] == "listenbrainz"
                              and value["model"] == model
                              and value["candidate_policy"] == "full_catalog"
                              and value["k"] == 1)
                for key in ("users", "recall", "ndcg", "mrr",
                            "mean_average_precision", "precision", "hit_rate",
                            "catalog_coverage", "novelty"):
                    self.assertEqual(result["metrics"][key], existing[key])
            random_results = [
                value for value in report["results"]
                if value["case_name"] == "listenbrainz"
                and value["model"] == "random"
                and value["candidate_policy"] == "full_catalog"
                and value["k"] == 1]
            self.assertEqual([value["model_seed"] for value in random_results], [0, 7])
            self.assertEqual(len({value["candidate_rows_hash"]
                                  for value in random_results}), 1)
            self.assertEqual(len({value["result_hash"] for value in random_results}), 2)

    def test_mask_configuration_is_bound_and_safety_limits_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            unused_lfm, as_of_path, unused_wrapper, unused_lb = self.make_artifacts(
                directory, "as_of_split")
            unused_lfm, oracle_path, unused_wrapper, unused_lb = self.make_artifacts(
                directory, "all_observed")
            options = dict(targets=("track",), splits=("validation",),
                           candidate_policies=("full_catalog",),
                           models=("popularity",), k_values=(1,),
                           random_seeds=(), neighbor_limit=1)
            cases = (
                case_from_artifact("as-of", "lfm1b", "lfm1b", str(as_of_path),
                                   **options),
                case_from_artifact("oracle", "lfm1b", "lfm1b", str(oracle_path),
                                   **options),
            )
            report = run_plan(create_plan(cases, 1000, 1000))
            horizons = {value["case_name"]:
                        value["mask_configuration"]["positive_filter_horizon"]
                        for value in report["results"]}
            self.assertEqual(horizons, {"as-of": "as_of_split",
                                        "oracle": "all_observed"})
            with self.assertRaisesRegex(ComparisonError, "upper bound"):
                run_plan(create_plan(cases, 1, 1000))
            with self.assertRaisesRegex(ComparisonError, "requests .* results"):
                run_plan(create_plan(cases, 1000, 1000, max_results=1))
            wide = copy.deepcopy(cases[0])
            wide["k_values"] = list(range(1, 101))
            with self.assertRaisesRegex(ComparisonError, "contribution-row upper bound"):
                run_plan(create_plan(
                    (wide,), 100000, 100000, max_results=1000,
                    max_contribution_rows=100, max_ranking_items=1000000))
            with mock.patch("protocol_comparison.runner.evaluate_protocol_baseline") as score:
                with self.assertRaisesRegex(ComparisonError, "report exceeds"):
                    run_plan(create_plan(
                        (cases[0],), 1000, 1000, max_report_bytes=1))
                score.assert_not_called()

    def test_fixed_width_plan_and_cutoff_fields_fail_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_plan(directory)
            bad = copy.deepcopy(plan)
            bad["cases"][0]["random_seeds"] = [1 << 80]
            bad["plan_hash"] = sha256({
                "schema_version": bad["schema_version"],
                "cases": [{key: value for key, value in case.items()
                           if key != "artifact_path"} for case in bad["cases"]],
                "limits": bad["limits"],
            })
            with mock.patch("protocol_comparison.runner.evaluate_protocol_baseline") as score:
                with self.assertRaisesRegex(ComparisonError, "signed 63-bit"):
                    run_plan(bad)
                score.assert_not_called()

            base = 1 << 70
            events = tuple(
                Event(user_id=user, artist_id=index, album_id=index,
                      track_id=index, timestamp=base + index - 1)
                for user in (1, 2) for index in (1, 2, 3))
            protocol = prepare_protocol_artifact(
                events, sampled_negatives=1, seed=0, catalog_policy="all_mapped",
                split_strategy="global_time_cutoffs", validation_cutoff=base + 1,
                test_cutoff=base + 2, repeat_policy="repeat_allowed")
            artifact_path = Path(directory) / "huge-cutoff"
            save_protocol_artifact(artifact_path, protocol)
            case = case_from_artifact(
                "huge-cutoff", "synthetic", "lfm1b", artifact_path,
                models=("popularity",), k_values=(1,), random_seeds=())
            with mock.patch("protocol_comparison.runner.evaluate_protocol_baseline") as score:
                with self.assertRaisesRegex(ComparisonError, "cutoff timestamps"):
                    run_plan(create_plan((case,)))
                score.assert_not_called()

    def test_cli_init_run_verify_and_semantic_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            unused_lfm, lfm_path, unused_wrapper, unused_lb = self.make_artifacts(directory)
            plan_path = Path(directory) / "plan.json"
            report_path = Path(directory) / "report.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = comparison_cli([
                    "init", str(plan_path), "--artifact", "lfm1b", "lfm1b",
                    "local-lfm", str(lfm_path), "--target", "track", "--split",
                    "test", "--candidate-policy", "full_catalog", "--model",
                    "popularity", "-k", "1"])
            self.assertEqual(code, 0)
            plan_hash = json.loads(stdout.getvalue())["plan_hash"]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = comparison_cli([
                    "run", str(plan_path), str(report_path),
                    "--expected-plan-hash", plan_hash])
            self.assertEqual(code, 0)
            report_hash = json.loads(stdout.getvalue())["report_hash"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(comparison_cli([
                    "verify-report", str(report_path),
                    "--expected-report-hash", report_hash]), 0)

            report = load_report(report_path)
            bad = copy.deepcopy(report)
            contribution = bad["results"][0]["contributions"][0]
            contribution["relevant_items"] = 0
            result = bad["results"][0]
            result["result_hash"] = sha256(
                {key: value for key, value in result.items()
                 if key != "result_hash"})
            bad["report_hash"] = sha256({
                "schema_version": bad["schema_version"],
                "plan_hash": bad["plan_hash"], "results": bad["results"]})
            with self.assertRaisesRegex(ComparisonError, "relevant_items"):
                validate_report(bad)

            bad = copy.deepcopy(report)
            result = bad["results"][0]
            contribution = result["contributions"][0]
            contribution["recommended_items"] = 0
            contribution["hit_items"] = 0
            contribution["recommended_item_ids"] = []
            contribution["hit_ranks"] = []
            for metric_name in ("recall", "ndcg", "mrr",
                                "mean_average_precision", "precision",
                                "hit_rate", "novelty"):
                contribution[metric_name] = 0.0
            contributions = result["contributions"]
            for metric_name in ("recall", "ndcg", "mrr",
                                "mean_average_precision", "precision",
                                "hit_rate"):
                result["metrics"][metric_name] = sum(
                    value[metric_name] for value in contributions) / len(contributions)
            recommendation_count = sum(
                value["recommended_items"] for value in contributions)
            result["metrics"]["novelty"] = (
                sum(value["novelty"] * value["recommended_items"]
                    for value in contributions) / recommendation_count
                if recommendation_count else 0.0)
            result["metrics"]["catalog_coverage"] = len({
                item for value in contributions
                for item in value["recommended_item_ids"]
            }) / float(result["catalog_items"])
            result["result_hash"] = sha256(
                {key: value for key, value in result.items()
                 if key != "result_hash"})
            bad["report_hash"] = sha256({
                "schema_version": bad["schema_version"],
                "plan_hash": bad["plan_hash"], "results": bad["results"]})
            with self.assertRaisesRegex(ComparisonError, "recommendation count"):
                validate_report(bad)

            for metric_name in ("ndcg", "mrr", "mean_average_precision"):
                bad = copy.deepcopy(report)
                contribution = bad["results"][0]["contributions"][0]
                current = contribution[metric_name]
                contribution[metric_name] = current - 0.01 if current > 0.01 else 0.01
                contributions = bad["results"][0]["contributions"]
                bad["results"][0]["metrics"][metric_name] = sum(
                    value[metric_name] for value in contributions) / len(contributions)
                result = bad["results"][0]
                result["result_hash"] = sha256(
                    {key: value for key, value in result.items()
                     if key != "result_hash"})
                bad["report_hash"] = sha256({
                    "schema_version": bad["schema_version"],
                    "plan_hash": bad["plan_hash"], "results": bad["results"]})
                with self.assertRaisesRegex(ComparisonError,
                                            "contribution metrics"):
                    validate_report(bad)

    def test_report_v3_rejects_one_ulp_novelty_tampering_after_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_plan(self.make_plan(directory))
            bad = copy.deepcopy(report)
            contribution = bad["results"][0]["contributions"][0]
            bits = struct.unpack(">Q", struct.pack(">d", contribution["novelty"]))[0]
            contribution["novelty"] = struct.unpack(
                ">d", struct.pack(">Q", bits + 1))[0]
            result = bad["results"][0]
            result["result_hash"] = sha256({
                key: value for key, value in result.items() if key != "result_hash"})
            bad["report_hash"] = sha256({
                "schema_version": bad["schema_version"],
                "numerical_semantics": bad["numerical_semantics"],
                "plan_hash": bad["plan_hash"], "results": bad["results"]})
            with self.assertRaisesRegex(ComparisonError, "contribution metrics"):
                validate_report(bad)

    def test_expected_report_hash_rejects_coordinated_novelty_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_plan(self.make_plan(directory))
            bad = copy.deepcopy(report)
            contribution = bad["results"][0]["contributions"][0]
            # This coordinated edit remains internally self-consistent.  The
            # original externally retained report hash is its trust anchor.
            contribution["recommended_item_novelties"] = list(
                contribution["recommended_item_novelties"])
            contribution["recommended_item_novelties"][0] += 0.25
            contribution["novelty"] = math.fsum(
                contribution["recommended_item_novelties"]) / len(
                    contribution["recommended_item_novelties"])
            result = bad["results"][0]
            count = sum(value["recommended_items"]
                        for value in result["contributions"])
            result["metrics"]["novelty"] = math.fsum(
                value["novelty"] * value["recommended_items"]
                for value in result["contributions"]) / count
            result["result_hash"] = sha256({
                key: value for key, value in result.items() if key != "result_hash"})
            bad["report_hash"] = sha256({
                "schema_version": bad["schema_version"],
                "numerical_semantics": bad["numerical_semantics"],
                "plan_hash": bad["plan_hash"], "results": bad["results"]})
            validate_report(bad)
            with self.assertRaisesRegex(ComparisonError, "expected comparison"):
                validate_report(bad, report["report_hash"])

    def test_protocol_rejects_unbound_top_level_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact, unused_path, unused_wrapper, unused_lb = self.make_artifacts(
                directory)
            bad = copy.deepcopy(artifact)
            bad["created_at"] = "unbound"
            with self.assertRaisesRegex(ValueError, "invalid shape"):
                validate_protocol_artifact(bad)

    def test_runner_validates_each_external_artifact_twice_by_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            unused_lfm, lfm_path, unused_wrapper, unused_lb = self.make_artifacts(
                directory)
            case = case_from_artifact(
                "lfm", "lfm1b", "lfm1b", lfm_path, targets=("track",),
                splits=("test",), candidate_policies=("full_catalog",),
                models=("popularity",), k_values=(1,), random_seeds=())
            plan = create_plan((case,), 1000, 1000)
            from lfm1b_protocol import artifacts
            original = artifacts.validate_protocol_artifact
            with mock.patch.object(artifacts, "validate_protocol_artifact",
                                   wraps=original) as validate:
                run_plan(plan)
            # One strict load is the whole-plan preflight; the second is the
            # scoring boundary after the preflight protocol was discarded.
            self.assertEqual(validate.call_count, 2)

    def test_report_depth_limits_are_controlled(self):
        deeply_nested = value = {}
        for unused in range(70):
            next_value = {}
            value["nested"] = next_value
            value = next_value
        report = {"schema_version": 3, "numerical_semantics": {},
                  "plan_hash": "0" * 64, "results": [],
                  "provenance": {"artifact_paths": deeply_nested},
                  "report_hash": "0" * 64}
        with self.assertRaisesRegex(ComparisonError, "hard byte limit"):
            validate_report(report)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep-report.json"
            path.write_text("[" * 1100 + "0" + "]" * 1100,
                            encoding="utf-8")
            with self.assertRaisesRegex(ComparisonError,
                                        "malformed|hard byte limit|invalid shape"):
                load_report(path)

    def test_all_serialized_report_budget_is_preflighted_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            unused_lfm, lfm_path, unused_wrapper, unused_lb = self.make_artifacts(
                directory)
            case = case_from_artifact(
                "lfm", "lfm1b", "lfm1b", lfm_path, targets=("track",),
                splits=("test",), candidate_policies=("full_catalog",),
                models=("popularity",), k_values=(1,), random_seeds=())
            plan = create_plan((case,), 1000, 1000, max_report_bytes=100000)
            oversized = {"candidate_scores": 1, "users": 1,
                         "contribution_rows": 1, "ranking_items": 1,
                         "serialized_ranking_items": 10 ** 12}
            with mock.patch(
                    "protocol_comparison.runner._comparison_work_upper_bounds_validated",
                    return_value=oversized), mock.patch(
                        "protocol_comparison.runner._evaluate_protocol_baseline_validated") as score:
                with self.assertRaisesRegex(ComparisonError, "report exceeds"):
                    run_plan(plan)
                score.assert_not_called()

    def test_generated_writes_reject_symlinks_and_preserve_unowned_stale_files(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_plan(directory)
            victim = Path(directory) / "victim.json"
            victim.write_bytes(b"must remain unchanged")
            destination = Path(directory) / "plan-link.json"
            destination.symlink_to(victim)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                save_plan(destination, plan)
            self.assertEqual(victim.read_bytes(), b"must remain unchanged")

            outside = Path(directory) / "outside"
            outside.mkdir()
            approved = Path(directory) / "approved"
            approved.mkdir()
            (approved / "redirect").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "path components.*symbolic"):
                save_plan(approved / "redirect" / "nested" / "plan.json", plan)
            self.assertFalse((outside / "nested" / "plan.json").exists())

            real_destination = Path(directory) / "plan.json"
            stale = Path(directory) / "plan.json.tmp-12345"
            stale.write_bytes(b"belongs to another invocation")
            save_plan(real_destination, plan)
            self.assertEqual(stale.read_bytes(), b"belongs to another invocation")

            collision = Path(directory) / ".generated-collision.tmp"
            collision.write_bytes(b"also belongs to another invocation")
            with mock.patch("research_data.hygiene.secrets.token_hex",
                            return_value="collision"):
                with self.assertRaisesRegex(ValueError, "unique generated temp"):
                    save_plan(Path(directory) / "second-plan.json", plan)
            self.assertEqual(collision.read_bytes(),
                             b"also belongs to another invocation")

    def test_paired_model_seed_and_mask_contrasts_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            unused_lfm, warm_path, unused_wrapper, unused_lb = self.make_artifacts(
                directory, catalog_policy="train_observed")
            unused_lfm, mapped_path, unused_wrapper, unused_lb = self.make_artifacts(
                directory, catalog_policy="all_mapped")
            options = dict(
                targets=("track",), splits=("validation",),
                candidate_policies=("full_catalog",),
                models=("random", "popularity"), k_values=(1,),
                random_seeds=(0, 7), neighbor_limit=1)
            cases = (
                case_from_artifact("warm", "lfm1b", "lfm1b", warm_path,
                                   **options),
                case_from_artifact("mapped", "lfm1b", "lfm1b", mapped_path,
                                   **options),
            )
            report = run_plan(create_plan(cases, 1000, 1000))
            warm = self.select_result(
                report, case_name="warm", model="popularity", model_seed=None)
            mapped = self.select_result(
                report, case_name="mapped", model="popularity", model_seed=None)
            random_zero = self.select_result(
                report, case_name="mapped", model="random", model_seed=0)
            random_seven = self.select_result(
                report, case_name="mapped", model="random", model_seed=7)
            self.assertEqual(
                warm["artifact_identity"]["population_hash"],
                mapped["artifact_identity"]["population_hash"])
            pairs = (
                contrast_pair("seed-effect", "seed", random_zero["result_hash"],
                              random_seven["result_hash"]),
                contrast_pair("mask-effect", "mask", warm["result_hash"],
                              mapped["result_hash"]),
                contrast_pair("model-effect", "model", mapped["result_hash"],
                              random_zero["result_hash"]),
            )
            first = create_contrast_analysis(report, pairs, report["report_hash"])
            second = create_contrast_analysis(report, reversed(pairs),
                                               report["report_hash"])
            self.assertEqual(first, second)
            self.assertEqual(
                [value["name"] for value in first["contrasts"]],
                ["mask-effect", "model-effect", "seed-effect"])
            mask = next(value for value in first["contrasts"]
                        if value["name"] == "mask-effect")
            self.assertEqual(mask["baseline"]["native_metrics"]["users"], 1)
            self.assertEqual(mask["comparison"]["native_metrics"]["users"], 2)
            self.assertEqual(mask["cohort"], {
                "common_users": 1,
                "baseline_only_user_ids": [],
                "comparison_only_user_ids": [2],
            })
            warm_by_user = {value["user_id"]: value
                            for value in warm["contributions"]}
            mapped_by_user = {value["user_id"]: value
                              for value in mapped["contributions"]}
            delta = mask["user_deltas"][0]
            self.assertEqual(delta["user_id"], 1)
            for metric in ("recall", "ndcg", "mrr", "mean_average_precision",
                           "precision", "hit_rate", "novelty"):
                expected = mapped_by_user[1][metric] - warm_by_user[1][metric]
                self.assertEqual(delta[metric], 0.0 if expected == 0 else expected)
                self.assertEqual(mask["paired_metrics"][metric]["mean_delta"],
                                 delta[metric])
            model = next(value for value in first["contrasts"]
                         if value["name"] == "model-effect")
            mapped_random_by_user = {
                value["user_id"]: value for value in random_zero["contributions"]}
            for metric in ("recall", "ndcg", "mrr", "mean_average_precision",
                           "precision", "hit_rate", "novelty"):
                expected = sum(
                    mapped_random_by_user[user][metric] -
                    mapped_by_user[user][metric]
                    for user in sorted(mapped_by_user)) / len(mapped_by_user)
                self.assertEqual(
                    model["paired_metrics"][metric]["mean_delta"],
                    0.0 if expected == 0 else expected)

            analysis_path = Path(directory) / "contrasts.json"
            save_contrast_analysis(analysis_path, first, report)
            self.assertEqual(load_contrast_analysis(
                analysis_path, report, first["analysis_hash"],
                report["report_hash"]), first)
            with self.assertRaisesRegex(ComparisonError, "paired-row limit"):
                create_contrast_analysis(report, pairs, max_paired_rows=4)

    def test_contrast_compatibility_and_duplicate_results_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_plan(self.make_plan(directory))
            lfm_popularity = self.select_result(
                report, case_name="lfm", candidate_policy="full_catalog",
                model="popularity", model_seed=None, k=1)
            lfm_random_zero = self.select_result(
                report, case_name="lfm", candidate_policy="full_catalog",
                model="random", model_seed=0, k=1)
            lfm_random_seven = self.select_result(
                report, case_name="lfm", candidate_policy="full_catalog",
                model="random", model_seed=7, k=1)
            lfm_fixed_random = self.select_result(
                report, case_name="lfm", candidate_policy="fixed_sampled",
                model="random", model_seed=0, k=1)
            listenbrainz_popularity = self.select_result(
                report, case_name="listenbrainz",
                candidate_policy="full_catalog", model="popularity",
                model_seed=None, k=1)

            with self.assertRaisesRegex(ComparisonError,
                                        "different model families"):
                create_contrast_analysis(report, (contrast_pair(
                    "wrong-kind", "model", lfm_random_zero["result_hash"],
                    lfm_random_seven["result_hash"]),))
            with self.assertRaisesRegex(ComparisonError,
                                        "matching candidate_policy"):
                create_contrast_analysis(report, (contrast_pair(
                    "different-candidates", "model",
                    lfm_popularity["result_hash"],
                    lfm_fixed_random["result_hash"]),))
            with self.assertRaisesRegex(ComparisonError, "matching dataset"):
                create_contrast_analysis(report, (contrast_pair(
                    "different-populations", "model",
                    lfm_random_zero["result_hash"],
                    listenbrainz_popularity["result_hash"]),))
            with self.assertRaisesRegex(ComparisonError, "absent"):
                create_contrast_analysis(report, (contrast_pair(
                    "missing", "model", lfm_popularity["result_hash"],
                    "0" * 64),))
            with self.assertRaisesRegex(ComparisonError, "expected comparison"):
                create_contrast_analysis(
                    report, (contrast_pair(
                        "valid", "model", lfm_popularity["result_hash"],
                        lfm_random_zero["result_hash"]),), "0" * 64)

            bad = copy.deepcopy(report)
            bad["results"].append(copy.deepcopy(bad["results"][0]))
            bad["results"].sort(key=lambda value: (
                value["case_name"], value["target"], value["split"],
                value["candidate_policy"], value["model"],
                -1 if value["model_seed"] is None else value["model_seed"],
                value["k"]))
            bad["report_hash"] = sha256({
                "schema_version": bad["schema_version"],
                "plan_hash": bad["plan_hash"], "results": bad["results"]})
            with self.assertRaisesRegex(ComparisonError, "must be unique"):
                validate_report(bad)

    def test_population_identity_and_analysis_input_limits_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = {
                "source_kind": "listening_events",
                "parser_version": "lfm_listening_events_tsv_v1",
                "bytes": 123,
                "sha256": "a" * 64,
                "has_header": False,
                "column_order": ("user_id", "artist_id", "album_id",
                                 "track_id", "timestamp"),
                "parsed_row_count": 6,
            }

            def prepare_variant(test_track, horizon, path):
                events = tuple(
                    Event(user_id=user, artist_id=1, album_id=1,
                          track_id=track, timestamp=timestamp)
                    for user in (1, 2)
                    for track, timestamp in ((1, 1), (1, 2), (test_track, 3)))
                protocol = prepare_protocol_artifact(
                    events, sampled_negatives=2, seed=0,
                    catalog_policy="all_mapped", sources=(source,),
                    split_strategy="global_time_cutoffs", validation_cutoff=2,
                    test_cutoff=3, repeat_policy="repeat_allowed",
                    positive_filter_horizon=horizon)
                save_protocol_artifact(path, protocol)
                return path

            first_path = prepare_variant(
                2, "as_of_split", Path(directory) / "first")
            second_path = prepare_variant(
                3, "all_observed", Path(directory) / "second")
            options = dict(
                targets=("track",), splits=("validation",),
                candidate_policies=("full_catalog",),
                models=("random", "popularity"), k_values=(1,),
                random_seeds=(0,), neighbor_limit=1)
            report = run_plan(create_plan((
                case_from_artifact("first", "self-attested", "lfm1b",
                                   first_path, **options),
                case_from_artifact("second", "self-attested", "lfm1b",
                                   second_path, **options),
            ), 1000, 1000))
            first_popularity = self.select_result(
                report, case_name="first", model="popularity", model_seed=None)
            second_popularity = self.select_result(
                report, case_name="second", model="popularity", model_seed=None)
            self.assertNotEqual(
                first_popularity["artifact_identity"]["population_hash"],
                second_popularity["artifact_identity"]["population_hash"])
            with self.assertRaisesRegex(ComparisonError,
                                        "normalized population identities"):
                create_contrast_analysis(report, (contrast_pair(
                    "unsafe-mask-pair", "mask",
                    first_popularity["result_hash"],
                    second_popularity["result_hash"]),))

            first_random = self.select_result(
                report, case_name="first", model="random", model_seed=0)
            valid_pair = contrast_pair(
                "bounded", "model", first_popularity["result_hash"],
                first_random["result_hash"])
            consumed = [0]

            def unbounded_pairs():
                while True:
                    consumed[0] += 1
                    yield valid_pair

            with self.assertRaisesRegex(ComparisonError,
                                        "contrast-count limit"):
                create_contrast_analysis(
                    report, unbounded_pairs(), max_contrasts=1)
            self.assertEqual(consumed[0], 2)
            with self.assertRaisesRegex(ComparisonError,
                                        "preflight byte limit"):
                create_contrast_analysis(
                    report, (valid_pair,), max_analysis_bytes=1)

            deeply_nested = Path(directory) / "deep.json"
            deeply_nested.write_text("[" * 1100 + "0" + "]" * 1100,
                                     encoding="utf-8")
            with self.assertRaisesRegex(ComparisonError,
                                        "malformed|hard byte limit"):
                load_contrast_analysis(deeply_nested, report)
            duplicate_keys = Path(directory) / "duplicates.json"
            duplicate_keys.write_text(
                '{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ComparisonError, "duplicate JSON key"):
                load_contrast_analysis(duplicate_keys, report)

    def test_contrast_cli_and_rehashed_semantic_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_plan(self.make_plan(directory))
            report_path = Path(directory) / "report.json"
            analysis_path = Path(directory) / "analysis.json"
            save_report(report_path, report)
            popularity = self.select_result(
                report, case_name="lfm", candidate_policy="full_catalog",
                model="popularity", model_seed=None, k=1)
            random_result = self.select_result(
                report, case_name="lfm", candidate_policy="full_catalog",
                model="random", model_seed=0, k=1)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(comparison_cli([
                    "list-results", str(report_path),
                    "--expected-report-hash", report["report_hash"]]), 0)
            listed = json.loads(stdout.getvalue())
            self.assertEqual(listed["report_hash"], report["report_hash"])
            self.assertIn(popularity["result_hash"],
                          {value["result_hash"] for value in listed["results"]})

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(comparison_cli([
                    "contrast", str(report_path), str(analysis_path),
                    "--expected-report-hash", report["report_hash"],
                    "--pair", "popularity-v-random", "model",
                    popularity["result_hash"], random_result["result_hash"]]), 0)
            analysis_hash = json.loads(stdout.getvalue())["analysis_hash"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(comparison_cli([
                    "verify-contrast", str(report_path), str(analysis_path),
                    "--expected-report-hash", report["report_hash"],
                    "--expected-analysis-hash", analysis_hash]), 0)

            analysis = load_contrast_analysis(
                analysis_path, report, analysis_hash, report["report_hash"])
            bad = copy.deepcopy(analysis)
            contrast = bad["contrasts"][0]
            contrast["user_deltas"][0]["recall"] += 0.125
            contrast["contrast_hash"] = sha256({
                key: value for key, value in contrast.items()
                if key != "contrast_hash"})
            bad["analysis_hash"] = sha256({
                key: value for key, value in bad.items()
                if key != "analysis_hash"})
            with self.assertRaisesRegex(ComparisonError,
                                        "semantically inconsistent"):
                validate_contrast_analysis(bad, report)


if __name__ == "__main__":
    unittest.main()
