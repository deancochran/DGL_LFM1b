import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lfm1b_protocol.canonical import sha256
from protocol_comparison import (
    ComparisonError,
    create_contrast_analysis_from_matrix_plan,
    create_mask_matrix_spec,
    create_matrix_comparison_plan,
    load_mask_matrix_spec,
    load_matrix_contrast_plan,
    load_matrix_index,
    load_plan,
    load_report,
    matrix_preview,
    prepare_mask_matrix,
    resolve_matrix_contrast_plan,
    run_plan,
    save_mask_matrix_spec,
    save_matrix_contrast_plan,
    save_matrix_index,
    save_plan,
    save_report,
    validate_mask_matrix_spec,
    validate_matrix_contrast_plan,
    validate_matrix_index,
)
from protocol_comparison.cli import main as comparison_cli


ROOT = Path(__file__).resolve().parent.parent
LFM_FIXTURE = ROOT / "examples" / "tiny_events.dat"
LISTENBRAINZ_FIXTURE = ROOT / "examples" / "listenbrainz_synthetic.listens"


class ProtocolMatrixTests(unittest.TestCase):
    def lfm_spec(self, source=LFM_FIXTURE, **overrides):
        values = dict(
            name="local", dataset="lfm1b", adapter="lfm1b",
            source_path=str(source), source_config={"has_header": False},
            split={"name": "global", "strategy": "global_time_cutoffs",
                   "validation_cutoff": 2, "test_cutoff": 3},
            repeat_policies=("repeat_allowed",),
            positive_filter_horizons=("as_of_split",),
            catalog_policies=("train_observed", "all_mapped"),
            targets=("track",), splits=("validation",),
            candidate_policies=("full_catalog",), models=("popularity",),
            k_values=(1,), random_seeds=(), neighbor_limit=1,
            sampled_negatives=3,
            limits={"max_candidate_scores": 1000,
                    "max_itemknn_pairs": 1000,
                    "max_contrasts": 20},
        )
        values.update(overrides)
        return create_mask_matrix_spec(**values)

    def listenbrainz_spec(self, source=LISTENBRAINZ_FIXTURE):
        return create_mask_matrix_spec(
            "listenbrainz", "listenbrainz-synthetic", "listenbrainz",
            str(source), {
                "dump_id": "synthetic", "dump_type": "full",
                "archive_sha256": "a" * 64,
                "member_path": "synthetic/member.listens",
                "track_identity_policy": "server_mapped_musicbrainz",
                "album_entity": "release_group",
            },
            {"name": "global", "strategy": "global_time_cutoffs",
             "validation_cutoff": 20, "test_cutoff": 30},
            repeat_policies=("repeat_allowed",),
            positive_filter_horizons=("as_of_split", "all_observed"),
            catalog_policies=("all_mapped",), targets=("track",),
            splits=("validation", "test"),
            candidate_policies=("full_catalog",), models=("popularity",),
            k_values=(1,), random_seeds=(), neighbor_limit=1,
            sampled_negatives=3,
            limits={"max_candidate_scores": 1000,
                    "max_itemknn_pairs": 1000,
                    "max_contrasts": 10})

    def test_lfm_matrix_end_to_end_is_deterministic_and_relocatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.lfm_spec()
            preview = matrix_preview(spec, spec["spec_hash"])
            self.assertEqual(preview["variants"], 2)
            self.assertEqual(preview["planned_results"], 2)
            self.assertEqual(preview["planned_contrasts"], 1)

            spec_path = root / "spec.json"
            index_path = root / "index.json"
            plan_path = root / "plan.json"
            report_path = root / "report.json"
            contrast_path = root / "contrast-plan.json"
            save_mask_matrix_spec(spec_path, spec)
            self.assertEqual(load_mask_matrix_spec(
                spec_path, spec["spec_hash"]), spec)

            index = prepare_mask_matrix(
                spec, root / "artifacts", spec["spec_hash"])
            save_matrix_index(index_path, index, spec)
            self.assertEqual(load_matrix_index(
                index_path, spec, index["index_hash"], spec["spec_hash"]),
                index)
            self.assertEqual(len({
                value["artifact_identity"]["population_hash"]
                for value in index["entries"]}), 1)

            plan = create_matrix_comparison_plan(
                spec, index, spec["spec_hash"], index["index_hash"])
            save_plan(plan_path, plan)
            report = run_plan(plan, plan["plan_hash"])
            save_report(report_path, report)
            contrast_plan = resolve_matrix_contrast_plan(
                spec, index, plan, report, spec["spec_hash"],
                index["index_hash"], plan["plan_hash"],
                report["report_hash"])
            save_matrix_contrast_plan(
                contrast_path, contrast_plan, spec, index, plan, report)
            self.assertEqual(load_matrix_contrast_plan(
                contrast_path, spec, index, plan, report,
                contrast_plan["contrast_plan_hash"], report["report_hash"]),
                contrast_plan)
            self.assertEqual(len(contrast_plan["pairs"]), 1)
            selector = contrast_plan["selectors"][0]
            self.assertEqual(selector["axis"], "catalog_policy")
            self.assertIn(".train", selector["baseline_case_name"])
            self.assertIn(".mapped", selector["comparison_case_name"])
            analysis = create_contrast_analysis_from_matrix_plan(
                contrast_plan, spec, index, plan, report,
                contrast_plan["contrast_plan_hash"], report["report_hash"])
            self.assertEqual(len(analysis["contrasts"]), 1)

            second_plan = create_matrix_comparison_plan(spec, index)
            second_contrast = resolve_matrix_contrast_plan(
                spec, index, second_plan, run_plan(second_plan))
            self.assertEqual(plan, second_plan)
            self.assertEqual(contrast_plan, second_contrast)

            relocated_spec = copy.deepcopy(spec)
            relocated_spec["source_path"] = "/relocated/events.dat"
            self.assertEqual(validate_mask_matrix_spec(
                relocated_spec)["spec_hash"], spec["spec_hash"])
            relocated_index = copy.deepcopy(index)
            relocated_index["provenance"] = {
                "source_path": "/relocated/events.dat",
                "artifact_root": "/relocated/artifacts",
            }
            for entry in relocated_index["entries"]:
                entry["artifact_path"] = "/relocated/" + entry["case_name"]
            self.assertEqual(validate_matrix_index(
                relocated_index, relocated_spec)["index_hash"],
                index["index_hash"])

    def test_matrix_rewinds_reused_root_descriptor_before_final_listing(self):
        """The empty-root check and final integrity check share a directory FD."""
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("protocol_comparison.matrix.os.lseek",
                            wraps=os.lseek) as rewind:
                prepare_mask_matrix(self.lfm_spec(), Path(directory) / "artifacts")
            self.assertTrue(any(
                call.args[1:] == (0, os.SEEK_SET) for call in rewind.call_args_list))

    def test_listenbrainz_horizon_matrix_skips_redundant_test_contrast(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.listenbrainz_spec()
            self.assertEqual(matrix_preview(spec)["planned_results"], 4)
            self.assertEqual(matrix_preview(spec)["planned_contrasts"], 1)
            index = prepare_mask_matrix(spec, Path(directory) / "artifacts")
            plan = create_matrix_comparison_plan(spec, index)
            report = run_plan(plan)
            contrast_plan = resolve_matrix_contrast_plan(
                spec, index, plan, report)
            self.assertEqual(len(contrast_plan["pairs"]), 1)
            self.assertEqual(contrast_plan["selectors"][0]["axis"],
                             "positive_filter_horizon")
            self.assertEqual(contrast_plan["selectors"][0]["split"],
                             "validation")
            self.assertEqual(len({
                entry["artifact_identity"]["population_hash"]
                for entry in index["entries"]}), 1)

    def test_matrix_limits_roots_and_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ComparisonError,
                                        "planned-contrast limit"):
                self.lfm_spec(limits={"max_contrasts": 1},
                              repeat_policies=("novel_only", "repeat_allowed"),
                              positive_filter_horizons=("as_of_split",
                                                        "all_observed"),
                              catalog_policies=("train_observed", "all_mapped"))
            with self.assertRaisesRegex(ComparisonError, "no eligible"):
                self.lfm_spec(candidate_policies=("fixed_sampled",))
            with self.assertRaisesRegex(ComparisonError, "at least two"):
                self.lfm_spec(catalog_policies=("all_mapped",))
            bounded_source = self.lfm_spec(limits={"max_source_bytes": 1})
            with mock.patch("protocol_comparison.matrix._prepare_variant") as prepare:
                with self.assertRaisesRegex(ComparisonError, "source exceeds"):
                    prepare_mask_matrix(bounded_source, root / "too-large")
                prepare.assert_not_called()
            with self.assertRaisesRegex(ValueError, "event limit"):
                prepare_mask_matrix(
                    self.lfm_spec(limits={"max_events": 1}),
                    root / "too-many-events")
            with self.assertRaisesRegex(ValueError, "candidate-item limit"):
                prepare_mask_matrix(
                    self.lfm_spec(limits={
                        "max_preparation_candidate_items": 1,
                        "max_artifact_bytes": 134217728,
                    }, sampled_negatives=1), root / "too-many-candidates")
            with self.assertRaisesRegex(ComparisonError, "artifact exceeds"):
                prepare_mask_matrix(
                    self.lfm_spec(limits={"max_artifact_bytes": 1}),
                    root / "artifact-too-large")
            with self.assertRaisesRegex(ComparisonError, "aggregate byte limit"):
                prepare_mask_matrix(
                    self.lfm_spec(limits={
                        "max_artifact_bytes": 134217728,
                        "max_total_artifact_bytes": 1,
                    }), root / "matrix-too-large")

            spec = self.lfm_spec()
            occupied = root / "occupied"
            occupied.mkdir()
            marker = occupied / "belongs-to-user"
            marker.write_text("preserve", encoding="utf-8")
            with mock.patch("protocol_comparison.matrix._prepare_variant") as prepare:
                with self.assertRaisesRegex(ComparisonError, "must be empty"):
                    prepare_mask_matrix(spec, occupied)
                prepare.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

            victim = root / "victim"
            victim.mkdir()
            symlink = root / "artifacts-link"
            symlink.symlink_to(victim, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                prepare_mask_matrix(spec, symlink)

            raced_root = root / "raced"
            expected_case = matrix_preview(spec)["case_names"][0]
            original_empty = __import__(
                "protocol_comparison.matrix", fromlist=["_empty_generated_root"]
            )._empty_generated_root

            def inject_case(path):
                resolved, descriptor = original_empty(path)
                Path(resolved, expected_case).mkdir()
                return resolved, descriptor

            with mock.patch("protocol_comparison.matrix._empty_generated_root",
                            side_effect=inject_case):
                with self.assertRaisesRegex(ComparisonError,
                                            "case output already exists"):
                    prepare_mask_matrix(spec, raced_root)

            index = prepare_mask_matrix(spec, root / "artifacts")
            protected_artifact = Path(
                index["entries"][0]["artifact_path"]) / "protocol.json"
            protected_content = protected_artifact.read_bytes()
            with self.assertRaisesRegex(ComparisonError,
                                        "must not be inside"):
                save_matrix_index(protected_artifact, index, spec)
            self.assertEqual(protected_artifact.read_bytes(), protected_content)
            bad = copy.deepcopy(index)
            bad["entries"][0]["artifact_identity"]["protocol_hash"] = "0" * 64
            bad["index_hash"] = sha256({
                "schema_version": bad["schema_version"],
                "spec_hash": bad["spec_hash"],
                "entries": [{key: item for key, item in value.items()
                             if key != "artifact_path"}
                            for value in bad["entries"]],
            })
            validate_matrix_index(bad, spec)
            with self.assertRaisesRegex(ComparisonError, "no longer matches"):
                create_matrix_comparison_plan(spec, bad)

            plan = create_matrix_comparison_plan(spec, index)
            report = run_plan(plan)
            contrast_plan = resolve_matrix_contrast_plan(spec, index, plan, report)
            altered = copy.deepcopy(contrast_plan)
            altered["pairs"][0]["baseline_result_hash"] = "0" * 64
            altered["contrast_plan_hash"] = sha256({
                key: value for key, value in altered.items()
                if key != "contrast_plan_hash"})
            with self.assertRaisesRegex(ComparisonError,
                                        "semantically inconsistent|absent"):
                validate_matrix_contrast_plan(
                    altered, spec, index, plan, report)
            reversed_plan = copy.deepcopy(contrast_plan)
            pair = reversed_plan["pairs"][0]
            pair["baseline_result_hash"], pair["comparison_result_hash"] = (
                pair["comparison_result_hash"], pair["baseline_result_hash"])
            reversed_plan["contrast_plan_hash"] = sha256({
                key: value for key, value in reversed_plan.items()
                if key != "contrast_plan_hash"})
            with self.assertRaisesRegex(ComparisonError,
                                        "semantically inconsistent"):
                create_contrast_analysis_from_matrix_plan(
                    reversed_plan, spec, index, plan, report)

    def test_matrix_cli_complete_local_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            index_path = root / "index.json"
            plan_path = root / "plan.json"
            report_path = root / "report.json"
            contrast_plan_path = root / "contrast-plan.json"
            analysis_path = root / "analysis.json"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(comparison_cli([
                    "matrix-init", str(spec_path), "--name", "cli",
                    "--dataset", "lfm1b", "--adapter", "lfm1b",
                    "--source", str(LFM_FIXTURE), "--split-name", "global",
                    "--split-strategy", "global_time_cutoffs",
                    "--validation-cutoff", "2", "--test-cutoff", "3",
                    "--repeat-policy", "repeat_allowed",
                    "--positive-filter-horizon", "as_of_split",
                    "--catalog-policy", "train_observed",
                    "--catalog-policy", "all_mapped", "--target", "track",
                    "--split", "validation", "--candidate-policy",
                    "full_catalog", "--model", "popularity", "-k", "1",
                    "--sampled-negatives", "3", "--max-candidate-scores",
                    "1000", "--max-itemknn-pairs", "1000"]), 0)
            spec_hash = json.loads(stdout.getvalue())["spec_hash"]

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(comparison_cli([
                    "matrix-prepare", str(spec_path), str(root / "artifacts"),
                    str(index_path), "--expected-spec-hash", spec_hash]), 0)
            index_hash = json.loads(stdout.getvalue())["index_hash"]

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(comparison_cli([
                    "matrix-plan", str(spec_path), str(index_path), str(plan_path),
                    "--expected-spec-hash", spec_hash,
                    "--expected-index-hash", index_hash]), 0)
            plan_hash = json.loads(stdout.getvalue())["plan_hash"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(comparison_cli([
                    "run", str(plan_path), str(report_path),
                    "--expected-plan-hash", plan_hash]), 0)
            report = load_report(report_path)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(comparison_cli([
                    "matrix-resolve-contrasts", str(spec_path), str(index_path),
                    str(plan_path), str(report_path), str(contrast_plan_path),
                    "--expected-spec-hash", spec_hash,
                    "--expected-index-hash", index_hash,
                    "--expected-plan-hash", plan_hash,
                    "--expected-report-hash", report["report_hash"]]), 0)
            contrast_hash = json.loads(stdout.getvalue())["contrast_plan_hash"]

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(comparison_cli([
                    "matrix-contrast", str(spec_path), str(index_path),
                    str(plan_path), str(report_path), str(contrast_plan_path),
                    str(analysis_path), "--expected-spec-hash", spec_hash,
                    "--expected-index-hash", index_hash,
                    "--expected-plan-hash", plan_hash,
                    "--expected-report-hash", report["report_hash"],
                    "--expected-contrast-plan-hash", contrast_hash]), 0)
            self.assertEqual(json.loads(stdout.getvalue())["contrasts"], 1)
            self.assertTrue(analysis_path.is_file())
            self.assertEqual(load_plan(plan_path, plan_hash)["plan_hash"], plan_hash)


if __name__ == "__main__":
    unittest.main()
