"""Deterministic execution and reporting for bounded comparison plans."""

import json
import math
import re

from lfm1b_protocol.canonical import canonical_json, sha256
from lfm1b_protocol.metrics import (RankingContribution, aggregate_metric_values,
                                    ranking_metric_values)
from research_data.hygiene import atomic_write_generated

from .adapters import load_case_artifact, mask_configuration
from .plan import (ADAPTERS, CANDIDATE_POLICIES, MAX_REPORT_BYTES, MODELS,
                   MAX_SIGNED_INTEGER, SPLITS, TARGETS, ComparisonError,
                   validate_plan)
from .scoring import (comparison_work_upper_bounds,
                       _comparison_work_upper_bounds_validated,
                       _evaluate_protocol_baseline_validated,
                       evaluate_protocol_baseline)


REPORT_SCHEMA_VERSION = 3
NUMERICAL_SEMANTICS = {
    "ranking_metrics": "lfm1b_protocol.metrics.ranking_metric_values.v1",
    "aggregate_metrics": "lfm1b_protocol.metrics.aggregate_metric_values.v1",
    "paired_metrics": "lfm1b_protocol.metrics.paired_metric_values.v1",
    "float_contract": "python_math_fsum_platform_libm_log2_v1",
}
_RESULT_KEYS = {
    "case_name", "dataset", "adapter", "target", "split", "candidate_policy",
    "mask_configuration", "model", "model_seed", "model_config", "k",
    "candidate_rows_hash", "candidate_scores", "catalog_items",
    "artifact_identity", "metrics", "contributions", "result_hash",
}
_METRIC_KEYS = {"k", "users", "recall", "ndcg", "mrr",
                "mean_average_precision", "precision", "hit_rate",
                "catalog_coverage", "novelty"}
_CONTRIBUTION_KEYS = {"user_id", "k", "relevant_items", "candidate_items",
                      "recommended_items", "hit_items", "recall", "ndcg",
                      "mrr", "mean_average_precision", "precision", "hit_rate",
                       "novelty", "hit_ranks", "recommended_item_ids",
                       "recommended_item_novelties"}
_IDENTITY_KEYS = {"adapter", "artifact_hash", "wrapper_config_hash",
                  "protocol_hash", "protocol_config_hash", "population_hash"}
_MASK_KEYS = {"split_strategy", "validation_cutoff", "test_cutoff",
              "repeat_policy", "positive_filter_horizon", "catalog_policy",
              "temporal_scope", "catalog_uses_future_information",
              "candidate_filter_uses_future_information", "globally_time_causal"}
_RESULT_OVERHEAD_BYTES = 4096
_CONTRIBUTION_OVERHEAD_BYTES = 1024
_RANKED_ITEM_BYTES = 64


def _result_identity(result):
    return {key: value for key, value in result.items() if key != "result_hash"}


def _report_identity(report):
    return {"schema_version": report["schema_version"],
            "numerical_semantics": report["numerical_semantics"],
            "plan_hash": report["plan_hash"], "results": report["results"]}


def _result_sort_key(value):
    seed = value["model_seed"]
    return (value["case_name"], value["target"], value["split"],
            value["candidate_policy"], value["model"],
            -1 if seed is None else seed, value["k"])


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ComparisonError(label + " must be a SHA256 digest")


def _integer(value, label, positive=False):
    if (isinstance(value, bool) or not isinstance(value, int) or value < 0 or
            value > MAX_SIGNED_INTEGER or (positive and value == 0)):
        raise ComparisonError(label + " must be a %sinteger" %
                              ("positive " if positive else "non-negative "))


def _finite(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value)):
        raise ComparisonError(label + " must be finite")


def _planned_result_count(plan):
    total = 0
    for case in plan["cases"]:
        model_runs = sum(len(case["random_seeds"]) if model == "random" else 1
                         for model in case["models"])
        total += (len(case["targets"]) * len(case["splits"]) *
                  len(case["candidate_policies"]) * model_runs *
                  len(case["k_values"]))
    return total


def _rough_json_upper_bound(value, ceiling, depth=0):
    """Conservative allocation-free upper bound for canonical JSON bytes."""
    if depth > 64:
        return ceiling + 1
    if value is None or isinstance(value, bool):
        return 8
    if isinstance(value, int):
        return value.bit_length() + 4
    if isinstance(value, float):
        return 32
    if isinstance(value, str):
        return len(value) * 12 + 2
    if isinstance(value, (list, tuple)):
        total = 2
        for item in value:
            total += 1 + _rough_json_upper_bound(item, ceiling, depth + 1)
            if total > ceiling:
                return total
        return total
    if isinstance(value, dict):
        total = 2
        for key, item in value.items():
            total += 1 + _rough_json_upper_bound(str(key), ceiling, depth + 1)
            total += 1 + _rough_json_upper_bound(item, ceiling, depth + 1)
            if total > ceiling:
                return total
        return total
    return ceiling + 1


def run_plan(plan, expected_plan_hash=None):
    plan = validate_plan(plan, expected_plan_hash)
    planned_results = _planned_result_count(plan)
    if planned_results > plan["limits"]["max_results"]:
        raise ComparisonError(
            "comparison plan requests %d results; limit is %d" %
            (planned_results, plan["limits"]["max_results"]))
    estimated_report_bytes = (len(canonical_json(plan)) + 4096 +
                              planned_results * _RESULT_OVERHEAD_BYTES)
    if estimated_report_bytes > plan["limits"]["max_report_bytes"]:
        raise ComparisonError("comparison report exceeds its preflight byte limit")
    # Keep preflight memory bounded: retain only compact task summaries, discard
    # each validated artifact, then reload it at the scoring boundary.
    artifact_paths = {}
    prepared_cases = []
    for case in plan["cases"]:
        loaded = load_case_artifact(case)
        artifact_paths[case["name"]] = loaded.artifact_path
        masking = mask_configuration(loaded.protocol)
        prepared_runs = []
        for target in case["targets"]:
            for split in case["splits"]:
                for candidate_policy in case["candidate_policies"]:
                    for model in case["models"]:
                        seeds = case["random_seeds"] if model == "random" else (None,)
                        for seed in seeds:
                            bounds = _comparison_work_upper_bounds_validated(
                                loaded.protocol, target, split, candidate_policy,
                                case["k_values"])
                            estimated_increment = (
                                bounds["contribution_rows"] *
                                _CONTRIBUTION_OVERHEAD_BYTES +
                                bounds["serialized_ranking_items"] *
                                _RANKED_ITEM_BYTES)
                            if (estimated_report_bytes + estimated_increment >
                                    plan["limits"]["max_report_bytes"]):
                                raise ComparisonError(
                                    "comparison report exceeds its preflight byte limit")
                            estimated_report_bytes += estimated_increment
                            prepared_runs.append((target, split, candidate_policy,
                                                  model, seed))
        prepared_cases.append((case, masking, tuple(prepared_runs)))
        del loaded
    results = []
    total_candidate_scores = 0
    total_itemknn_pairs = 0
    total_contribution_rows = 0
    total_ranking_items = 0
    for case, masking, prepared_runs in prepared_cases:
        loaded = load_case_artifact(case)
        for target, split, candidate_policy, model, seed in prepared_runs:
            remaining_scores = (plan["limits"]["max_candidate_scores"] -
                                total_candidate_scores)
            if remaining_scores <= 0:
                raise ComparisonError("comparison candidate-score limit exhausted")
            remaining_pairs = (plan["limits"]["max_itemknn_pairs"] -
                               total_itemknn_pairs)
            if model == "itemknn" and remaining_pairs <= 0:
                raise ComparisonError("comparison ItemKNN-pair limit exhausted")
            remaining_contributions = (
                plan["limits"]["max_contribution_rows"] -
                total_contribution_rows)
            if remaining_contributions <= 0:
                raise ComparisonError(
                    "comparison contribution-row limit exhausted")
            remaining_ranking = (plan["limits"]["max_ranking_items"] -
                                 total_ranking_items)
            if remaining_ranking <= 0:
                raise ComparisonError(
                    "comparison ranking-item limit exhausted")
            scored = _evaluate_protocol_baseline_validated(
                loaded.protocol, target, split, candidate_policy,
                model, case["k_values"], seed,
                case["neighbor_limit"],
                remaining_scores,
                max(1, remaining_pairs), remaining_contributions,
                remaining_ranking)
            total_candidate_scores += scored["candidate_scores"]
            total_itemknn_pairs += scored["itemknn_pairs"]
            total_contribution_rows += scored["contribution_rows"]
            total_ranking_items += scored["ranking_items"]
            for evaluation in scored["evaluations"]:
                identity = {
                    "case_name": case["name"],
                    "dataset": case["dataset"],
                    "adapter": case["adapter"],
                    "target": target,
                    "split": split,
                    "candidate_policy": candidate_policy,
                    "mask_configuration": masking,
                    "model": model,
                    "model_seed": seed,
                    "model_config": scored["model_config"],
                    "k": evaluation["k"],
                    "candidate_rows_hash": scored["candidate_rows_hash"],
                    "candidate_scores": scored["candidate_scores"],
                    "catalog_items": scored["catalog_items"],
                    "artifact_identity": dict(loaded.identity),
                    "metrics": evaluation["metrics"],
                    "contributions": evaluation["contributions"],
                }
                results.append(dict(identity, result_hash=sha256(identity)))
        del loaded
    results.sort(key=_result_sort_key)
    identity = {"schema_version": REPORT_SCHEMA_VERSION,
                "numerical_semantics": NUMERICAL_SEMANTICS,
                "plan_hash": plan["plan_hash"], "results": results}
    report = dict(identity,
                  provenance={"artifact_paths": artifact_paths},
                  report_hash=sha256(identity))
    if len(canonical_json(report)) > plan["limits"]["max_report_bytes"]:
        raise ComparisonError("comparison report exceeds its byte limit")
    return validate_report(report)


def validate_report(report, expected_report_hash=None):
    required = {"schema_version", "numerical_semantics", "plan_hash", "results",
                "provenance", "report_hash"}
    if not isinstance(report, dict) or set(report) != required:
        raise ComparisonError("comparison report has an invalid shape")
    if _rough_json_upper_bound(report, MAX_REPORT_BYTES) > MAX_REPORT_BYTES:
        raise ComparisonError("comparison report exceeds its hard byte limit")
    if (isinstance(report["schema_version"], bool) or
            not isinstance(report["schema_version"], int) or
            report["schema_version"] != REPORT_SCHEMA_VERSION):
        raise ComparisonError("unsupported comparison report schema")
    if report["numerical_semantics"] != NUMERICAL_SEMANTICS:
        raise ComparisonError("unsupported comparison numerical semantics")
    for key in ("plan_hash", "report_hash"):
        _digest(report[key], "comparison " + key)
    results = report["results"]
    if not isinstance(results, list) or not results:
        raise ComparisonError("comparison report must contain results")
    result_hashes = []
    for result in results:
        if not isinstance(result, dict) or set(result) != _RESULT_KEYS:
            raise ComparisonError("comparison result has an invalid shape")
        for key in ("case_name", "dataset"):
            if not isinstance(result[key], str) or not result[key]:
                raise ComparisonError("comparison result label is invalid")
        if (result["adapter"] not in ADAPTERS or result["target"] not in TARGETS or
                result["split"] not in SPLITS or
                result["candidate_policy"] not in CANDIDATE_POLICIES or
                result["model"] not in MODELS):
            raise ComparisonError("comparison result request fields are invalid")
        if result["model_seed"] is not None:
            if (isinstance(result["model_seed"], bool) or
                    not isinstance(result["model_seed"], int) or
                    result["model_seed"] < -MAX_SIGNED_INTEGER or
                    result["model_seed"] > MAX_SIGNED_INTEGER):
                raise ComparisonError(
                    "comparison model seed must be a signed 63-bit integer")
        if ((result["model"] == "random" and result["model_seed"] is None) or
                (result["model"] != "random" and result["model_seed"] is not None)):
            raise ComparisonError("comparison model seed contradicts its model")
        _integer(result["k"], "comparison K", positive=True)
        _integer(result["candidate_scores"], "comparison candidate-score count",
                 positive=True)
        _integer(result["catalog_items"], "comparison catalog-item count",
                 positive=True)
        _digest(result["candidate_rows_hash"], "comparison candidate-row hash")
        _digest(result["result_hash"], "comparison result hash")
        result_hashes.append(result["result_hash"])
        model_config = result["model_config"]
        if not isinstance(model_config, dict):
            raise ComparisonError("comparison model config must be an object")
        if result["model"] == "random":
            expected_model = {"algorithm": "protocol_comparison_sha256_priority_v1",
                              "random_seed": result["model_seed"]}
            if model_config != expected_model:
                raise ComparisonError("random model config is inconsistent")
        elif result["model"] == "popularity":
            if model_config != {"algorithm": "train_play_count_popularity_v1"}:
                raise ComparisonError("popularity model config is inconsistent")
        else:
            if set(model_config) != {"algorithm", "neighbor_limit",
                                     "ordered_cooccurrence_pairs"} or model_config.get(
                    "algorithm") != "binary_cosine_itemknn_v1":
                raise ComparisonError("ItemKNN model config is inconsistent")
            _integer(model_config["neighbor_limit"], "ItemKNN neighbor limit",
                     positive=True)
            _integer(model_config["ordered_cooccurrence_pairs"],
                     "ItemKNN ordered pair count")
        masking = result["mask_configuration"]
        if not isinstance(masking, dict) or set(masking) != _MASK_KEYS:
            raise ComparisonError("comparison mask configuration must be an object")
        if (masking["split_strategy"] not in
                ("per_user_last_timestamp_groups", "global_time_cutoffs") or
                masking["repeat_policy"] not in ("novel_only", "repeat_allowed") or
                masking["positive_filter_horizon"] not in
                ("as_of_split", "all_observed") or
                masking["catalog_policy"] not in ("train_observed", "all_mapped") or
                masking["temporal_scope"] not in ("global", "user_relative")):
            raise ComparisonError("comparison mask configuration is invalid")
        for key in ("catalog_uses_future_information",
                    "candidate_filter_uses_future_information", "globally_time_causal"):
            if not isinstance(masking[key], bool):
                raise ComparisonError("comparison mask causality flags must be boolean")
        for key in ("validation_cutoff", "test_cutoff"):
            value = masking[key]
            if (value is not None and
                    (isinstance(value, bool) or not isinstance(value, int) or
                     value < -MAX_SIGNED_INTEGER or value > MAX_SIGNED_INTEGER)):
                raise ComparisonError(
                    "comparison cutoff timestamps must fit signed 63-bit integers")
        if (masking["catalog_uses_future_information"] !=
                (masking["catalog_policy"] == "all_mapped") or
                masking["candidate_filter_uses_future_information"] !=
                (masking["positive_filter_horizon"] == "all_observed") or
                masking["temporal_scope"] !=
                ("global" if masking["split_strategy"] == "global_time_cutoffs"
                 else "user_relative") or
                masking["globally_time_causal"] !=
                (masking["split_strategy"] == "global_time_cutoffs" and
                 masking["catalog_policy"] == "train_observed" and
                 masking["positive_filter_horizon"] == "as_of_split")):
            raise ComparisonError("comparison mask causality metadata is inconsistent")
        identity = result["artifact_identity"]
        if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
            raise ComparisonError("comparison artifact identity is invalid")
        if identity["adapter"] != result["adapter"]:
            raise ComparisonError("comparison adapter identity mismatch")
        for key in ("protocol_hash", "protocol_config_hash", "population_hash"):
            _digest(identity[key], "comparison artifact " + key)
        for key in ("artifact_hash", "wrapper_config_hash"):
            if identity[key] is not None:
                _digest(identity[key], "comparison artifact " + key)
        if ((result["adapter"] == "lfm1b" and
             (identity["artifact_hash"] is not None or
              identity["wrapper_config_hash"] is not None)) or
                (result["adapter"] == "listenbrainz" and
                 (identity["artifact_hash"] is None or
                  identity["wrapper_config_hash"] is None))):
            raise ComparisonError("comparison wrapper identity contradicts its adapter")
        metrics = result["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != _METRIC_KEYS:
            raise ComparisonError("comparison metrics have an invalid shape")
        _integer(metrics["k"], "comparison metric K", positive=True)
        _integer(metrics["users"], "comparison metric users", positive=True)
        for key in _METRIC_KEYS - {"k", "users"}:
            _finite(metrics[key], "comparison metric " + key)
            if metrics[key] < 0 or (key != "novelty" and metrics[key] > 1):
                raise ComparisonError("comparison metric is outside its valid range")
        if result["result_hash"] != sha256(_result_identity(result)):
            raise ComparisonError("comparison result hash mismatch")
        contributions = result["contributions"]
        if not isinstance(contributions, list) or not contributions:
            raise ComparisonError("comparison contributions are invalid or unsorted")
        user_ids = []
        for contribution in contributions:
            if (not isinstance(contribution, dict) or
                    set(contribution) != _CONTRIBUTION_KEYS):
                raise ComparisonError("comparison contribution has an invalid shape")
            _integer(contribution["user_id"], "contribution user ID")
            _integer(contribution["k"], "contribution K", positive=True)
            if contribution["k"] != result["k"]:
                raise ComparisonError("comparison contribution K is inconsistent")
            user_ids.append(contribution["user_id"])
            for key in ("relevant_items", "candidate_items", "recommended_items",
                        "hit_items"):
                _integer(contribution[key], "contribution " + key,
                         positive=key in ("relevant_items", "candidate_items"))
            if contribution["recommended_items"] != min(
                    result["k"], contribution["candidate_items"]):
                raise ComparisonError(
                    "comparison contribution recommendation count is inconsistent "
                    "with K and candidates")
            if (contribution["relevant_items"] > contribution["candidate_items"] or
                    contribution["recommended_items"] > contribution["candidate_items"] or
                    contribution["recommended_items"] > result["k"] or
                    contribution["hit_items"] > contribution["relevant_items"] or
                    contribution["hit_items"] > contribution["recommended_items"]):
                raise ComparisonError("comparison contribution counts are inconsistent")
            for key in ("recall", "ndcg", "mrr", "mean_average_precision",
                        "precision", "hit_rate", "novelty"):
                _finite(contribution[key], "contribution " + key)
                if (contribution[key] < 0 or
                        (key != "novelty" and contribution[key] > 1)):
                    raise ComparisonError("comparison contribution is outside its range")
            recommended = contribution["recommended_item_ids"]
            if (not isinstance(recommended, (list, tuple)) or
                    len(recommended) != contribution["recommended_items"] or
                    len(recommended) != len(set(recommended)) or
                    any(isinstance(item, bool) or not isinstance(item, int)
                        for item in recommended)):
                raise ComparisonError("recommended contribution items are invalid")
            novelties = contribution["recommended_item_novelties"]
            if (not isinstance(novelties, (list, tuple)) or
                    len(novelties) != len(recommended)):
                raise ComparisonError("recommended contribution novelties are invalid")
            for novelty in novelties:
                _finite(novelty, "recommended item novelty")
                if novelty < 0:
                    raise ComparisonError("recommended item novelty is outside its range")
            hit_ranks = contribution["hit_ranks"]
            if (not isinstance(hit_ranks, (list, tuple)) or
                    len(hit_ranks) != contribution["hit_items"] or
                    any(isinstance(rank, bool) or not isinstance(rank, int) or
                        rank <= 0 or rank > contribution["recommended_items"]
                        for rank in hit_ranks) or
                    list(hit_ranks) != sorted(set(hit_ranks))):
                raise ComparisonError("comparison contribution hit ranks are invalid")
            expected = ranking_metric_values(
                contribution["relevant_items"], contribution["recommended_items"],
                result["k"], hit_ranks, novelties)
            if any(contribution[key] != value for key, value in expected.items()):
                raise ComparisonError("comparison contribution metrics are inconsistent")
        if user_ids != sorted(set(user_ids)):
            raise ComparisonError("comparison contributions are invalid or unsorted")
        if result["metrics"].get("k") != result["k"]:
            raise ComparisonError("comparison metric K contradicts its result")
        if result["metrics"]["users"] != len(contributions):
            raise ComparisonError("comparison metric user count is inconsistent")
        if result["candidate_scores"] != sum(
                value["candidate_items"] for value in contributions):
            raise ComparisonError("comparison candidate-score count is inconsistent")
        contribution_values = tuple(RankingContribution(**value)
                                    for value in contributions)
        expected_metrics = aggregate_metric_values(contribution_values,
                                                   result["catalog_items"])
        if any(metrics[key] != value for key, value in expected_metrics.items()):
            raise ComparisonError("comparison novelty aggregate is inconsistent")
    if results != sorted(results, key=_result_sort_key):
        raise ComparisonError("comparison results are not in canonical order")
    if len(result_hashes) != len(set(result_hashes)):
        raise ComparisonError("comparison result hashes must be unique")
    provenance = report["provenance"]
    if (not isinstance(provenance, dict) or set(provenance) != {"artifact_paths"} or
            not isinstance(provenance["artifact_paths"], dict) or
            any(not isinstance(key, str) or not isinstance(value, str)
                for key, value in provenance["artifact_paths"].items())):
        raise ComparisonError("comparison report provenance is invalid")
    if set(provenance["artifact_paths"]) != {
            value["case_name"] for value in results}:
        raise ComparisonError("comparison artifact-path provenance is incomplete")
    if report["report_hash"] != sha256(_report_identity(report)):
        raise ComparisonError("comparison report hash mismatch")
    if expected_report_hash is not None and report["report_hash"] != expected_report_hash:
        raise ComparisonError("expected comparison report hash does not match")
    return report


def save_report(path, report):
    report = validate_report(report)
    content = canonical_json(report)
    if len(content) > MAX_REPORT_BYTES:
        raise ComparisonError("comparison report exceeds its hard byte limit")
    atomic_write_generated(path, content)
    return report


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError("duplicate JSON key in comparison report: " + key)
        result[key] = value
    return result


def _json_constant(value):
    raise ComparisonError("non-finite JSON value in comparison report: " + value)


def load_report(path, expected_report_hash=None, max_bytes=MAX_REPORT_BYTES):
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ComparisonError("report byte limit must be positive")
    try:
        with open(path, "rb") as stream:
            content = stream.read(max_bytes + 1)
    except OSError as error:
        raise ComparisonError("cannot read comparison report: %s" % error)
    if len(content) > max_bytes:
        raise ComparisonError("comparison report exceeds its byte limit")
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_json_pairs,
                           parse_constant=_json_constant)
    except (RecursionError, UnicodeDecodeError, ValueError, TypeError) as error:
        if isinstance(error, ComparisonError):
            raise
        raise ComparisonError("malformed comparison report: %s" % error)
    return validate_report(value, expected_report_hash)
