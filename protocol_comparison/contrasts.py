"""Deterministic paired contrasts over a validated comparison report."""

import json
import re

from lfm1b_protocol.canonical import canonical_json, sha256, to_primitive
from lfm1b_protocol.metrics import paired_metric_values
from research_data.hygiene import atomic_write_generated

from .plan import ComparisonError
from .runner import validate_report


CONTRAST_SCHEMA_VERSION = 1
CONTRAST_KINDS = ("model", "seed", "mask")
PAIRED_METRICS = ("recall", "ndcg", "mrr", "mean_average_precision",
                  "precision", "hit_rate", "novelty")
DEFAULT_MAX_CONTRASTS = 100
DEFAULT_MAX_PAIRED_ROWS = 100000
DEFAULT_MAX_ANALYSIS_BYTES = 64 * 1024 * 1024
MAX_CONTRASTS = 1000
MAX_PAIRED_ROWS = 1000000
MAX_ANALYSIS_BYTES = 256 * 1024 * 1024
_PAIR_KEYS = {"name", "kind", "baseline_result_hash",
              "comparison_result_hash"}
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ComparisonError(label + " must be a lowercase SHA256 digest")
    return value


def _positive_integer(value, label, maximum):
    if (isinstance(value, bool) or not isinstance(value, int) or value <= 0 or
            value > maximum):
        raise ComparisonError("%s must be an integer from 1 through %d" %
                              (label, maximum))
    return value


def _rough_json_upper_bound(value, ceiling, depth=0):
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
            total += 2 + len(str(key)) * 12
            total += _rough_json_upper_bound(item, ceiling, depth + 1)
            if total > ceiling:
                return total
        return total
    return ceiling + 1


def _validate_pair(pair):
    if not isinstance(pair, dict) or set(pair) != _PAIR_KEYS:
        raise ComparisonError("contrast pair has an invalid shape")
    name = pair["name"]
    if (not isinstance(name, str) or len(name) > 128 or
            _SAFE_LABEL.fullmatch(name) is None):
        raise ComparisonError("contrast pair name must be a safe nonempty label")
    if pair["kind"] not in CONTRAST_KINDS:
        raise ComparisonError("contrast pair kind is unsupported")
    _digest(pair["baseline_result_hash"], "baseline result hash")
    _digest(pair["comparison_result_hash"], "comparison result hash")
    if pair["baseline_result_hash"] == pair["comparison_result_hash"]:
        raise ComparisonError("contrast pair must reference two different results")
    return pair


def contrast_pair(name, kind, baseline_result_hash, comparison_result_hash):
    """Create one strict, orientation-preserving paired-contrast request."""
    return _validate_pair({
        "name": name,
        "kind": kind,
        "baseline_result_hash": baseline_result_hash,
        "comparison_result_hash": comparison_result_hash,
    })


def _normalize_pairs(pairs, maximum=MAX_CONTRASTS):
    normalized = []
    try:
        for index, value in enumerate(pairs):
            if index >= maximum:
                raise ComparisonError(
                    "contrast analysis exceeds its contrast-count limit")
            normalized.append(to_primitive(value))
    except ComparisonError:
        raise
    except (RecursionError, TypeError, ValueError) as error:
        raise ComparisonError("contrast pairs must be canonical objects: %s" % error)
    if not normalized:
        raise ComparisonError("contrast analysis requires at least one pair")
    for pair in normalized:
        _validate_pair(pair)
    normalized.sort(key=lambda value: value["name"])
    names = [value["name"] for value in normalized]
    if names != sorted(set(names)):
        raise ComparisonError("contrast pair names must be unique")
    requests = [(value["kind"], value["baseline_result_hash"],
                 value["comparison_result_hash"]) for value in normalized]
    if len(requests) != len(set(requests)):
        raise ComparisonError("contrast result pairs must not be duplicated")
    return normalized


def _limits(max_contrasts, max_paired_rows, max_analysis_bytes):
    return {
        "max_contrasts": _positive_integer(
            max_contrasts, "contrast-count limit", MAX_CONTRASTS),
        "max_paired_rows": _positive_integer(
            max_paired_rows, "paired-row limit", MAX_PAIRED_ROWS),
        "max_analysis_bytes": _positive_integer(
            max_analysis_bytes, "analysis-byte limit", MAX_ANALYSIS_BYTES),
    }


def _require_equal(baseline, comparison, fields, kind):
    for field in fields:
        if baseline[field] != comparison[field]:
            raise ComparisonError(
                "%s contrast requires matching %s" % (kind, field))


def _model_request_config(result):
    config = dict(result["model_config"])
    if result["model"] == "itemknn":
        config.pop("ordered_cooccurrence_pairs")
    return config


def _validate_compatibility(pair, baseline, comparison):
    kind = pair["kind"]
    _require_equal(
        baseline, comparison,
        ("dataset", "adapter", "target", "split", "candidate_policy", "k"),
        kind)
    exact_task_fields = (
        "case_name", "mask_configuration", "candidate_rows_hash",
        "candidate_scores", "catalog_items", "artifact_identity")
    if kind in ("model", "seed"):
        _require_equal(baseline, comparison, exact_task_fields, kind)
    if kind == "model":
        if baseline["model"] == comparison["model"]:
            raise ComparisonError(
                "model contrast requires two different model families; use a seed "
                "contrast for random-seed variation")
    elif kind == "seed":
        if baseline["model"] != "random" or comparison["model"] != "random":
            raise ComparisonError("seed contrasts currently require two random results")
        if baseline["model_seed"] == comparison["model_seed"]:
            raise ComparisonError("seed contrast requires two different model seeds")
    else:
        if baseline["candidate_policy"] != "full_catalog":
            raise ComparisonError(
                "mask contrasts require full_catalog to avoid sampled-candidate confounding")
        if baseline["model"] != comparison["model"]:
            raise ComparisonError("mask contrast requires the same model family")
        if baseline["model_seed"] != comparison["model_seed"]:
            raise ComparisonError("mask contrast requires the same model seed")
        if _model_request_config(baseline) != _model_request_config(comparison):
            raise ComparisonError("mask contrast requires the same model configuration")
        if (baseline["artifact_identity"]["population_hash"] !=
                comparison["artifact_identity"]["population_hash"]):
            raise ComparisonError(
                "mask contrast requires matching normalized population identities")
        if baseline["mask_configuration"] == comparison["mask_configuration"]:
            raise ComparisonError("mask contrast requires different mask configurations")


def _result_summary(result):
    return {
        "result_hash": result["result_hash"],
        "case_name": result["case_name"],
        "model": result["model"],
        "model_seed": result["model_seed"],
        "model_config": to_primitive(result["model_config"]),
        "mask_configuration": to_primitive(result["mask_configuration"]),
        "candidate_rows_hash": result["candidate_rows_hash"],
        "candidate_scores": result["candidate_scores"],
        "catalog_items": result["catalog_items"],
        "artifact_identity": to_primitive(result["artifact_identity"]),
        "native_metrics": to_primitive(result["metrics"]),
    }


def _zero_normalized(value):
    return 0.0 if value == 0 else value


def _build_contrast(pair, baseline, comparison, common_users):
    baseline_by_user = {value["user_id"]: value
                        for value in baseline["contributions"]}
    comparison_by_user = {value["user_id"]: value
                          for value in comparison["contributions"]}
    baseline_users = set(baseline_by_user)
    comparison_users = set(comparison_by_user)
    user_deltas = []
    for user_id in common_users:
        row = {"user_id": user_id}
        for metric in PAIRED_METRICS:
            row[metric] = _zero_normalized(
                comparison_by_user[user_id][metric] -
                baseline_by_user[user_id][metric])
        user_deltas.append(row)
    paired_metrics = {}
    for metric in PAIRED_METRICS:
        values = paired_metric_values(
            [baseline_by_user[user][metric] for user in common_users],
            [comparison_by_user[user][metric] for user in common_users])
        paired_metrics[metric] = {key: _zero_normalized(value)
                                  for key, value in values.items()}
    identity = {
        "name": pair["name"],
        "kind": pair["kind"],
        "task": {
            "dataset": baseline["dataset"],
            "adapter": baseline["adapter"],
            "target": baseline["target"],
            "split": baseline["split"],
            "candidate_policy": baseline["candidate_policy"],
            "k": baseline["k"],
            "population_hash": baseline["artifact_identity"]["population_hash"],
        },
        "baseline": _result_summary(baseline),
        "comparison": _result_summary(comparison),
        "cohort": {
            "common_users": len(common_users),
            "baseline_only_user_ids": sorted(baseline_users - comparison_users),
            "comparison_only_user_ids": sorted(comparison_users - baseline_users),
        },
        "paired_metrics": paired_metrics,
        "user_deltas": user_deltas,
    }
    return dict(identity, contrast_hash=sha256(identity))


def _analysis_identity(analysis):
    return {key: value for key, value in analysis.items()
            if key != "analysis_hash"}


def _assemble(report, pairs, limits):
    if len(pairs) > limits["max_contrasts"]:
        raise ComparisonError("contrast analysis exceeds its contrast-count limit")
    by_hash = {result["result_hash"]: result for result in report["results"]}
    selected = []
    total_rows = 0
    for pair in pairs:
        try:
            baseline = by_hash[pair["baseline_result_hash"]]
            comparison = by_hash[pair["comparison_result_hash"]]
        except KeyError as error:
            raise ComparisonError(
                "contrast pair references a result absent from the input report: %s" %
                error.args[0])
        _validate_compatibility(pair, baseline, comparison)
        baseline_users = {value["user_id"] for value in baseline["contributions"]}
        comparison_users = {value["user_id"]
                            for value in comparison["contributions"]}
        if pair["kind"] in ("model", "seed") and baseline_users != comparison_users:
            raise ComparisonError(
                "%s contrast requires identical user cohorts" % pair["kind"])
        common_users = sorted(baseline_users & comparison_users)
        if not common_users:
            raise ComparisonError("contrast pair has no common users")
        total_rows += len(common_users)
        if total_rows > limits["max_paired_rows"]:
            raise ComparisonError("contrast analysis exceeds its paired-row limit")
        estimated_bytes = (4096 + len(pairs) * 8192 + total_rows * 512)
        if estimated_bytes > limits["max_analysis_bytes"]:
            raise ComparisonError("contrast analysis exceeds its preflight byte limit")
        selected.append((pair, baseline, comparison, common_users))
    contrasts = [_build_contrast(*values) for values in selected]
    identity = {
        "schema_version": CONTRAST_SCHEMA_VERSION,
        "input_report_hash": report["report_hash"],
        "pairs": pairs,
        "limits": limits,
        "contrasts": contrasts,
    }
    analysis = dict(identity, analysis_hash=sha256(identity))
    if len(canonical_json(analysis)) > limits["max_analysis_bytes"]:
        raise ComparisonError("contrast analysis exceeds its byte limit")
    return analysis


def _validate_envelope(analysis, expected_analysis_hash=None):
    if _rough_json_upper_bound(analysis, MAX_ANALYSIS_BYTES) > MAX_ANALYSIS_BYTES:
        raise ComparisonError("contrast analysis exceeds its hard byte limit")
    required = {"schema_version", "input_report_hash", "pairs", "limits",
                "contrasts", "analysis_hash"}
    if not isinstance(analysis, dict) or set(analysis) != required:
        raise ComparisonError("contrast analysis has an invalid shape")
    if (isinstance(analysis["schema_version"], bool) or
            not isinstance(analysis["schema_version"], int) or
            analysis["schema_version"] != CONTRAST_SCHEMA_VERSION):
        raise ComparisonError("unsupported contrast analysis schema")
    _digest(analysis["input_report_hash"], "input report hash")
    _digest(analysis["analysis_hash"], "analysis hash")
    limits = analysis["limits"]
    if (not isinstance(limits, dict) or
            set(limits) != {"max_contrasts", "max_paired_rows",
                            "max_analysis_bytes"}):
        raise ComparisonError("contrast analysis limits have an invalid shape")
    validated_limits = _limits(
        limits["max_contrasts"], limits["max_paired_rows"],
        limits["max_analysis_bytes"])
    pairs = _normalize_pairs(
        analysis["pairs"], validated_limits["max_contrasts"])
    if pairs != analysis["pairs"]:
        raise ComparisonError("contrast pairs must use canonical order")
    if len(pairs) > validated_limits["max_contrasts"]:
        raise ComparisonError("contrast analysis exceeds its contrast-count limit")
    if (not isinstance(analysis["contrasts"], list) or
            len(analysis["contrasts"]) != len(pairs)):
        raise ComparisonError("contrast results have an invalid shape")
    if analysis["analysis_hash"] != sha256(_analysis_identity(analysis)):
        raise ComparisonError("contrast analysis hash mismatch")
    if expected_analysis_hash is not None:
        _digest(expected_analysis_hash, "expected analysis hash")
        if analysis["analysis_hash"] != expected_analysis_hash:
            raise ComparisonError("expected contrast analysis hash does not match")
    if len(canonical_json(analysis)) > validated_limits["max_analysis_bytes"]:
        raise ComparisonError("contrast analysis exceeds its byte limit")
    return pairs, validated_limits


def create_contrast_analysis(
        report, pairs, expected_report_hash=None,
        max_contrasts=DEFAULT_MAX_CONTRASTS,
        max_paired_rows=DEFAULT_MAX_PAIRED_ROWS,
        max_analysis_bytes=DEFAULT_MAX_ANALYSIS_BYTES):
    """Build descriptive paired deltas from explicitly selected report results."""
    limits = _limits(max_contrasts, max_paired_rows, max_analysis_bytes)
    normalized_pairs = _normalize_pairs(pairs, limits["max_contrasts"])
    report = validate_report(report, expected_report_hash)
    analysis = _assemble(report, normalized_pairs, limits)
    _validate_envelope(analysis)
    return analysis


def validate_contrast_analysis(analysis, report, expected_analysis_hash=None,
                               expected_report_hash=None):
    """Recompute a contrast artifact from its pinned input report."""
    pairs, limits = _validate_envelope(analysis, expected_analysis_hash)
    report = validate_report(report, expected_report_hash)
    if report["report_hash"] != analysis["input_report_hash"]:
        raise ComparisonError("contrast analysis references a different input report")
    expected = _assemble(report, pairs, limits)
    if analysis != expected:
        raise ComparisonError("contrast analysis is semantically inconsistent")
    return analysis


def result_summaries(report, expected_report_hash=None):
    """Return bounded selectors for constructing explicit contrast pairs."""
    report = validate_report(report, expected_report_hash)
    return [{
        "result_hash": result["result_hash"],
        "case_name": result["case_name"],
        "dataset": result["dataset"],
        "adapter": result["adapter"],
        "target": result["target"],
        "split": result["split"],
        "candidate_policy": result["candidate_policy"],
        "model": result["model"],
        "model_seed": result["model_seed"],
        "model_config": result["model_config"],
        "k": result["k"],
        "mask_configuration": result["mask_configuration"],
        "population_hash": result["artifact_identity"]["population_hash"],
    } for result in report["results"]]


def save_contrast_analysis(path, analysis, report):
    analysis = validate_contrast_analysis(analysis, report)
    atomic_write_generated(path, canonical_json(analysis))
    return analysis


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError("duplicate JSON key in contrast analysis: " + key)
        result[key] = value
    return result


def _json_constant(value):
    raise ComparisonError("non-finite JSON value in contrast analysis: " + value)


def load_contrast_analysis(path, report, expected_analysis_hash=None,
                           expected_report_hash=None,
                           max_bytes=MAX_ANALYSIS_BYTES):
    _positive_integer(max_bytes, "analysis byte limit", MAX_ANALYSIS_BYTES)
    try:
        with open(path, "rb") as stream:
            content = stream.read(max_bytes + 1)
    except OSError as error:
        raise ComparisonError("cannot read contrast analysis: %s" % error)
    if len(content) > max_bytes:
        raise ComparisonError("contrast analysis exceeds its byte limit")
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_json_pairs,
                           parse_constant=_json_constant)
    except (RecursionError, UnicodeDecodeError, ValueError, TypeError) as error:
        if isinstance(error, ComparisonError):
            raise
        raise ComparisonError("malformed contrast analysis: %s" % error)
    return validate_contrast_analysis(
        value, report, expected_analysis_hash, expected_report_hash)
