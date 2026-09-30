"""Canonical, path-independent plans for bounded local comparisons."""

import json
import re

from lfm1b_protocol.canonical import canonical_json, sha256, to_primitive
from research_data.hygiene import atomic_write_generated


PLAN_SCHEMA_VERSION = 1
DEFAULT_MAX_CANDIDATE_SCORES = 1000000
DEFAULT_MAX_ITEMKNN_PAIRS = 1000000
DEFAULT_MAX_RESULTS = 1000
DEFAULT_MAX_CONTRIBUTION_ROWS = 100000
DEFAULT_MAX_RANKING_ITEMS = 5000000
DEFAULT_MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_REPORT_BYTES = 256 * 1024 * 1024
MAX_PLAN_BYTES = 4 * 1024 * 1024
ADAPTERS = ("lfm1b", "listenbrainz")
TARGETS = ("artist", "album", "track")
SPLITS = ("validation", "test")
CANDIDATE_POLICIES = ("fixed_sampled", "full_catalog")
MODELS = ("random", "popularity", "itemknn")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HASH_KEYS = ("artifact_hash", "wrapper_config_hash", "protocol_hash",
              "protocol_config_hash")
MAX_SIGNED_INTEGER = (1 << 63) - 1


class ComparisonError(ValueError):
    pass


def _positive_integer(value, label):
    if (isinstance(value, bool) or not isinstance(value, int) or value <= 0 or
            value > MAX_SIGNED_INTEGER):
        raise ComparisonError(label + " must be a positive signed 63-bit integer")
    return value


def _integer(value, label):
    if (isinstance(value, bool) or not isinstance(value, int) or
            value < -MAX_SIGNED_INTEGER or value > MAX_SIGNED_INTEGER):
        raise ComparisonError(label + " must be a signed 63-bit integer")
    return value


def _digest(value, label, optional=False):
    if optional and value is None:
        return None
    if (not isinstance(value, str) or len(value) != 64 or
            re.fullmatch(r"[0-9a-f]{64}", value) is None):
        raise ComparisonError(label + " must be a lowercase SHA256 digest")
    return value


def _canonical_subset(value, allowed, label):
    if not isinstance(value, list) or not value:
        raise ComparisonError(label + " must be a nonempty list")
    if any(item not in allowed for item in value) or len(value) != len(set(value)):
        raise ComparisonError(label + " contains an invalid or duplicate value")
    expected = [item for item in allowed if item in value]
    if value != expected:
        raise ComparisonError(label + " must use canonical order")
    return value


def _sorted_integers(value, label, positive=False):
    if not isinstance(value, list) or not value:
        raise ComparisonError(label + " must be a nonempty list")
    for item in value:
        (_positive_integer if positive else _integer)(item, label + " value")
    if value != sorted(set(value)):
        raise ComparisonError(label + " must be sorted and unique")
    return value


def _case_identity(case):
    return {key: value for key, value in case.items() if key != "artifact_path"}


def _plan_identity(plan):
    return {"schema_version": plan["schema_version"],
            "cases": [_case_identity(value) for value in plan["cases"]],
            "limits": plan["limits"]}


def validate_case(case):
    required = {"name", "dataset", "adapter", "artifact_path", "expected_hashes",
                "targets", "splits", "candidate_policies", "models", "k_values",
                "random_seeds", "neighbor_limit"}
    if not isinstance(case, dict) or set(case) != required:
        raise ComparisonError("comparison case has an invalid shape")
    for key in ("name", "dataset"):
        if (not isinstance(case[key], str) or len(case[key]) > 128 or
                _SAFE_LABEL.fullmatch(case[key]) is None):
            raise ComparisonError("case %s must be a safe nonempty label" % key)
    if case["adapter"] not in ADAPTERS:
        raise ComparisonError("case adapter is unsupported")
    if (not isinstance(case["artifact_path"], str) or not case["artifact_path"] or
            len(case["artifact_path"]) > 4096 or "\0" in case["artifact_path"]):
        raise ComparisonError("case artifact path must be a nonempty local path")
    hashes = case["expected_hashes"]
    if not isinstance(hashes, dict) or set(hashes) != set(_HASH_KEYS):
        raise ComparisonError("case expected hashes have an invalid shape")
    _digest(hashes["protocol_hash"], "expected protocol hash")
    _digest(hashes["protocol_config_hash"], "expected protocol config hash")
    if case["adapter"] == "lfm1b":
        if hashes["artifact_hash"] is not None or hashes["wrapper_config_hash"] is not None:
            raise ComparisonError("LFM cases cannot declare wrapper hashes")
    else:
        _digest(hashes["artifact_hash"], "expected wrapper artifact hash")
        _digest(hashes["wrapper_config_hash"], "expected wrapper config hash")
    _canonical_subset(case["targets"], TARGETS, "case targets")
    _canonical_subset(case["splits"], SPLITS, "case splits")
    _canonical_subset(case["candidate_policies"], CANDIDATE_POLICIES,
                      "case candidate policies")
    _canonical_subset(case["models"], MODELS, "case models")
    _sorted_integers(case["k_values"], "case K values", positive=True)
    if "random" in case["models"]:
        _sorted_integers(case["random_seeds"], "case random seeds")
    elif case["random_seeds"] != []:
        raise ComparisonError("random seeds must be empty when random is not requested")
    if "itemknn" in case["models"]:
        _positive_integer(case["neighbor_limit"], "case neighbor limit")
    elif case["neighbor_limit"] is not None:
        raise ComparisonError("neighbor limit must be null when ItemKNN is not requested")
    return case


def validate_plan(plan, expected_plan_hash=None):
    required = {"schema_version", "cases", "limits", "plan_hash"}
    if not isinstance(plan, dict) or set(plan) != required:
        raise ComparisonError("comparison plan has an invalid shape")
    if (isinstance(plan["schema_version"], bool) or
            not isinstance(plan["schema_version"], int) or
            plan["schema_version"] != PLAN_SCHEMA_VERSION):
        raise ComparisonError("unsupported comparison plan schema")
    if not isinstance(plan["cases"], list) or not plan["cases"]:
        raise ComparisonError("comparison plan must contain at least one case")
    for case in plan["cases"]:
        validate_case(case)
    names = [case["name"] for case in plan["cases"]]
    if names != sorted(set(names)):
        raise ComparisonError("comparison cases must have unique canonical names")
    limits = plan["limits"]
    if (not isinstance(limits, dict) or
            set(limits) != {"max_candidate_scores", "max_itemknn_pairs",
                           "max_results", "max_contribution_rows",
                           "max_ranking_items", "max_report_bytes"}):
        raise ComparisonError("comparison limits have an invalid shape")
    _positive_integer(limits["max_candidate_scores"], "candidate-score limit")
    _positive_integer(limits["max_itemknn_pairs"], "ItemKNN-pair limit")
    _positive_integer(limits["max_results"], "result-count limit")
    _positive_integer(limits["max_contribution_rows"], "contribution-row limit")
    _positive_integer(limits["max_ranking_items"], "ranking-item limit")
    _positive_integer(limits["max_report_bytes"], "report-byte limit")
    if limits["max_report_bytes"] > MAX_REPORT_BYTES:
        raise ComparisonError("report-byte limit exceeds the hard safety ceiling")
    _digest(plan["plan_hash"], "plan hash")
    if plan["plan_hash"] != sha256(_plan_identity(plan)):
        raise ComparisonError("comparison plan hash mismatch")
    if expected_plan_hash is not None and plan["plan_hash"] != expected_plan_hash:
        raise ComparisonError("expected comparison plan hash does not match")
    return plan


def create_plan(cases, max_candidate_scores=DEFAULT_MAX_CANDIDATE_SCORES,
                max_itemknn_pairs=DEFAULT_MAX_ITEMKNN_PAIRS,
                max_results=DEFAULT_MAX_RESULTS,
                max_contribution_rows=DEFAULT_MAX_CONTRIBUTION_ROWS,
                max_ranking_items=DEFAULT_MAX_RANKING_ITEMS,
                max_report_bytes=DEFAULT_MAX_REPORT_BYTES):
    normalized = [to_primitive(value) for value in cases]
    if any(not isinstance(value, dict) for value in normalized):
        raise ComparisonError("comparison cases must be objects")
    normalized.sort(key=lambda value: value.get("name", ""))
    identity = {"schema_version": PLAN_SCHEMA_VERSION, "cases": normalized,
                "limits": {"max_candidate_scores": max_candidate_scores,
                           "max_itemknn_pairs": max_itemknn_pairs,
                           "max_results": max_results,
                           "max_contribution_rows": max_contribution_rows,
                           "max_ranking_items": max_ranking_items,
                           "max_report_bytes": max_report_bytes}}
    plan = dict(identity, plan_hash=sha256(
        {"schema_version": identity["schema_version"],
         "cases": [_case_identity(value) for value in identity["cases"]],
         "limits": identity["limits"]}))
    return validate_plan(plan)


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError("duplicate JSON key in comparison plan: " + key)
        result[key] = value
    return result


def _json_constant(value):
    raise ComparisonError("non-finite JSON value in comparison plan: " + value)


def load_plan(path, expected_plan_hash=None, max_bytes=MAX_PLAN_BYTES):
    _positive_integer(max_bytes, "plan byte limit")
    try:
        with open(path, "rb") as stream:
            content = stream.read(max_bytes + 1)
    except OSError as error:
        raise ComparisonError("cannot read comparison plan: %s" % error)
    if len(content) > max_bytes:
        raise ComparisonError("comparison plan exceeds its byte limit")
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_json_pairs,
                           parse_constant=_json_constant)
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        if isinstance(error, ComparisonError):
            raise
        raise ComparisonError("malformed comparison plan: %s" % error)
    return validate_plan(value, expected_plan_hash)


def save_plan(path, plan):
    plan = validate_plan(plan)
    atomic_write_generated(path, canonical_json(plan))
    return plan
