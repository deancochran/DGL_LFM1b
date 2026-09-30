"""Bounded local preparation and planning for protocol mask matrices."""

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from lfm1b_protocol.artifacts import (
    SCHEMA_VERSION as BUNDLE_SCHEMA_VERSION,
    prepare_protocol_artifact,
    validate_protocol_artifact,
)
from lfm1b_protocol.canonical import canonical_json, sha256, to_primitive
from lfm1b_protocol.io import read_listening_events_with_provenance
from listenbrainz_protocol.core import prepare as prepare_listenbrainz
from listenbrainz_protocol.core import validate as validate_listenbrainz
from research_data.hygiene import (atomic_write_generated,
                                   open_generated_directory)

from .adapters import inspect_artifact, mask_configuration
from .contrasts import (DEFAULT_MAX_ANALYSIS_BYTES, DEFAULT_MAX_CONTRASTS,
                        DEFAULT_MAX_PAIRED_ROWS, MAX_ANALYSIS_BYTES,
                        MAX_CONTRASTS, MAX_PAIRED_ROWS,
                        create_contrast_analysis)
from .plan import (ADAPTERS, CANDIDATE_POLICIES,
                   DEFAULT_MAX_CANDIDATE_SCORES,
                   DEFAULT_MAX_CONTRIBUTION_ROWS,
                   DEFAULT_MAX_ITEMKNN_PAIRS, DEFAULT_MAX_RANKING_ITEMS,
                   DEFAULT_MAX_REPORT_BYTES, DEFAULT_MAX_RESULTS,
                   MAX_REPORT_BYTES, MAX_SIGNED_INTEGER, MODELS, SPLITS,
                   TARGETS, ComparisonError, create_plan, validate_plan)
from .runner import validate_report


MATRIX_SPEC_SCHEMA_VERSION = 1
MATRIX_INDEX_SCHEMA_VERSION = 1
MATRIX_CONTRAST_PLAN_SCHEMA_VERSION = 1
DEFAULT_MAX_VARIANTS = 32
MAX_VARIANTS = 64
DEFAULT_MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024 * 1024
MAX_SAMPLED_NEGATIVES = 1000000
DEFAULT_MAX_EVENTS = 1000000
DEFAULT_MAX_USERS = 100000
DEFAULT_MAX_ITEMS_PER_TARGET = 1000000
DEFAULT_MAX_PREPARATION_CANDIDATE_ROWS = 100000
DEFAULT_MAX_PREPARATION_CANDIDATE_ITEMS = 5000000
DEFAULT_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_TOTAL_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_EVENTS = 10000000
MAX_USERS = 1000000
MAX_ITEMS_PER_TARGET = 5000000
MAX_PREPARATION_CANDIDATE_ROWS = 1000000
MAX_PREPARATION_CANDIDATE_ITEMS = 50000000
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_TOTAL_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
MAX_MATRIX_BYTES = 4 * 1024 * 1024
MAX_CONTRAST_PLAN_BYTES = 16 * 1024 * 1024
REPEAT_POLICIES = ("novel_only", "repeat_allowed")
POSITIVE_FILTER_HORIZONS = ("as_of_split", "all_observed")
CATALOG_POLICIES = ("train_observed", "all_mapped")
SPLIT_STRATEGIES = ("per_user_last_timestamp_groups", "global_time_cutoffs")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_IDENTITY_KEYS = {"adapter", "artifact_hash", "wrapper_config_hash",
                  "protocol_hash", "protocol_config_hash", "population_hash"}
_AXIS_ORIENTATIONS = (
    ("repeat_policy", "novel_only", "repeat_allowed"),
    ("positive_filter_horizon", "as_of_split", "all_observed"),
    ("catalog_policy", "train_observed", "all_mapped"),
)
_ALIASES = {
    "novel_only": "novel",
    "repeat_allowed": "repeat",
    "as_of_split": "asof",
    "all_observed": "observed",
    "train_observed": "train",
    "all_mapped": "mapped",
}


def _safe_label(value, label, maximum=64):
    if (not isinstance(value, str) or not value or len(value) > maximum or
            _SAFE_LABEL.fullmatch(value) is None):
        raise ComparisonError(label + " must be a safe nonempty label")
    return value


def _local_path(value, label):
    if (not isinstance(value, str) or not value or len(value) > 4096 or
            "\0" in value):
        raise ComparisonError(label + " must be a nonempty local path")
    return value


def _digest(value, label, optional=False):
    if optional and value is None:
        return value
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ComparisonError(label + " must be a lowercase SHA256 digest")
    return value


def _integer(value, label, positive=False, nonnegative=False, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, int) or
            value < -MAX_SIGNED_INTEGER or value > MAX_SIGNED_INTEGER or
            (positive and value <= 0) or (nonnegative and value < 0) or
            (maximum is not None and value > maximum)):
        raise ComparisonError(label + " is outside its valid integer range")
    return value


def _ordered(values, allowed, label):
    if not isinstance(values, list) or not values:
        raise ComparisonError(label + " must be a nonempty list")
    if len(values) != len(set(values)) or any(value not in allowed for value in values):
        raise ComparisonError(label + " contains invalid or duplicate values")
    expected = [value for value in allowed if value in values]
    if values != expected:
        raise ComparisonError(label + " must use canonical order")
    return values


def _sorted_integers(values, label, positive=False):
    if not isinstance(values, list) or not values:
        raise ComparisonError(label + " must be a nonempty list")
    for value in values:
        _integer(value, label + " value", positive=positive)
    if values != sorted(set(values)):
        raise ComparisonError(label + " must be sorted and unique")
    return values


def _canonical_values(values, allowed, label):
    values = list(values)
    if not values or len(values) != len(set(values)) or any(
            value not in allowed for value in values):
        raise ComparisonError(label + " contains invalid or duplicate values")
    return [value for value in allowed if value in values]


def _matrix_limits(limits=None):
    defaults = {
        "max_variants": DEFAULT_MAX_VARIANTS,
        "max_source_bytes": DEFAULT_MAX_SOURCE_BYTES,
        "max_events": DEFAULT_MAX_EVENTS,
        "max_users": DEFAULT_MAX_USERS,
        "max_items_per_target": DEFAULT_MAX_ITEMS_PER_TARGET,
        "max_preparation_candidate_rows": DEFAULT_MAX_PREPARATION_CANDIDATE_ROWS,
        "max_preparation_candidate_items": DEFAULT_MAX_PREPARATION_CANDIDATE_ITEMS,
        "max_artifact_bytes": DEFAULT_MAX_ARTIFACT_BYTES,
        "max_total_artifact_bytes": DEFAULT_MAX_TOTAL_ARTIFACT_BYTES,
        "max_candidate_scores": DEFAULT_MAX_CANDIDATE_SCORES,
        "max_itemknn_pairs": DEFAULT_MAX_ITEMKNN_PAIRS,
        "max_results": DEFAULT_MAX_RESULTS,
        "max_contribution_rows": DEFAULT_MAX_CONTRIBUTION_ROWS,
        "max_ranking_items": DEFAULT_MAX_RANKING_ITEMS,
        "max_report_bytes": DEFAULT_MAX_REPORT_BYTES,
        "max_contrasts": DEFAULT_MAX_CONTRASTS,
        "max_paired_rows": DEFAULT_MAX_PAIRED_ROWS,
        "max_analysis_bytes": DEFAULT_MAX_ANALYSIS_BYTES,
    }
    if limits is not None:
        if not isinstance(limits, dict) or set(limits) - set(defaults):
            raise ComparisonError("matrix limits contain unsupported fields")
        defaults.update(limits)
    _integer(defaults["max_variants"], "matrix variant limit", positive=True,
             maximum=MAX_VARIANTS)
    _integer(defaults["max_source_bytes"], "matrix source byte limit",
             positive=True, maximum=MAX_SOURCE_BYTES)
    for key, maximum in (
            ("max_events", MAX_EVENTS), ("max_users", MAX_USERS),
            ("max_items_per_target", MAX_ITEMS_PER_TARGET),
            ("max_preparation_candidate_rows",
             MAX_PREPARATION_CANDIDATE_ROWS),
            ("max_preparation_candidate_items",
             MAX_PREPARATION_CANDIDATE_ITEMS),
            ("max_artifact_bytes", MAX_ARTIFACT_BYTES),
            ("max_total_artifact_bytes", MAX_TOTAL_ARTIFACT_BYTES)):
        _integer(defaults[key], "matrix " + key, positive=True,
                 maximum=maximum)
    for key in ("max_candidate_scores", "max_itemknn_pairs", "max_results",
                "max_contribution_rows", "max_ranking_items"):
        _integer(defaults[key], "matrix " + key, positive=True)
    _integer(defaults["max_report_bytes"], "matrix report byte limit",
             positive=True, maximum=MAX_REPORT_BYTES)
    _integer(defaults["max_contrasts"], "matrix contrast limit",
             positive=True, maximum=MAX_CONTRASTS)
    _integer(defaults["max_paired_rows"], "matrix paired-row limit",
             positive=True, maximum=MAX_PAIRED_ROWS)
    _integer(defaults["max_analysis_bytes"], "matrix analysis byte limit",
             positive=True, maximum=MAX_ANALYSIS_BYTES)
    return defaults


def _source_config(adapter, value):
    if not isinstance(value, dict):
        raise ComparisonError("matrix source config must be an object")
    if adapter == "lfm1b":
        if set(value) != {"has_header"} or not isinstance(value["has_header"], bool):
            raise ComparisonError("LFM matrix source config is invalid")
    else:
        required = {"dump_id", "dump_type", "archive_sha256", "member_path",
                    "track_identity_policy", "album_entity"}
        if set(value) != required:
            raise ComparisonError("ListenBrainz matrix source config is invalid")
        if (not isinstance(value["dump_id"], str) or not value["dump_id"] or
                any(ord(character) < 32 for character in value["dump_id"])):
            raise ComparisonError("ListenBrainz dump ID is invalid")
        if value["dump_type"] not in ("full", "incremental"):
            raise ComparisonError("ListenBrainz dump type is invalid")
        _digest(value["archive_sha256"], "ListenBrainz archive hash")
        member = value["member_path"]
        if (not isinstance(member, str) or not member or "\\" in member or
                member.startswith("/") or any(
                    part in ("", ".", "..") for part in member.split("/"))):
            raise ComparisonError("ListenBrainz member path is invalid")
        if value["track_identity_policy"] not in (
                "server_mapped_musicbrainz", "recording_msid"):
            raise ComparisonError("ListenBrainz track identity policy is invalid")
        if value["album_entity"] not in ("release_group", "release"):
            raise ComparisonError("ListenBrainz album entity is invalid")
    return value


def _split_config(value):
    if (not isinstance(value, dict) or
            set(value) != {"name", "strategy", "validation_cutoff",
                           "test_cutoff"}):
        raise ComparisonError("matrix split config has an invalid shape")
    _safe_label(value["name"], "matrix split name", 32)
    strategy = value["strategy"]
    if strategy not in SPLIT_STRATEGIES:
        raise ComparisonError("matrix split strategy is invalid")
    validation = value["validation_cutoff"]
    test = value["test_cutoff"]
    if strategy == "global_time_cutoffs":
        _integer(validation, "matrix validation cutoff")
        _integer(test, "matrix test cutoff")
        if validation >= test:
            raise ComparisonError("matrix global cutoffs must be increasing")
    elif validation is not None or test is not None:
        raise ComparisonError("per-user matrix split cannot declare cutoffs")
    return value


def _evaluation(value):
    required = {"targets", "splits", "candidate_policies", "models",
                "k_values", "random_seeds", "neighbor_limit"}
    if not isinstance(value, dict) or set(value) != required:
        raise ComparisonError("matrix evaluation has an invalid shape")
    _ordered(value["targets"], TARGETS, "matrix targets")
    _ordered(value["splits"], SPLITS, "matrix splits")
    _ordered(value["candidate_policies"], CANDIDATE_POLICIES,
             "matrix candidate policies")
    _ordered(value["models"], MODELS, "matrix models")
    _sorted_integers(value["k_values"], "matrix K values", positive=True)
    if "random" in value["models"]:
        _sorted_integers(value["random_seeds"], "matrix random seeds")
    elif value["random_seeds"] != []:
        raise ComparisonError("matrix random seeds require the random model")
    if "itemknn" in value["models"]:
        _integer(value["neighbor_limit"], "matrix neighbor limit", positive=True)
    elif value["neighbor_limit"] is not None:
        raise ComparisonError("matrix neighbor limit requires ItemKNN")
    return value


def _spec_identity(spec):
    return {key: value for key, value in spec.items()
            if key not in ("source_path", "spec_hash")}


def _variant_count(spec):
    axes = spec["axes"]
    return (len(axes["repeat_policies"]) *
            len(axes["positive_filter_horizons"]) *
            len(axes["catalog_policies"]))


def _model_run_count(evaluation):
    return sum(len(evaluation["random_seeds"]) if model == "random" else 1
               for model in evaluation["models"])


def _planned_results(spec):
    evaluation = spec["evaluation"]
    return (_variant_count(spec) * len(evaluation["targets"]) *
            len(evaluation["splits"]) *
            len(evaluation["candidate_policies"]) *
            _model_run_count(evaluation) * len(evaluation["k_values"]))


def _planned_contrasts(spec):
    axes = spec["axes"]
    evaluation = spec["evaluation"]
    if "full_catalog" not in evaluation["candidate_policies"]:
        return 0
    variants = _variant_count(spec)
    edges = 0
    if set(axes["repeat_policies"]) == set(REPEAT_POLICIES):
        edges += variants // 2 * len(evaluation["splits"])
    if set(axes["catalog_policies"]) == set(CATALOG_POLICIES):
        edges += variants // 2 * len(evaluation["splits"])
    if (set(axes["positive_filter_horizons"]) ==
            set(POSITIVE_FILTER_HORIZONS) and
            "validation" in evaluation["splits"]):
        edges += variants // 2
    return (edges * len(evaluation["targets"]) *
            _model_run_count(evaluation) * len(evaluation["k_values"]))


def validate_mask_matrix_spec(spec, expected_spec_hash=None):
    required = {"schema_version", "name", "dataset", "adapter", "source_path",
                "source_config", "split", "axes", "preparation", "evaluation",
                "limits", "spec_hash"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ComparisonError("mask-matrix spec has an invalid shape")
    if (isinstance(spec["schema_version"], bool) or
            not isinstance(spec["schema_version"], int) or
            spec["schema_version"] != MATRIX_SPEC_SCHEMA_VERSION):
        raise ComparisonError("unsupported mask-matrix spec schema")
    _safe_label(spec["name"], "matrix name", 32)
    _safe_label(spec["dataset"], "matrix dataset", 128)
    if spec["adapter"] not in ADAPTERS:
        raise ComparisonError("matrix adapter is unsupported")
    _local_path(spec["source_path"], "matrix source path")
    _source_config(spec["adapter"], spec["source_config"])
    _split_config(spec["split"])
    axes = spec["axes"]
    if (not isinstance(axes, dict) or
            set(axes) != {"repeat_policies", "positive_filter_horizons",
                          "catalog_policies"}):
        raise ComparisonError("matrix axes have an invalid shape")
    _ordered(axes["repeat_policies"], REPEAT_POLICIES,
             "matrix repeat policies")
    _ordered(axes["positive_filter_horizons"], POSITIVE_FILTER_HORIZONS,
             "matrix positive-filter horizons")
    _ordered(axes["catalog_policies"], CATALOG_POLICIES,
             "matrix catalog policies")
    preparation = spec["preparation"]
    if (not isinstance(preparation, dict) or
            set(preparation) != {"sampled_negatives", "seed"}):
        raise ComparisonError("matrix preparation has an invalid shape")
    _integer(preparation["sampled_negatives"], "matrix sampled negatives",
             nonnegative=True, maximum=MAX_SAMPLED_NEGATIVES)
    _integer(preparation["seed"], "matrix preparation seed")
    _evaluation(spec["evaluation"])
    limits = _matrix_limits(spec["limits"])
    if limits != spec["limits"]:
        raise ComparisonError("matrix limits must contain the complete canonical set")
    if (preparation["sampled_negatives"] >
            limits["max_preparation_candidate_items"]):
        raise ComparisonError(
            "matrix sampled negatives exceed the candidate-item preparation limit")
    variants = _variant_count(spec)
    if variants < 2:
        raise ComparisonError("mask matrix must contain at least two variants")
    if variants > limits["max_variants"]:
        raise ComparisonError("mask matrix exceeds its variant limit")
    if _planned_results(spec) > limits["max_results"]:
        raise ComparisonError("mask matrix exceeds its planned-result limit")
    planned_contrasts = _planned_contrasts(spec)
    if planned_contrasts == 0:
        raise ComparisonError(
            "mask matrix has no eligible one-axis full-catalog contrasts")
    if planned_contrasts > limits["max_contrasts"]:
        raise ComparisonError("mask matrix exceeds its planned-contrast limit")
    _digest(spec["spec_hash"], "matrix spec hash")
    if spec["spec_hash"] != sha256(_spec_identity(spec)):
        raise ComparisonError("mask-matrix spec hash mismatch")
    if expected_spec_hash is not None and spec["spec_hash"] != expected_spec_hash:
        raise ComparisonError("expected mask-matrix spec hash does not match")
    if len(canonical_json(spec)) > MAX_MATRIX_BYTES:
        raise ComparisonError("mask-matrix spec exceeds its byte limit")
    return spec


def create_mask_matrix_spec(
        name, dataset, adapter, source_path, source_config, split,
        repeat_policies=REPEAT_POLICIES,
        positive_filter_horizons=POSITIVE_FILTER_HORIZONS,
        catalog_policies=CATALOG_POLICIES, targets=("track",),
        splits=("validation", "test"), candidate_policies=("full_catalog",),
        models=MODELS, k_values=(10,), random_seeds=(0,), neighbor_limit=50,
        sampled_negatives=1000, seed=0, limits=None):
    """Create a canonical, path-independent bounded mask-matrix specification."""
    selected_models = _canonical_values(models, MODELS, "matrix models")
    identity = {
        "schema_version": MATRIX_SPEC_SCHEMA_VERSION,
        "name": name,
        "dataset": dataset,
        "adapter": adapter,
        "source_path": str(source_path),
        "source_config": to_primitive(source_config),
        "split": to_primitive(split),
        "axes": {
            "repeat_policies": _canonical_values(
                repeat_policies, REPEAT_POLICIES, "matrix repeat policies"),
            "positive_filter_horizons": _canonical_values(
                positive_filter_horizons, POSITIVE_FILTER_HORIZONS,
                "matrix positive-filter horizons"),
            "catalog_policies": _canonical_values(
                catalog_policies, CATALOG_POLICIES, "matrix catalog policies"),
        },
        "preparation": {"sampled_negatives": sampled_negatives, "seed": seed},
        "evaluation": {
            "targets": _canonical_values(targets, TARGETS, "matrix targets"),
            "splits": _canonical_values(splits, SPLITS, "matrix splits"),
            "candidate_policies": _canonical_values(
                candidate_policies, CANDIDATE_POLICIES,
                "matrix candidate policies"),
            "models": selected_models,
            "k_values": sorted(set(k_values)),
            "random_seeds": (sorted(set(random_seeds))
                             if "random" in selected_models else []),
            "neighbor_limit": neighbor_limit if "itemknn" in selected_models else None,
        },
        "limits": _matrix_limits(limits),
    }
    spec = dict(identity, spec_hash=sha256({
        key: value for key, value in identity.items() if key != "source_path"}))
    return validate_mask_matrix_spec(spec)


def _case_name(spec, repeat_policy, horizon, catalog_policy):
    value = "%s.%s.%s.%s.%s" % (
        spec["name"], spec["split"]["name"], _ALIASES[repeat_policy],
        _ALIASES[horizon], _ALIASES[catalog_policy])
    return _safe_label(value, "generated matrix case name", 128)


def _expected_mask(spec, repeat_policy, horizon, catalog_policy):
    split = spec["split"]
    strategy = split["strategy"]
    return {
        "split_strategy": strategy,
        "validation_cutoff": split["validation_cutoff"],
        "test_cutoff": split["test_cutoff"],
        "repeat_policy": repeat_policy,
        "positive_filter_horizon": horizon,
        "catalog_policy": catalog_policy,
        "temporal_scope": ("global" if strategy == "global_time_cutoffs"
                           else "user_relative"),
        "catalog_uses_future_information": catalog_policy == "all_mapped",
        "candidate_filter_uses_future_information": horizon == "all_observed",
        "globally_time_causal": (
            strategy == "global_time_cutoffs" and
            catalog_policy == "train_observed" and horizon == "as_of_split"),
    }


def matrix_variants(spec, expected_spec_hash=None):
    spec = validate_mask_matrix_spec(spec, expected_spec_hash)
    variants = []
    for repeat_policy in spec["axes"]["repeat_policies"]:
        for horizon in spec["axes"]["positive_filter_horizons"]:
            for catalog_policy in spec["axes"]["catalog_policies"]:
                variants.append({
                    "case_name": _case_name(
                        spec, repeat_policy, horizon, catalog_policy),
                    "mask_configuration": _expected_mask(
                        spec, repeat_policy, horizon, catalog_policy),
                })
    variants.sort(key=lambda value: value["case_name"])
    return variants


def matrix_preview(spec, expected_spec_hash=None):
    spec = validate_mask_matrix_spec(spec, expected_spec_hash)
    variants = matrix_variants(spec)
    return {
        "spec_hash": spec["spec_hash"],
        "variants": len(variants),
        "planned_results": _planned_results(spec),
        "planned_contrasts": _planned_contrasts(spec),
        "case_names": [value["case_name"] for value in variants],
    }


def _prepare_variant(spec, mask):
    source_path = spec["source_path"]
    split = spec["split"]
    preparation = spec["preparation"]
    common = {
        "split_strategy": split["strategy"],
        "validation_cutoff": split["validation_cutoff"],
        "test_cutoff": split["test_cutoff"],
        "repeat_policy": mask["repeat_policy"],
        "positive_filter_horizon": mask["positive_filter_horizon"],
        "catalog_policy": mask["catalog_policy"],
        "sampled_negatives": preparation["sampled_negatives"],
        "seed": preparation["seed"],
    }
    limits = spec["limits"]
    if spec["adapter"] == "lfm1b":
        protocol = prepare_protocol_artifact(
            read_listening_events_with_provenance(
                source_path, spec["source_config"]["has_header"],
                limits["max_source_bytes"]),
            sampled_negatives=common["sampled_negatives"], seed=common["seed"],
            catalog_policy=common["catalog_policy"],
            split_strategy=common["split_strategy"],
            validation_cutoff=common["validation_cutoff"],
            test_cutoff=common["test_cutoff"],
            repeat_policy=common["repeat_policy"],
            positive_filter_horizon=common["positive_filter_horizon"],
            max_events=limits["max_events"], max_users=limits["max_users"],
            max_items_per_target=limits["max_items_per_target"],
            max_candidate_rows=limits["max_preparation_candidate_rows"],
            max_candidate_items=limits["max_preparation_candidate_items"])
        return "protocol", validate_protocol_artifact(protocol)
    source = spec["source_config"]
    wrapper = prepare_listenbrainz(
        source_path, dump_id=source["dump_id"], dump_type=source["dump_type"],
        archive_sha256=source["archive_sha256"], member_path=source["member_path"],
        track_identity_policy=source["track_identity_policy"],
        album_entity=source["album_entity"], include_local_path=False,
        split_strategy=common["split_strategy"],
        validation_cutoff=common["validation_cutoff"],
        test_cutoff=common["test_cutoff"],
        repeat_policy=common["repeat_policy"],
        positive_filter_horizon=common["positive_filter_horizon"],
        catalog_policy=common["catalog_policy"],
        sampled_negatives=common["sampled_negatives"], seed=common["seed"],
        max_source_bytes=limits["max_source_bytes"],
        max_events=limits["max_events"], max_users=limits["max_users"],
        max_items_per_target=limits["max_items_per_target"],
        max_candidate_rows=limits["max_preparation_candidate_rows"],
        max_candidate_items=limits["max_preparation_candidate_items"])
    return "listenbrainz", validate_listenbrainz(wrapper)


def _write_exclusive_file(directory_fd, name, content):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = None
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ComparisonError("matrix bundle output is not a private regular file")
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise ComparisonError("matrix bundle output already exists")
    except OSError as error:
        raise ComparisonError("cannot write matrix bundle output: %s" % error)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_generated_bundle(root_fd, case_name, logical_name, artifact,
                            max_artifact_bytes, remaining_total_bytes):
    content = canonical_json(artifact)
    filename = logical_name + ".json"
    files = {filename: hashlib.sha256(content).hexdigest()}
    manifest_identity = {"schema_version": BUNDLE_SCHEMA_VERSION, "files": files}
    manifest = dict(manifest_identity,
                    identity_sha256=sha256(manifest_identity))
    manifest_content = canonical_json(manifest)
    total_bytes = len(content) + len(manifest_content)
    if total_bytes > max_artifact_bytes:
        raise ComparisonError("prepared matrix artifact exceeds its byte limit")
    if total_bytes > remaining_total_bytes:
        raise ComparisonError("prepared matrix artifacts exceed their aggregate byte limit")
    try:
        os.mkdir(case_name, 0o700, dir_fd=root_fd)
    except FileExistsError:
        raise ComparisonError("matrix case output already exists")
    except OSError as error:
        raise ComparisonError("cannot create matrix case output: %s" % error)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    case_fd = None
    try:
        case_fd = os.open(case_name, flags, dir_fd=root_fd)
        opened = os.fstat(case_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise ComparisonError("matrix case output is not a directory")
        _write_exclusive_file(case_fd, filename, content)
        _write_exclusive_file(case_fd, "manifest.json", manifest_content)
        os.fsync(case_fd)
        os.fsync(root_fd)
    finally:
        if case_fd is not None:
            os.close(case_fd)
    return total_bytes


def _empty_generated_root(path):
    resolved, descriptor = open_generated_directory(path, create=True)
    if os.listdir(descriptor):
        os.close(descriptor)
        raise ComparisonError(
            "matrix artifact root must be empty to avoid stale or partial outputs")
    return resolved, descriptor


def _entry_identity(entry):
    return {key: value for key, value in entry.items() if key != "artifact_path"}


def _index_identity(index):
    return {
        "schema_version": index["schema_version"],
        "spec_hash": index["spec_hash"],
        "entries": [_entry_identity(value) for value in index["entries"]],
    }


def _validate_artifact_identity(identity, adapter):
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
        raise ComparisonError("matrix artifact identity has an invalid shape")
    if identity["adapter"] != adapter:
        raise ComparisonError("matrix artifact identity adapter mismatch")
    for key in ("protocol_hash", "protocol_config_hash", "population_hash"):
        _digest(identity[key], "matrix artifact " + key)
    if adapter == "lfm1b":
        if identity["artifact_hash"] is not None or identity["wrapper_config_hash"] is not None:
            raise ComparisonError("LFM matrix identity cannot contain wrapper hashes")
    else:
        _digest(identity["artifact_hash"], "matrix wrapper artifact hash")
        _digest(identity["wrapper_config_hash"], "matrix wrapper config hash")


def validate_matrix_index(index, spec, expected_index_hash=None,
                          expected_spec_hash=None):
    spec = validate_mask_matrix_spec(spec, expected_spec_hash)
    required = {"schema_version", "spec_hash", "entries", "provenance",
                "index_hash"}
    if not isinstance(index, dict) or set(index) != required:
        raise ComparisonError("mask-matrix index has an invalid shape")
    if (isinstance(index["schema_version"], bool) or
            not isinstance(index["schema_version"], int) or
            index["schema_version"] != MATRIX_INDEX_SCHEMA_VERSION):
        raise ComparisonError("unsupported mask-matrix index schema")
    if index["spec_hash"] != spec["spec_hash"]:
        raise ComparisonError("mask-matrix index references a different spec")
    expected_variants = {value["case_name"]: value
                         for value in matrix_variants(spec)}
    entries = index["entries"]
    if not isinstance(entries, list) or len(entries) != len(expected_variants):
        raise ComparisonError("mask-matrix index has an invalid entry count")
    names = []
    populations = set()
    for entry in entries:
        if (not isinstance(entry, dict) or
                set(entry) != {"case_name", "dataset", "adapter",
                               "artifact_path", "mask_configuration",
                               "artifact_identity"}):
            raise ComparisonError("matrix index entry has an invalid shape")
        name = _safe_label(entry["case_name"], "matrix case name", 128)
        names.append(name)
        if name not in expected_variants:
            raise ComparisonError("matrix index contains an unknown case")
        if entry["dataset"] != spec["dataset"] or entry["adapter"] != spec["adapter"]:
            raise ComparisonError("matrix index dataset or adapter mismatch")
        _local_path(entry["artifact_path"], "matrix artifact path")
        if entry["mask_configuration"] != expected_variants[name]["mask_configuration"]:
            raise ComparisonError("matrix index mask configuration mismatch")
        _validate_artifact_identity(entry["artifact_identity"], spec["adapter"])
        populations.add(entry["artifact_identity"]["population_hash"])
    if names != sorted(set(names)) or set(names) != set(expected_variants):
        raise ComparisonError("matrix index cases must be complete, unique, and sorted")
    if len(populations) != 1:
        raise ComparisonError("matrix variants do not share one normalized population")
    provenance = index["provenance"]
    if (not isinstance(provenance, dict) or
            set(provenance) != {"source_path", "artifact_root"}):
        raise ComparisonError("matrix index provenance has an invalid shape")
    _local_path(provenance["source_path"], "matrix source provenance")
    _local_path(provenance["artifact_root"], "matrix artifact-root provenance")
    _digest(index["index_hash"], "matrix index hash")
    if index["index_hash"] != sha256(_index_identity(index)):
        raise ComparisonError("mask-matrix index hash mismatch")
    if expected_index_hash is not None and index["index_hash"] != expected_index_hash:
        raise ComparisonError("expected mask-matrix index hash does not match")
    if len(canonical_json(index)) > MAX_MATRIX_BYTES:
        raise ComparisonError("mask-matrix index exceeds its byte limit")
    return index


def prepare_mask_matrix(spec, artifact_root, expected_spec_hash=None):
    """Prepare each local mask variant and return a canonical artifact index."""
    spec = validate_mask_matrix_spec(spec, expected_spec_hash)
    source_path = Path(spec["source_path"]).expanduser().resolve()
    try:
        source_stat = os.stat(str(source_path))
    except OSError as error:
        raise ComparisonError("cannot inspect matrix source: %s" % error)
    if not stat.S_ISREG(source_stat.st_mode):
        raise ComparisonError("matrix source must be an existing local regular file")
    if source_stat.st_size > spec["limits"]["max_source_bytes"]:
        raise ComparisonError("matrix source exceeds its byte limit")
    source_identity = (source_stat.st_dev, source_stat.st_ino,
                       source_stat.st_size, source_stat.st_mtime_ns)
    root, root_fd = _empty_generated_root(artifact_root)
    preparation_spec = dict(spec, source_path=str(source_path))
    entries = []
    total_artifact_bytes = 0
    expected_names = set()
    try:
        for variant in matrix_variants(spec):
            expected_names.add(variant["case_name"])
            logical_name, artifact = _prepare_variant(
                preparation_spec, variant["mask_configuration"])
            try:
                current_stat = os.stat(str(source_path))
            except OSError as error:
                raise ComparisonError(
                    "cannot recheck matrix source after preparation: %s" % error)
            if ((current_stat.st_dev, current_stat.st_ino, current_stat.st_size,
                 current_stat.st_mtime_ns) != source_identity):
                raise ComparisonError("matrix source changed during preparation")
            artifact_bytes = _write_generated_bundle(
                root_fd, variant["case_name"], logical_name, artifact,
                spec["limits"]["max_artifact_bytes"],
                spec["limits"]["max_total_artifact_bytes"] -
                total_artifact_bytes)
            total_artifact_bytes += artifact_bytes
            directory = root / variant["case_name"]
            loaded = inspect_artifact(spec["adapter"], directory)
            actual_mask = mask_configuration(loaded.protocol)
            if actual_mask != variant["mask_configuration"]:
                raise ComparisonError("prepared matrix artifact has an unexpected mask")
            entries.append({
                "case_name": variant["case_name"],
                "dataset": spec["dataset"],
                "adapter": spec["adapter"],
                "artifact_path": loaded.artifact_path,
                "mask_configuration": actual_mask,
                "artifact_identity": dict(loaded.identity),
            })
        # ``os.listdir(fd)`` is stateful on some filesystems.  This descriptor
        # was already enumerated while proving the generated root was empty;
        # rewind it before using a second enumeration as an integrity check.
        os.lseek(root_fd, 0, os.SEEK_SET)
        if set(os.listdir(root_fd)) != expected_names:
            raise ComparisonError("matrix artifact root changed during preparation")
    finally:
        os.close(root_fd)
    identity = {
        "schema_version": MATRIX_INDEX_SCHEMA_VERSION,
        "spec_hash": spec["spec_hash"],
        "entries": entries,
    }
    index = dict(identity,
                 provenance={"source_path": str(source_path),
                             "artifact_root": str(root)},
                 index_hash=sha256({
                     "schema_version": identity["schema_version"],
                     "spec_hash": identity["spec_hash"],
                     "entries": [_entry_identity(value) for value in entries],
                 }))
    return validate_matrix_index(index, spec)


def create_matrix_comparison_plan(spec, index, expected_spec_hash=None,
                                  expected_index_hash=None):
    spec = validate_mask_matrix_spec(spec, expected_spec_hash)
    index = validate_matrix_index(
        index, spec, expected_index_hash, expected_spec_hash)
    evaluation = spec["evaluation"]
    cases = []
    for entry in index["entries"]:
        loaded = inspect_artifact(entry["adapter"], entry["artifact_path"])
        if (dict(loaded.identity) != entry["artifact_identity"] or
                mask_configuration(loaded.protocol) != entry["mask_configuration"]):
            raise ComparisonError("matrix artifact no longer matches its pinned index")
        identity = entry["artifact_identity"]
        cases.append({
            "name": entry["case_name"],
            "dataset": entry["dataset"],
            "adapter": entry["adapter"],
            "artifact_path": loaded.artifact_path,
            "expected_hashes": {
                "artifact_hash": identity["artifact_hash"],
                "wrapper_config_hash": identity["wrapper_config_hash"],
                "protocol_hash": identity["protocol_hash"],
                "protocol_config_hash": identity["protocol_config_hash"],
            },
            "targets": evaluation["targets"],
            "splits": evaluation["splits"],
            "candidate_policies": evaluation["candidate_policies"],
            "models": evaluation["models"],
            "k_values": evaluation["k_values"],
            "random_seeds": evaluation["random_seeds"],
            "neighbor_limit": evaluation["neighbor_limit"],
        })
    limits = spec["limits"]
    return create_plan(
        cases, limits["max_candidate_scores"], limits["max_itemknn_pairs"],
        limits["max_results"], limits["max_contribution_rows"],
        limits["max_ranking_items"], limits["max_report_bytes"])


def _mask_axes(mask):
    return {key: mask[key] for key in (
        "repeat_policy", "positive_filter_horizon", "catalog_policy")}


def _one_axis_edges(index):
    by_axes = {tuple(sorted(_mask_axes(entry["mask_configuration"]).items())): entry
               for entry in index["entries"]}
    edges = []
    for axis, baseline_value, comparison_value in _AXIS_ORIENTATIONS:
        for entry in index["entries"]:
            baseline_axes = _mask_axes(entry["mask_configuration"])
            if baseline_axes[axis] != baseline_value:
                continue
            comparison_axes = dict(baseline_axes, **{axis: comparison_value})
            comparison = by_axes.get(tuple(sorted(comparison_axes.items())))
            if comparison is not None:
                edges.append((axis, entry, comparison))
    return sorted(edges, key=lambda value: (
        value[0], value[1]["case_name"], value[2]["case_name"]))


def _contrast_plan_identity(plan):
    return {key: value for key, value in plan.items()
            if key != "contrast_plan_hash"}


def _resolve_matrix_contrasts(spec, index, comparison_plan, report):
    result_lookup = {}
    for result in report["results"]:
        key = (result["case_name"], result["target"], result["split"],
               result["candidate_policy"], result["model"],
               result["model_seed"], result["k"])
        if key in result_lookup:
            raise ComparisonError("matrix report contains an ambiguous result selector")
        result_lookup[key] = result
    selectors = []
    pairs = []
    for axis, baseline_entry, comparison_entry in _one_axis_edges(index):
        for baseline in report["results"]:
            if (baseline["case_name"] != baseline_entry["case_name"] or
                    baseline["candidate_policy"] != "full_catalog"):
                continue
            if axis == "positive_filter_horizon" and baseline["split"] == "test":
                continue
            key = (comparison_entry["case_name"], baseline["target"],
                   baseline["split"], baseline["candidate_policy"],
                   baseline["model"], baseline["model_seed"], baseline["k"])
            comparison = result_lookup.get(key)
            if comparison is None:
                raise ComparisonError("matrix contrast selector has no matching result")
            number = len(pairs) + 1
            name = "mask-%04d-%s" % (number, axis.replace("_", "-"))
            selector = {
                "name": name,
                "axis": axis,
                "baseline_case_name": baseline["case_name"],
                "comparison_case_name": comparison["case_name"],
                "target": baseline["target"],
                "split": baseline["split"],
                "candidate_policy": baseline["candidate_policy"],
                "model": baseline["model"],
                "model_seed": baseline["model_seed"],
                "k": baseline["k"],
            }
            selectors.append(selector)
            pairs.append({
                "name": name,
                "kind": "mask",
                "baseline_result_hash": baseline["result_hash"],
                "comparison_result_hash": comparison["result_hash"],
            })
    if not pairs:
        raise ComparisonError(
            "matrix produced no eligible one-axis full-catalog mask contrasts")
    limits = spec["limits"]
    if len(pairs) > limits["max_contrasts"]:
        raise ComparisonError("matrix contrast plan exceeds its contrast limit")
    create_contrast_analysis(
        report, pairs, report["report_hash"], limits["max_contrasts"],
        limits["max_paired_rows"], limits["max_analysis_bytes"])
    identity = {
        "schema_version": MATRIX_CONTRAST_PLAN_SCHEMA_VERSION,
        "spec_hash": spec["spec_hash"],
        "index_hash": index["index_hash"],
        "comparison_plan_hash": comparison_plan["plan_hash"],
        "input_report_hash": report["report_hash"],
        "selectors": selectors,
        "pairs": pairs,
        "limits": {
            "max_contrasts": limits["max_contrasts"],
            "max_paired_rows": limits["max_paired_rows"],
            "max_analysis_bytes": limits["max_analysis_bytes"],
        },
    }
    return dict(identity, contrast_plan_hash=sha256(identity))


def validate_matrix_contrast_plan(
        contrast_plan, spec, index, comparison_plan, report,
        expected_contrast_plan_hash=None, expected_report_hash=None):
    spec = validate_mask_matrix_spec(spec)
    index = validate_matrix_index(index, spec)
    comparison_plan = validate_plan(comparison_plan)
    report = validate_report(report, expected_report_hash)
    if comparison_plan != create_matrix_comparison_plan(spec, index):
        raise ComparisonError("comparison plan was not generated from this matrix")
    if report["plan_hash"] != comparison_plan["plan_hash"]:
        raise ComparisonError("matrix report does not belong to the comparison plan")
    required = {"schema_version", "spec_hash", "index_hash",
                "comparison_plan_hash", "input_report_hash", "selectors",
                "pairs", "limits", "contrast_plan_hash"}
    if not isinstance(contrast_plan, dict) or set(contrast_plan) != required:
        raise ComparisonError("matrix contrast plan has an invalid shape")
    if (isinstance(contrast_plan["schema_version"], bool) or
            not isinstance(contrast_plan["schema_version"], int) or
            contrast_plan["schema_version"] != MATRIX_CONTRAST_PLAN_SCHEMA_VERSION):
        raise ComparisonError("unsupported matrix contrast-plan schema")
    _digest(contrast_plan["contrast_plan_hash"], "matrix contrast-plan hash")
    if contrast_plan["contrast_plan_hash"] != sha256(
            _contrast_plan_identity(contrast_plan)):
        raise ComparisonError("matrix contrast-plan hash mismatch")
    if (expected_contrast_plan_hash is not None and
            contrast_plan["contrast_plan_hash"] != expected_contrast_plan_hash):
        raise ComparisonError("expected matrix contrast-plan hash does not match")
    expected = _resolve_matrix_contrasts(
        spec, index, comparison_plan, report)
    if contrast_plan != expected:
        raise ComparisonError("matrix contrast plan is semantically inconsistent")
    if len(canonical_json(contrast_plan)) > MAX_CONTRAST_PLAN_BYTES:
        raise ComparisonError("matrix contrast plan exceeds its byte limit")
    return contrast_plan


def resolve_matrix_contrast_plan(
        spec, index, comparison_plan, report, expected_spec_hash=None,
        expected_index_hash=None, expected_plan_hash=None,
        expected_report_hash=None):
    spec = validate_mask_matrix_spec(spec, expected_spec_hash)
    index = validate_matrix_index(
        index, spec, expected_index_hash, expected_spec_hash)
    comparison_plan = validate_plan(comparison_plan, expected_plan_hash)
    expected_matrix_plan = create_matrix_comparison_plan(spec, index)
    if comparison_plan != expected_matrix_plan:
        raise ComparisonError("comparison plan was not generated from this matrix")
    report = validate_report(report, expected_report_hash)
    if report["plan_hash"] != comparison_plan["plan_hash"]:
        raise ComparisonError("matrix report does not belong to the comparison plan")
    contrast_plan = _resolve_matrix_contrasts(
        spec, index, comparison_plan, report)
    return validate_matrix_contrast_plan(
        contrast_plan, spec, index, comparison_plan, report)


def create_contrast_analysis_from_matrix_plan(
        contrast_plan, spec, index, comparison_plan, report,
        expected_contrast_plan_hash=None, expected_report_hash=None):
    contrast_plan = validate_matrix_contrast_plan(
        contrast_plan, spec, index, comparison_plan, report,
        expected_contrast_plan_hash, expected_report_hash)
    report = validate_report(report, expected_report_hash)
    limits = contrast_plan["limits"]
    return create_contrast_analysis(
        report, contrast_plan["pairs"], report["report_hash"],
        limits["max_contrasts"], limits["max_paired_rows"],
        limits["max_analysis_bytes"])


def _save(path, value, validator):
    value = validator(value)
    atomic_write_generated(path, canonical_json(value))
    return value


def save_mask_matrix_spec(path, spec):
    return _save(path, spec, validate_mask_matrix_spec)


def _reject_nested_output(path, root):
    output_lexical = Path(os.path.abspath(str(Path(path).expanduser())))
    root_lexical = Path(os.path.abspath(str(Path(root).expanduser())))
    pairs = ((output_lexical, root_lexical),
             (output_lexical.resolve(), root_lexical.resolve()))
    for output, artifact_root in pairs:
        try:
            output.relative_to(artifact_root)
        except ValueError:
            continue
        raise ComparisonError(
            "matrix index output must not be inside the artifact root")


def validate_matrix_output_paths(index_path, artifact_root):
    """Fail before preparation when the index would alias an artifact bundle."""
    _local_path(str(index_path), "matrix index output path")
    _local_path(str(artifact_root), "matrix artifact root")
    _reject_nested_output(index_path, artifact_root)
    return True


def save_matrix_index(path, index, spec):
    if (isinstance(index, dict) and isinstance(index.get("provenance"), dict) and
            "artifact_root" in index["provenance"]):
        _reject_nested_output(path, index["provenance"]["artifact_root"])
    return _save(path, index, lambda value: validate_matrix_index(value, spec))


def save_matrix_contrast_plan(path, contrast_plan, spec, index,
                              comparison_plan, report):
    return _save(path, contrast_plan, lambda value: validate_matrix_contrast_plan(
        value, spec, index, comparison_plan, report))


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError("duplicate JSON key in matrix artifact: " + key)
        result[key] = value
    return result


def _json_constant(value):
    raise ComparisonError("non-finite JSON value in matrix artifact: " + value)


def _load(path, maximum, label):
    try:
        with open(path, "rb") as stream:
            content = stream.read(maximum + 1)
    except OSError as error:
        raise ComparisonError("cannot read %s: %s" % (label, error))
    if len(content) > maximum:
        raise ComparisonError(label + " exceeds its byte limit")
    try:
        return json.loads(content.decode("utf-8"), object_pairs_hook=_json_pairs,
                          parse_constant=_json_constant)
    except (RecursionError, UnicodeDecodeError, ValueError, TypeError) as error:
        if isinstance(error, ComparisonError):
            raise
        raise ComparisonError("malformed %s: %s" % (label, error))


def load_mask_matrix_spec(path, expected_spec_hash=None):
    return validate_mask_matrix_spec(
        _load(path, MAX_MATRIX_BYTES, "mask-matrix spec"), expected_spec_hash)


def load_matrix_index(path, spec, expected_index_hash=None,
                      expected_spec_hash=None):
    return validate_matrix_index(
        _load(path, MAX_MATRIX_BYTES, "mask-matrix index"), spec,
        expected_index_hash, expected_spec_hash)


def load_matrix_contrast_plan(
        path, spec, index, comparison_plan, report,
        expected_contrast_plan_hash=None, expected_report_hash=None):
    return validate_matrix_contrast_plan(
        _load(path, MAX_CONTRAST_PLAN_BYTES, "matrix contrast plan"),
        spec, index, comparison_plan, report, expected_contrast_plan_hash,
        expected_report_hash)
