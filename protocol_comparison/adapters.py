"""Uniform read-only access to prepared LFM and ListenBrainz artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from lfm1b_protocol.artifacts import load_protocol_artifact
from lfm1b_protocol.canonical import sha256
from listenbrainz_protocol.core import load as load_listenbrainz

from .plan import (CANDIDATE_POLICIES, MODELS, SPLITS, TARGETS, ComparisonError,
                   MAX_SIGNED_INTEGER, validate_case)


@dataclass(frozen=True)
class LoadedArtifact:
    adapter: str
    artifact_path: str
    identity: Mapping[str, Any]
    protocol: Mapping[str, Any]


def _lfm_population_hash(protocol):
    return sha256({
        "adapter": "lfm1b",
        "sources": protocol["sources"],
        "known_positive_hashes": {
            target: sha256(protocol["targets"][target]["known_positives"])
            for target in TARGETS
        },
    })


def _listenbrainz_population_hash(wrapper):
    mapping_config = {
        key: value for key, value in wrapper["config"].items()
        if key != "embedded_protocol_config_hash"
    }
    return sha256({
        "adapter": "listenbrainz",
        "source": wrapper["source"],
        "mapping_config": mapping_config,
        "mapping_hash": sha256(wrapper["mappings"]),
    })


def _mask_configuration(protocol):
    config = protocol["config"]
    keys = ("split_strategy", "validation_cutoff", "test_cutoff", "repeat_policy",
            "positive_filter_horizon", "catalog_policy", "temporal_scope",
            "catalog_uses_future_information", "candidate_filter_uses_future_information",
            "globally_time_causal")
    output = {key: config[key] for key in keys}
    for key in ("validation_cutoff", "test_cutoff"):
        value = output[key]
        if (value is not None and
                (isinstance(value, bool) or not isinstance(value, int) or
                 value < -MAX_SIGNED_INTEGER or value > MAX_SIGNED_INTEGER)):
            raise ComparisonError(
                "comparison cutoff timestamps must fit signed 63-bit integers")
    return output


def inspect_artifact(adapter, artifact_path):
    """Validate a local artifact and return its identities without trusting them."""
    resolved = str(Path(artifact_path).expanduser().resolve())
    if adapter == "lfm1b":
        protocol = load_protocol_artifact(resolved)
        identity = {"adapter": adapter, "artifact_hash": None,
                     "wrapper_config_hash": None,
                     "protocol_hash": protocol["protocol_hash"],
                     "protocol_config_hash": protocol["config_hash"],
                     "population_hash": _lfm_population_hash(protocol)}
    elif adapter == "listenbrainz":
        wrapper = load_listenbrainz(resolved)
        protocol = wrapper["protocol"]
        identity = {"adapter": adapter,
                    "artifact_hash": wrapper["artifact_hash"],
                     "wrapper_config_hash": wrapper["config_hash"],
                     "protocol_hash": protocol["protocol_hash"],
                     "protocol_config_hash": protocol["config_hash"],
                     "population_hash": _listenbrainz_population_hash(wrapper)}
    else:
        raise ComparisonError("unsupported comparison adapter: %s" % adapter)
    return LoadedArtifact(adapter=adapter, artifact_path=resolved,
                          identity=identity, protocol=protocol)


def load_case_artifact(case):
    case = validate_case(case)
    expected = case["expected_hashes"]
    path = case["artifact_path"]
    if case["adapter"] == "lfm1b":
        protocol = load_protocol_artifact(
            path, expected["protocol_hash"], expected["protocol_config_hash"])
        identity = {"adapter": "lfm1b", "artifact_hash": None,
                     "wrapper_config_hash": None,
                     "protocol_hash": protocol["protocol_hash"],
                     "protocol_config_hash": protocol["config_hash"],
                     "population_hash": _lfm_population_hash(protocol)}
    else:
        wrapper = load_listenbrainz(
            path, expected["artifact_hash"], expected["protocol_hash"],
            expected["wrapper_config_hash"])
        protocol = wrapper["protocol"]
        if protocol["config_hash"] != expected["protocol_config_hash"]:
            raise ComparisonError("expected embedded protocol config hash does not match")
        identity = {"adapter": "listenbrainz",
                     "artifact_hash": wrapper["artifact_hash"],
                     "wrapper_config_hash": wrapper["config_hash"],
                     "protocol_hash": protocol["protocol_hash"],
                     "protocol_config_hash": protocol["config_hash"],
                     "population_hash": _listenbrainz_population_hash(wrapper)}
    actual = {key: identity[key] for key in expected}
    if actual != expected:
        raise ComparisonError("loaded artifact identity contradicts the comparison case")
    return LoadedArtifact(adapter=case["adapter"],
                          artifact_path=str(Path(path).expanduser().resolve()),
                          identity=identity, protocol=protocol)


def _ordered(values, allowed, label):
    values = tuple(values)
    if not values or len(values) != len(set(values)) or any(
            value not in allowed for value in values):
        raise ComparisonError(label + " contains invalid or duplicate values")
    return [value for value in allowed if value in values]


def _sorted_integers(values, label, positive=False):
    values = tuple(values)
    if not values:
        raise ComparisonError(label + " must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or
           value < -MAX_SIGNED_INTEGER or value > MAX_SIGNED_INTEGER or
           (positive and value <= 0) for value in values):
        raise ComparisonError(label + " must contain signed 63-bit integers")
    return sorted(set(values))


def case_from_artifact(name, dataset, adapter, artifact_path, targets=("track",),
                       splits=("test",), candidate_policies=("full_catalog",),
                       models=MODELS, k_values=(10,), random_seeds=(0,),
                       neighbor_limit=50):
    """Create a strict case whose expected hashes come from a local artifact."""
    loaded = inspect_artifact(adapter, artifact_path)
    selected_models = _ordered(models, MODELS, "models")
    case = {
        "name": name,
        "dataset": dataset,
        "adapter": adapter,
        "artifact_path": loaded.artifact_path,
        "expected_hashes": {
            "artifact_hash": loaded.identity["artifact_hash"],
            "wrapper_config_hash": loaded.identity["wrapper_config_hash"],
            "protocol_hash": loaded.identity["protocol_hash"],
            "protocol_config_hash": loaded.identity["protocol_config_hash"],
        },
        "targets": _ordered(targets, TARGETS, "targets"),
        "splits": _ordered(splits, SPLITS, "splits"),
        "candidate_policies": _ordered(
            candidate_policies, CANDIDATE_POLICIES, "candidate policies"),
        "models": selected_models,
        "k_values": _sorted_integers(k_values, "K values", positive=True),
        "random_seeds": (_sorted_integers(random_seeds, "random seeds")
                         if "random" in selected_models else []),
        "neighbor_limit": neighbor_limit if "itemknn" in selected_models else None,
    }
    return validate_case(case)


def artifact_summary(adapter, artifact_path):
    loaded = inspect_artifact(adapter, artifact_path)
    return {"artifact_path": loaded.artifact_path,
            "identity": dict(loaded.identity),
            "mask_configuration": _mask_configuration(loaded.protocol)}


def mask_configuration(protocol):
    return _mask_configuration(protocol)
