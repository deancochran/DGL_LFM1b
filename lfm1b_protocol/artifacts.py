import hashlib
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, Mapping, Optional, Tuple

from .canonical import canonical_json, sha256, to_primitive
from .candidates import build_candidates, candidate_digest
from .models import Event, ITEM_TYPES, PairInteraction
from .split import temporal_split


SCHEMA_VERSION = 1


class ArtifactIntegrityError(ValueError):
    pass


def _split_row(row: PairInteraction) -> Dict[str, int]:
    return {
        "user_id": row.user_id,
        "item_id": row.item_id,
        "play_count": row.play_count,
        "first_timestamp": row.first_timestamp,
        "last_timestamp": row.last_timestamp,
    }


def _candidate_row(candidate: Any) -> Dict[str, Any]:
    return {
        "user_id": candidate.user_id,
        "candidate_item_ids": candidate.items,
        "positive_item_ids": candidate.positives,
        "candidate_hash": candidate.candidate_hash,
    }


def _protocol_identity(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in artifact.items()
            if key not in ("protocol_hash", "provenance", "created_at", "created_utc")}


def prepare_protocol_artifact(
        events: Iterable[Event], sampled_negatives: int = 1000, seed: int = 0,
        catalog_policy: str = "train_observed",
        sources: Optional[Iterable[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """Build the self-describing artifact consumed by ``train_temporal.py``.

    ``train_observed`` is a warm-start benchmark. ``all_mapped`` explicitly
    exposes IDs from every horizon and is therefore transductive.
    """
    if sampled_negatives < 0:
        raise ValueError("sampled_negatives must be non-negative")
    if catalog_policy not in ("train_observed", "all_mapped"):
        raise ValueError("catalog_policy must be train_observed or all_mapped")
    protocol = temporal_split(events)
    config = {
        "candidate_policies": {
            "fixed_sampled": {
                "materialized": True,
                "negatives": sampled_negatives,
                "seed": seed,
                "without_replacement": True,
            },
            "full_catalog": {"materialized": False},
        },
        "catalog_policy": catalog_policy,
        "catalog_uses_future_information": catalog_policy == "all_mapped",
        "minimum_timestamp_groups": 3,
        "sampled_negatives": sampled_negatives,
        "seed": seed,
        "split_strategy": "global_per_user_timestamp_groups",
    }
    targets = {}
    target_statistics = {}
    training_users = set(row.user_id for row in protocol.split.train)
    for item_type in ITEM_TYPES:
        raw_splits = {}
        for split_name in ("train", "validation", "test"):
            interactions = getattr(protocol.split, split_name)
            raw_splits[split_name] = tuple(
                _split_row(row) for row in interactions if row.item_type == item_type)
        known = protocol.known_positives[item_type]
        known_index = defaultdict(set)  # type: DefaultDict[int, set]
        for user_id, item_id in known:
            known_index[user_id].add(item_id)
        if catalog_policy == "train_observed":
            catalog = tuple(sorted(set(row["item_id"] for row in raw_splits["train"])))
        else:
            catalog = protocol.catalogs[item_type]
        catalog_set = set(catalog)
        target_splits = {"train": raw_splits["train"]}
        exclusions = {}
        for split_name in ("validation", "test"):
            included = tuple(row for row in raw_splits[split_name]
                             if row["item_id"] in catalog_set and
                             (catalog_policy == "all_mapped" or
                              row["user_id"] in training_users))
            excluded = tuple(row for row in raw_splits[split_name]
                             if row["item_id"] not in catalog_set or
                             (catalog_policy == "train_observed" and
                              row["user_id"] not in training_users))
            target_splits[split_name] = included
            exclusions[split_name] = {
                "excluded_positive_pairs": len(excluded),
                "excluded_user_ids": tuple(sorted(set(
                    row["user_id"] for row in excluded))),
                "excluded_users": len(set(row["user_id"] for row in excluded)),
            }
        target_statistics[item_type] = {"cold_start_exclusions": exclusions}
        candidates = {}
        for split_name in ("validation", "test"):
            by_user = defaultdict(set)  # type: DefaultDict[int, set]
            for row in target_splits[split_name]:
                by_user[row["user_id"]].add(row["item_id"])
            candidates[split_name] = tuple(
                _candidate_row(build_candidates(
                    user_id, item_type, split_name, by_user[user_id], catalog,
                    known, "fixed_sampled", sampled_negatives, seed,
                    known_index, catalog, catalog_set))
                for user_id in sorted(by_user)
            )
        targets[item_type] = {
            "catalog": catalog,
            "known_positives": tuple(
                {"user_id": user_id, "item_id": item_id}
                for user_id, item_id in known),
            "splits": target_splits,
            # Direct split keys are the frozen sampled policy expected by training.
            "candidates": candidates,
        }
    statistics = to_primitive(protocol.statistics)
    statistics["targets"] = target_statistics
    normalized_sources = []
    source_paths = []
    for source in (sources or ()):
        normalized = to_primitive(source)
        path = normalized.pop("path", None)
        normalized_sources.append(normalized)
        if path is not None:
            source_paths.append(path)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "config_hash": sha256(config),
        "sources": tuple(normalized_sources),
        "provenance": {"source_paths": tuple(source_paths)},
        "statistics": statistics,
        "targets": targets,
    }
    artifact["protocol_hash"] = sha256(_protocol_identity(artifact))
    return artifact


def candidate_rows_for_policy(artifact: Mapping[str, Any], target: str,
                              split: str, policy: str) -> Tuple[Dict[str, Any], ...]:
    """Return frozen sampled rows or derive deterministic full-catalog rows."""
    if target not in ITEM_TYPES:
        raise ValueError("target must be artist, album, or track")
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    payload = artifact["targets"][target]
    if policy == "fixed_sampled":
        rows = payload["candidates"][split]
        return tuple({
            "user_id": row["user_id"],
            "candidate_item_ids": tuple(row["candidate_item_ids"]),
            "positive_item_ids": tuple(row["positive_item_ids"]),
            "candidate_hash": row["candidate_hash"],
        } for row in rows)
    if policy != "full_catalog":
        raise ValueError("policy must be full_catalog or fixed_sampled")
    known = tuple((row["user_id"], row["item_id"])
                  for row in payload["known_positives"])
    known_index = defaultdict(set)  # type: DefaultDict[int, set]
    for user_id, item_id in known:
        known_index[user_id].add(item_id)
    positives = defaultdict(set)  # type: DefaultDict[int, set]
    for row in payload["splits"][split]:
        positives[row["user_id"]].add(row["item_id"])
    catalog_items = tuple(payload["catalog"])
    catalog_index = set(catalog_items)
    return tuple(_candidate_row(build_candidates(
        user_id, target, split, positives[user_id], catalog_items, known,
        "full_catalog", known_positive_index=known_index,
        sorted_catalog=catalog_items, catalog_index=catalog_index))
        for user_id in sorted(positives))


def validate_protocol_artifact(
        artifact: Mapping[str, Any], expected_protocol_hash: Optional[str] = None,
        expected_config_hash: Optional[str] = None) -> Dict[str, Any]:
    required = ("schema_version", "config", "config_hash", "protocol_hash",
                "sources", "targets")
    missing = [name for name in required if name not in artifact]
    if missing:
        raise ArtifactIntegrityError("protocol artifact is missing: %s" %
                                     ", ".join(missing))
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ArtifactIntegrityError("unsupported protocol schema version")
    if artifact["config_hash"] != sha256(artifact["config"]):
        raise ArtifactIntegrityError("protocol config hash mismatch")
    if artifact["protocol_hash"] != sha256(_protocol_identity(artifact)):
        raise ArtifactIntegrityError("protocol hash mismatch")
    if (expected_protocol_hash is not None and
            artifact["protocol_hash"] != expected_protocol_hash):
        raise ArtifactIntegrityError("expected protocol hash does not match")
    if (expected_config_hash is not None and
            artifact["config_hash"] != expected_config_hash):
        raise ArtifactIntegrityError("expected config hash does not match")
    config = artifact["config"]
    if not isinstance(config, dict):
        raise ArtifactIntegrityError("protocol config must be a mapping")
    catalog_policy = config.get("catalog_policy")
    if catalog_policy not in ("train_observed", "all_mapped"):
        raise ArtifactIntegrityError("invalid catalog policy")
    if config.get("split_strategy") != "global_per_user_timestamp_groups":
        raise ArtifactIntegrityError("unsupported split strategy")
    if config.get("minimum_timestamp_groups") != 3:
        raise ArtifactIntegrityError("minimum timestamp groups must be 3")
    if config.get("catalog_uses_future_information") != (catalog_policy == "all_mapped"):
        raise ArtifactIntegrityError("catalog future-information flag is inconsistent")
    policies = config.get("candidate_policies")
    if not isinstance(policies, dict):
        raise ArtifactIntegrityError("candidate policies are missing")
    fixed = policies.get("fixed_sampled", {})
    full = policies.get("full_catalog", {})
    if (fixed.get("materialized") is not True or
            fixed.get("without_replacement") is not True or
            fixed.get("negatives") != config.get("sampled_negatives") or
            fixed.get("seed") != config.get("seed") or
            full.get("materialized") is not False):
        raise ArtifactIntegrityError("candidate policy config is inconsistent")
    if (not isinstance(config.get("sampled_negatives"), int) or
            config["sampled_negatives"] < 0 or
            not isinstance(config.get("seed"), int)):
        raise ArtifactIntegrityError("candidate seed and negative count must be integers")
    statistics = artifact.get("statistics")
    if not isinstance(statistics, dict):
        raise ArtifactIntegrityError("protocol statistics are missing")
    insufficient = statistics.get("insufficient_users")
    total_users = statistics.get("total_users")
    eligible_users = statistics.get("eligible_users")
    if (not isinstance(total_users, int) or not isinstance(eligible_users, int) or
            total_users < 0 or eligible_users < 0 or
            not isinstance(insufficient, (list, tuple)) or
            list(insufficient) != sorted(set(insufficient)) or
            total_users != eligible_users + len(insufficient)):
        raise ArtifactIntegrityError("user eligibility statistics are inconsistent")
    for name in ("repeated_validation_pairs_excluded",
                 "repeated_test_pairs_excluded"):
        if not isinstance(statistics.get(name), int) or statistics[name] < 0:
            raise ArtifactIntegrityError("repeat exclusion statistics are invalid")
    target_statistics = statistics.get("targets")
    if not isinstance(target_statistics, dict) or set(target_statistics) != set(ITEM_TYPES):
        raise ArtifactIntegrityError("target statistics are incomplete")
    if (not isinstance(artifact["targets"], dict) or
            set(artifact["targets"]) != set(ITEM_TYPES)):
        raise ArtifactIntegrityError("protocol must contain artist, album, and track targets")
    global_timestamps = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
        "test": defaultdict(list),
    }
    for target in ITEM_TYPES:
        payload = artifact["targets"][target]
        if not isinstance(payload, dict):
            raise ArtifactIntegrityError("target payload must be a mapping")
        splits_payload = payload.get("splits")
        candidates_payload = payload.get("candidates")
        if not isinstance(splits_payload, dict) or not isinstance(candidates_payload, dict):
            raise ArtifactIntegrityError("target splits and candidates must be mappings")
        catalog = payload.get("catalog")
        if not isinstance(catalog, (list, tuple)) or list(catalog) != sorted(set(catalog)):
            raise ArtifactIntegrityError("%s catalog must be sorted and unique" % target)
        catalog_items = tuple(catalog)
        catalog_set = set(catalog_items)
        known_rows = payload.get("known_positives")
        if (not isinstance(known_rows, (list, tuple)) or
                any(not isinstance(row, dict) for row in known_rows)):
            raise ArtifactIntegrityError("%s known positives are missing" % target)
        known_pairs = [(row.get("user_id"), row.get("item_id"))
                       for row in known_rows]
        if known_pairs != sorted(set(known_pairs)):
            raise ArtifactIntegrityError("%s known positives must be sorted and unique" % target)
        known_set = set(known_pairs)
        known_by_user = defaultdict(set)  # type: DefaultDict[int, set]
        for user_id, item_id in known_pairs:
            known_by_user[user_id].add(item_id)
        split_pairs = {}
        split_by_user = {}
        for split in ("train", "validation", "test"):
            rows = splits_payload.get(split)
            if (not isinstance(rows, (list, tuple)) or
                    any(not isinstance(row, dict) for row in rows)):
                raise ArtifactIntegrityError("%s %s split is missing" % (target, split))
            keys = [(row.get("user_id"), row.get("item_id")) for row in rows]
            if keys != sorted(set(keys)):
                raise ArtifactIntegrityError("%s %s rows must be sorted and unique" %
                                             (target, split))
            split_pairs[split] = set(keys)
            by_user = defaultdict(list)  # type: DefaultDict[int, list]
            for row in rows:
                required_row = ("user_id", "item_id", "play_count",
                                "first_timestamp", "last_timestamp")
                if any(name not in row for name in required_row):
                    raise ArtifactIntegrityError("incomplete %s %s split row" %
                                                 (target, split))
                if row["play_count"] <= 0:
                    raise ArtifactIntegrityError("split play_count must be positive")
                if row["first_timestamp"] > row["last_timestamp"]:
                    raise ArtifactIntegrityError("split timestamp range is invalid")
                if row["item_id"] not in catalog_set:
                    raise ArtifactIntegrityError("split item is outside the catalog")
                if (row["user_id"], row["item_id"]) not in known_set:
                    raise ArtifactIntegrityError("split pair is absent from known positives")
                by_user[row["user_id"]].append(row)
                global_timestamps[split][row["user_id"]].append(row)
            split_by_user[split] = by_user
        if (split_pairs["train"].intersection(split_pairs["validation"]) or
                split_pairs["train"].intersection(split_pairs["test"]) or
                split_pairs["validation"].intersection(split_pairs["test"])):
            raise ArtifactIntegrityError("target split pairs must be disjoint")
        expected_catalog = (set(row["item_id"] for row in payload["splits"]["train"])
                            if catalog_policy == "train_observed"
                            else set(item_id for unused_user, item_id in known_pairs))
        if catalog_set != expected_catalog:
            raise ArtifactIntegrityError("catalog does not match its configured policy")
        users = set().union(*(set(rows) for rows in split_by_user.values()))
        for user_id in users:
            train_rows = split_by_user["train"].get(user_id, ())
            validation_rows = split_by_user["validation"].get(user_id, ())
            test_rows = split_by_user["test"].get(user_id, ())
            if (train_rows and validation_rows and
                    max(row["last_timestamp"] for row in train_rows) >=
                    min(row["first_timestamp"] for row in validation_rows)):
                raise ArtifactIntegrityError("training must precede validation")
            if (train_rows and test_rows and
                    max(row["last_timestamp"] for row in train_rows) >=
                    min(row["first_timestamp"] for row in test_rows)):
                raise ArtifactIntegrityError("training must precede test")
            if (validation_rows and test_rows and
                    max(row["last_timestamp"] for row in validation_rows) >=
                    min(row["first_timestamp"] for row in test_rows)):
                raise ArtifactIntegrityError("validation must precede test")
        for split in ("validation", "test"):
            rows = candidates_payload.get(split)
            if (not isinstance(rows, (list, tuple)) or
                    any(not isinstance(row, dict) for row in rows)):
                raise ArtifactIntegrityError("%s %s candidates are missing" %
                                             (target, split))
            candidate_users = [row.get("user_id") for row in rows]
            expected_positives = defaultdict(set)  # type: DefaultDict[int, set]
            for row in payload["splits"][split]:
                expected_positives[row["user_id"]].add(row["item_id"])
            if candidate_users != sorted(expected_positives):
                raise ArtifactIntegrityError("candidate rows do not match split users")
            for row in rows:
                items = row.get("candidate_item_ids")
                positives = row.get("positive_item_ids")
                if (not isinstance(items, (list, tuple)) or
                        not isinstance(positives, (list, tuple)) or
                        list(items) != sorted(set(items)) or
                        not set(positives).issubset(set(items))):
                    raise ArtifactIntegrityError("invalid frozen candidate row")
                if list(positives) != sorted(expected_positives[row["user_id"]]):
                    raise ArtifactIntegrityError("candidate positives do not match split")
                if not set(items).issubset(catalog_set):
                    raise ArtifactIntegrityError("candidate item is outside the catalog")
                negatives = set(items) - set(positives)
                if negatives.intersection(known_by_user[row["user_id"]]):
                    raise ArtifactIntegrityError("candidate negatives contain known positives")
                eligible_count = len(catalog_set - known_by_user[row["user_id"]])
                expected_negatives = min(config["sampled_negatives"], eligible_count)
                if len(negatives) != expected_negatives:
                    raise ArtifactIntegrityError("fixed candidate count is incorrect")
                if row.get("candidate_hash") != candidate_digest(items):
                    raise ArtifactIntegrityError("candidate hash mismatch")
                expected = build_candidates(
                    row["user_id"], target, split,
                    expected_positives[row["user_id"]], catalog_items, known_pairs,
                    "fixed_sampled", config["sampled_negatives"], config["seed"],
                    known_by_user, catalog_items, catalog_set)
                if tuple(items) != expected.items:
                    raise ArtifactIntegrityError("fixed candidates do not match seed")
        target_stats = target_statistics.get(target)
        if not isinstance(target_stats, dict):
            raise ArtifactIntegrityError("target statistics are incomplete")
        cold = target_stats.get("cold_start_exclusions")
        if not isinstance(cold, dict) or set(cold) != {"validation", "test"}:
            raise ArtifactIntegrityError("cold-start statistics are incomplete")
        for split in ("validation", "test"):
            values = cold[split]
            if not isinstance(values, dict):
                raise ArtifactIntegrityError("cold-start statistics are inconsistent")
            user_ids = values.get("excluded_user_ids")
            pairs = values.get("excluded_positive_pairs")
            users_count = values.get("excluded_users")
            if (not isinstance(user_ids, (list, tuple)) or
                    list(user_ids) != sorted(set(user_ids)) or
                    users_count != len(user_ids) or
                    not isinstance(pairs, int) or pairs < users_count):
                raise ArtifactIntegrityError("cold-start statistics are inconsistent")
            if catalog_policy == "all_mapped" and (pairs != 0 or users_count != 0):
                raise ArtifactIntegrityError("transductive catalogs cannot report cold exclusions")
    global_users = set().union(*(set(rows) for rows in global_timestamps.values()))
    global_training_users = set(global_timestamps["train"])
    if catalog_policy == "train_observed":
        heldout_users = (set(global_timestamps["validation"]) |
                         set(global_timestamps["test"]))
        if not heldout_users.issubset(global_training_users):
            raise ArtifactIntegrityError("warm heldout users require training interactions")
    for user_id in global_users:
        validation_rows = global_timestamps["validation"].get(user_id, ())
        test_rows = global_timestamps["test"].get(user_id, ())
        validation_times = set()
        test_times = set()
        for row in validation_rows:
            if row["first_timestamp"] != row["last_timestamp"]:
                raise ArtifactIntegrityError("validation rows must be one timestamp group")
            validation_times.add(row["first_timestamp"])
        for row in test_rows:
            if row["first_timestamp"] != row["last_timestamp"]:
                raise ArtifactIntegrityError("test rows must be one timestamp group")
            test_times.add(row["first_timestamp"])
        if len(validation_times) > 1 or len(test_times) > 1:
            raise ArtifactIntegrityError("target relations do not share timestamp boundaries")
        if validation_times and test_times and min(test_times) <= min(validation_times):
            raise ArtifactIntegrityError("test timestamp must follow validation")
        train_rows = global_timestamps["train"].get(user_id, ())
        if validation_times and any(row["last_timestamp"] >= min(validation_times)
                                    for row in train_rows):
            raise ArtifactIntegrityError("all training rows must precede validation")
        if test_times and any(row["last_timestamp"] >= min(test_times)
                              for row in train_rows):
            raise ArtifactIntegrityError("all training rows must precede test")
    return dict(artifact)


def save_protocol_artifact(directory: str, artifact: Mapping[str, Any]) -> Dict[str, Any]:
    validated = validate_protocol_artifact(artifact)
    return save_bundle(directory, {"protocol": validated})


def load_protocol_artifact(
        directory: str, expected_protocol_hash: Optional[str] = None,
        expected_config_hash: Optional[str] = None) -> Dict[str, Any]:
    loaded = load_bundle(directory)
    if set(loaded) != {"protocol"}:
        raise ArtifactIntegrityError("protocol bundle must contain only protocol.json")
    return validate_protocol_artifact(loaded["protocol"], expected_protocol_hash,
                                      expected_config_hash)


def graph_input_from_protocol(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    """Build an interaction-only original-to-contiguous thesis graph mapping."""
    protocol = validate_protocol_artifact(artifact)
    ids = {node_type: set() for node_type in ("user",) + ITEM_TYPES}
    if protocol["config"]["catalog_policy"] == "train_observed":
        for target in ITEM_TYPES:
            payload = protocol["targets"][target]
            ids[target].update(payload["catalog"])
            for row in payload["splits"]["train"]:
                ids["user"].add(row["user_id"])
    else:
        for target in ITEM_TYPES:
            payload = protocol["targets"][target]
            for row in payload["known_positives"]:
                ids["user"].add(row["user_id"])
                ids[target].add(row["item_id"])
            for split in ("validation", "test"):
                for row in payload["candidates"][split]:
                    ids["user"].add(row["user_id"])
                    ids[target].update(row["candidate_item_ids"])
    mappings = {
        node_type: {str(original_id): internal_id
                    for internal_id, original_id in enumerate(sorted(values))}
        for node_type, values in ids.items()
    }
    return {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "node_counts": {node_type: len(mapping)
                        for node_type, mapping in mappings.items()},
        "node_mappings": mappings,
        "static_edges": [],
    }


def save_graph_input(path: str, artifact: Mapping[str, Any]) -> Dict[str, Any]:
    graph_input = graph_input_from_protocol(artifact)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as destination:
        destination.write(canonical_json(graph_input))
    return graph_input


def save_bundle(directory: str, artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    """Save named JSON artifacts and a content-addressed manifest."""
    if not artifacts:
        raise ValueError("artifacts must not be empty")
    os.makedirs(directory, exist_ok=True)
    files = {}  # type: Dict[str, str]
    for name in sorted(artifacts):
        if not name or "/" in name or "\\" in name or name == "manifest":
            raise ValueError("artifact names must be simple and non-empty")
        filename = name + ".json"
        content = canonical_json(artifacts[name])
        with open(os.path.join(directory, filename), "wb") as destination:
            destination.write(content)
        files[filename] = hashlib.sha256(content).hexdigest()
    identity = {"schema_version": SCHEMA_VERSION, "files": files}
    manifest = dict(identity)
    manifest["identity_sha256"] = sha256(identity)
    with open(os.path.join(directory, "manifest.json"), "wb") as destination:
        destination.write(canonical_json(manifest))
    return manifest


def load_bundle(directory: str) -> Dict[str, Any]:
    manifest_path = os.path.join(directory, "manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, ValueError) as error:
        raise ArtifactIntegrityError("cannot read manifest: %s" % error)
    identity = {"schema_version": manifest.get("schema_version"),
                "files": manifest.get("files")}
    if manifest.get("identity_sha256") != sha256(identity):
        raise ArtifactIntegrityError("manifest identity hash mismatch")
    loaded = {}
    for filename, expected in sorted((manifest.get("files") or {}).items()):
        path = os.path.join(directory, filename)
        try:
            with open(path, "rb") as source:
                content = source.read()
        except OSError as error:
            raise ArtifactIntegrityError("cannot read %s: %s" % (filename, error))
        if hashlib.sha256(content).hexdigest() != expected:
            raise ArtifactIntegrityError("artifact hash mismatch: %s" % filename)
        try:
            loaded[filename[:-5]] = json.loads(content.decode("utf-8"))
        except ValueError as error:
            raise ArtifactIntegrityError("invalid JSON in %s: %s" % (filename, error))
    return loaded
