import hashlib
import json
import os
import stat
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .canonical import canonical_json, sha256, to_primitive
from .candidates import SAMPLER_ALGORITHM, _build_candidates_indexed, candidate_digest
from .models import Event, ITEM_TYPES, PairInteraction
from .io import ListeningEventProvenanceStream
from .split import temporal_split

SCHEMA_VERSION = 2
GRAPH_INPUT_SCHEMA_VERSION = 2


class ArtifactIntegrityError(ValueError):
    pass


def _int(value: Any, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        raise ArtifactIntegrityError("%s must be %sa non-boolean integer" %
                                     (label, "a positive " if positive else ""))
    return value


def _split_row(row: PairInteraction) -> Dict[str, int]:
    return {"user_id": row.user_id, "item_id": row.item_id, "play_count": row.play_count,
            "first_timestamp": row.first_timestamp, "last_timestamp": row.last_timestamp}


def _candidate_row(candidate: Any) -> Dict[str, Any]:
    return {"user_id": candidate.user_id, "candidate_item_ids": candidate.items,
            "positive_item_ids": candidate.positives, "candidate_hash": candidate.candidate_hash}


def _protocol_identity(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in artifact.items()
            if key not in ("protocol_hash", "provenance", "created_at", "created_utc")}


def _candidate_exclusions(payload: Mapping[str, Any], split: str, repeat_policy: str,
                          horizon: str):
    """Return pairs ineligible as competitors for the declared task."""
    snapshots = payload["positive_snapshots"]
    if repeat_policy == "novel_only":
        names = (("train", "validation", "test") if horizon == "all_observed" else
                 (("train", "validation") if split == "validation" else
                  ("train", "validation", "test")))
    elif horizon == "all_observed" and split == "validation":
        # Oracle filtering may hide future-only items, never historical repeats.
        prior = set((pair["user_id"], pair["item_id"]) if isinstance(pair, dict) else pair
                    for name in ("train", "validation") for pair in snapshots[name])
        return tuple((pair["user_id"], pair["item_id"]) if isinstance(pair, dict) else pair
                     for pair in snapshots["test"]
                     if ((pair["user_id"], pair["item_id"]) if isinstance(pair, dict) else pair)
                     not in prior)
    else:
        names = ()
    return tuple((pair["user_id"], pair["item_id"]) if isinstance(pair, dict) else pair
                 for name in names for pair in snapshots[name])


def _retained_pairs(raw_pairs, split, repeat_policy):
    """Apply the declared pair task to raw per-window positive snapshots."""
    if split == "train" or repeat_policy == "repeat_allowed":
        return set(raw_pairs[split])
    earlier = set(raw_pairs["train"])
    if split == "test":
        earlier.update(raw_pairs["validation"])
    return set(raw_pairs[split]) - earlier


def prepare_protocol_artifact(events: Iterable[Event], sampled_negatives: int = 1000,
                               seed: int = 0, catalog_policy: str = "train_observed",
                               sources: Optional[Iterable[Mapping[str, Any]]] = None, *,
                               split_strategy: str,
                               validation_cutoff: Optional[int] = None,
                               test_cutoff: Optional[int] = None,
                               repeat_policy: str = "novel_only",
                               positive_filter_horizon: str = "as_of_split",
                               max_events: Optional[int] = None,
                               max_users: Optional[int] = None,
                               max_items_per_target: Optional[int] = None,
                               max_candidate_rows: Optional[int] = None,
                               max_candidate_items: Optional[int] = None) -> Dict[str, Any]:
    if isinstance(sampled_negatives, bool) or not isinstance(sampled_negatives, int) or sampled_negatives < 0:
        raise ValueError("sampled_negatives must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if catalog_policy not in ("train_observed", "all_mapped"):
        raise ValueError("invalid catalog policy")
    if positive_filter_horizon not in ("as_of_split", "all_observed"):
        raise ValueError("invalid positive filter horizon")
    bounds = {
        "events": max_events, "users": max_users,
        "items per target": max_items_per_target,
        "candidate rows": max_candidate_rows,
        "candidate items": max_candidate_items,
    }
    for label, value in bounds.items():
        if (value is not None and
                (isinstance(value, bool) or not isinstance(value, int) or
                 value <= 0)):
            raise ValueError("max_%s must be a positive integer or None" %
                             label.replace(" ", "_"))
    parser_stream = events if isinstance(events, ListeningEventProvenanceStream) else None
    if parser_stream is not None and sources is not None:
        raise ValueError("parser-bound streams cannot be combined with supplied sources")
    def bounded_events():
        event_count = 0
        users = set()
        items = {kind: set() for kind in ITEM_TYPES}
        for event in events:
            event_count += 1
            if max_events is not None and event_count > max_events:
                raise ValueError("protocol input exceeds its event limit")
            if (isinstance(event, Event) and
                    not isinstance(event.user_id, bool) and
                    isinstance(event.user_id, int)):
                users.add(event.user_id)
                if max_users is not None and len(users) > max_users:
                    raise ValueError("protocol input exceeds its user limit")
                for kind in ITEM_TYPES:
                    item = getattr(event, kind + "_id")
                    if (item is not None and not isinstance(item, bool) and
                            isinstance(item, int)):
                        items[kind].add(item)
                        if (max_items_per_target is not None and
                                len(items[kind]) > max_items_per_target):
                            raise ValueError(
                                "protocol input exceeds its per-target item limit")
            yield event

    bounded = any(value is not None for value in (
        max_events, max_users, max_items_per_target))
    protocol = temporal_split(bounded_events() if bounded else events,
                              strategy=split_strategy,
                              validation_cutoff=validation_cutoff, test_cutoff=test_cutoff,
                              repeat_policy=repeat_policy)
    config = {"split_strategy": split_strategy, "validation_cutoff": validation_cutoff,
              "test_cutoff": test_cutoff, "repeat_policy": repeat_policy,
              "positive_filter_horizon": positive_filter_horizon,
              "catalog_policy": catalog_policy,
              "catalog_uses_future_information": catalog_policy == "all_mapped",
              "temporal_scope": "global" if split_strategy == "global_time_cutoffs" else "user_relative",
              "candidate_filter_uses_future_information": positive_filter_horizon == "all_observed",
              "globally_time_causal": (split_strategy == "global_time_cutoffs" and
                                        catalog_policy == "train_observed" and
                                        positive_filter_horizon == "as_of_split"),
              "minimum_timestamp_groups": 3 if split_strategy == "per_user_last_timestamp_groups" else None,
              "sampled_negatives": sampled_negatives, "seed": seed,
              "candidate_sampler_algorithm": SAMPLER_ALGORITHM,
              "candidate_policies": {"fixed_sampled": {"materialized": True,
                  "negatives": sampled_negatives, "seed": seed, "without_replacement": True,
                  "algorithm": SAMPLER_ALGORITHM}, "full_catalog": {"materialized": False}}}
    targets = {}
    target_statistics = {}
    candidate_row_count = 0
    candidate_item_count = 0
    for kind in ITEM_TYPES:
        raw = {name: tuple(_split_row(row) for row in getattr(protocol.raw_split, name)
                           if row.item_type == kind) for name in ("train", "validation", "test")}
        catalog = (tuple(sorted({row["item_id"] for row in raw["train"]}))
                   if catalog_policy == "train_observed" else protocol.catalogs[kind])
        catalog_set = set(catalog)
        snapshots = {name: tuple({"user_id": u, "item_id": i}
                                 for u, i in protocol.window_positives[kind][name])
                     for name in ("train", "validation", "test")}
        snapshot_pairs = {name: set(protocol.window_positives[kind][name])
                          for name in ("train", "validation", "test")}
        task_pairs = {name: _retained_pairs(snapshot_pairs, name, repeat_policy)
                      for name in ("train", "validation", "test")}
        splits = {"train": tuple(dict(row) for row in raw["train"]
                                  if (row["user_id"], row["item_id"]) in task_pairs["train"])}
        cold = {}
        for name in ("validation", "test"):
            task_rows = tuple(row for row in raw[name]
                              if (row["user_id"], row["item_id"]) in task_pairs[name])
            train_users = {user for user, item in snapshot_pairs["train"]}
            included, excluded = [], []
            for row in task_rows:
                (included if catalog_policy == "all_mapped" or
                 (row["user_id"] in train_users and row["item_id"] in catalog_set) else excluded).append(row)
            included, excluded = tuple(dict(row) for row in included), tuple(excluded)
            splits[name] = included
            ids = tuple(sorted({row["user_id"] for row in excluded}))
            reasons = {}
            for reason, predicate in (
                    ("no_same_target_training_history_only", lambda r: r["user_id"] not in train_users and r["item_id"] in catalog_set),
                    ("unseen_training_catalog_item_only", lambda r: r["user_id"] in train_users and r["item_id"] not in catalog_set),
                    ("no_same_target_training_history_and_unseen_training_catalog_item", lambda r: r["user_id"] not in train_users and r["item_id"] not in catalog_set)):
                rows = tuple(row for row in excluded if predicate(row))
                reason_ids = tuple(sorted({row["user_id"] for row in rows}))
                reasons[reason] = {"excluded_positive_pairs": len(rows),
                                   "excluded_user_ids": reason_ids, "excluded_users": len(reason_ids)}
            cold[name] = dict({"excluded_positive_pairs": len(excluded), "excluded_user_ids": ids,
                               "excluded_users": len(ids)}, **reasons)
        payload = {"catalog": catalog,
                   "known_positives": tuple({"user_id": u, "item_id": i}
                                             for u, i in protocol.known_positives[kind]),
                    "positive_snapshots": snapshots, "raw_splits": raw, "splits": splits, "candidates": {}}
        for name in ("validation", "test"):
            positives = defaultdict(set)
            for row in splits[name]: positives[row["user_id"]].add(row["item_id"])
            known = _candidate_exclusions(payload, name, repeat_policy, positive_filter_horizon)
            index = defaultdict(set)
            for user_id, item_id in known: index[user_id].add(item_id)
            candidate_values = []
            for user in sorted(positives):
                candidate_row_count += 1
                if (max_candidate_rows is not None and
                        candidate_row_count > max_candidate_rows):
                    raise ValueError("protocol exceeds its candidate-row limit")
                value = _candidate_row(_build_candidates_indexed(
                    user, kind, name, positives[user], catalog, "fixed_sampled",
                    sampled_negatives, seed, index[user]))
                candidate_item_count += len(value["candidate_item_ids"])
                if (max_candidate_items is not None and
                        candidate_item_count > max_candidate_items):
                    raise ValueError("protocol exceeds its candidate-item limit")
                candidate_values.append(value)
            payload["candidates"][name] = tuple(candidate_values)
        targets[kind] = payload
        target_statistics[kind] = {"cold_start_exclusions": cold}
    normalized_sources, paths = [], []
    if parser_stream is not None:
        source = parser_stream.completed_record()
        _validate_source_record(source, ValueError)
        normalized_sources.append(source); paths.append(parser_stream.path)
    for source in sources or ():
        if not isinstance(source, Mapping): raise ValueError("source provenance must be a mapping")
        value = to_primitive(source); path = value.pop("path", None)
        if value.get("provenance_binding") == "parser_bound":
            raise ValueError("caller-supplied sources cannot claim parser_bound")
        value["provenance_binding"] = "self_attested"
        _validate_source_record(value, ValueError)
        if path is not None and not isinstance(path, str):
            raise ValueError("source path must be a string or None")
        normalized_sources.append(value)
        paths.append(path)
    artifact = {"schema_version": SCHEMA_VERSION, "config": config, "config_hash": sha256(config),
                "sources": tuple(normalized_sources), "provenance": {"source_paths": tuple(paths)},
                "statistics": dict(to_primitive(protocol.statistics), targets=target_statistics), "targets": targets}
    artifact["protocol_hash"] = sha256(_protocol_identity(artifact))
    return artifact


def _pairs(rows: Any, label: str) -> Tuple[Tuple[int, int], ...]:
    if not isinstance(rows, (list, tuple)): raise ArtifactIntegrityError(label + " must be a sequence")
    pairs = []
    for row in rows:
        if not isinstance(row, dict): raise ArtifactIntegrityError(label + " row must be a mapping")
        pairs.append((_int(row.get("user_id"), label + " user_id"), _int(row.get("item_id"), label + " item_id")))
    if pairs != sorted(set(pairs)): raise ArtifactIntegrityError(label + " must be sorted and unique")
    return tuple(pairs)


_LFM_COLUMNS = ("user_id", "artist_id", "album_id", "track_id", "timestamp")


def _validate_source_record(source: Any, error_type: Any = ArtifactIntegrityError) -> None:
    """Validate identity-bearing raw-input provenance for supplied sources."""
    if not isinstance(source, dict):
        raise error_type("source record must be a mapping")
    required = {"provenance_binding", "source_kind", "parser_version", "bytes", "sha256",
                "has_header", "column_order", "parsed_row_count"}
    if set(source) != required:
        raise error_type("source record is incomplete")
    if source["provenance_binding"] not in ("parser_bound", "self_attested"):
        raise error_type("source provenance binding is invalid")
    if source["source_kind"] != "listening_events" or source["parser_version"] != "lfm_listening_events_tsv_v1":
        raise error_type("source parser identity is invalid")
    if (isinstance(source["bytes"], bool) or not isinstance(source["bytes"], int) or
             source["bytes"] < 0):
        raise error_type("source bytes must be non-negative integer")
    if (not isinstance(source["sha256"], str) or len(source["sha256"]) != 64 or
            any(ch not in "0123456789abcdef" for ch in source["sha256"])):
        raise error_type("source sha256 is invalid")
    if not isinstance(source["has_header"], bool):
        raise error_type("source has_header must be boolean")
    if (not isinstance(source["column_order"], (list, tuple)) or
            tuple(source["column_order"]) != _LFM_COLUMNS):
        raise error_type("source column order is invalid")
    if (isinstance(source["parsed_row_count"], bool) or
            not isinstance(source["parsed_row_count"], int) or source["parsed_row_count"] < 0):
        raise error_type("source parsed row count is invalid")


def validate_protocol_artifact(artifact: Mapping[str, Any], expected_protocol_hash: Optional[str] = None,
                               expected_config_hash: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not isinstance(artifact, Mapping): raise ArtifactIntegrityError("protocol artifact must be a mapping")
        required_keys = {"schema_version", "config", "config_hash", "protocol_hash",
                         "sources", "provenance", "targets", "statistics"}
        if set(artifact) != required_keys:
            raise ArtifactIntegrityError("protocol artifact has an invalid shape")
        if (isinstance(artifact["schema_version"], bool) or
                not isinstance(artifact["schema_version"], int) or
                artifact["schema_version"] != SCHEMA_VERSION):
            raise ArtifactIntegrityError("unsupported protocol schema version")
        if not isinstance(artifact["config"], dict): raise ArtifactIntegrityError("protocol config must be a mapping")
        if not all(isinstance(artifact[x], str) and len(artifact[x]) == 64 and
                   all(c in "0123456789abcdef" for c in artifact[x]) for x in ("config_hash", "protocol_hash")):
            raise ArtifactIntegrityError("protocol hashes must be lowercase SHA256 digests")
        if artifact["config_hash"] != sha256(artifact["config"]) or artifact["protocol_hash"] != sha256(_protocol_identity(artifact)):
            raise ArtifactIntegrityError("protocol hash mismatch")
        if not isinstance(artifact["sources"], (list, tuple)):
            raise ArtifactIntegrityError("sources must be a sequence")
        for source in artifact["sources"]:
            _validate_source_record(source)
        provenance = artifact["provenance"]
        if (not isinstance(provenance, dict) or set(provenance) != {"source_paths"} or
                not isinstance(provenance["source_paths"], (list, tuple)) or
                len(provenance["source_paths"]) != len(artifact["sources"]) or
                any(path is not None and not isinstance(path, str)
                    for path in provenance["source_paths"])):
            raise ArtifactIntegrityError("source path provenance is invalid")
        if expected_protocol_hash is not None and artifact["protocol_hash"] != expected_protocol_hash: raise ArtifactIntegrityError("expected protocol hash does not match")
        if expected_config_hash is not None and artifact["config_hash"] != expected_config_hash: raise ArtifactIntegrityError("expected config hash does not match")
        c = artifact["config"]
        strategy = c.get("split_strategy")
        if strategy not in ("per_user_last_timestamp_groups", "global_time_cutoffs"): raise ArtifactIntegrityError("invalid split strategy")
        if c.get("repeat_policy") not in ("novel_only", "repeat_allowed") or c.get("positive_filter_horizon") not in ("as_of_split", "all_observed"):
            raise ArtifactIntegrityError("invalid repeat or positive filtering policy")
        if c.get("catalog_policy") not in ("train_observed", "all_mapped") or c.get("catalog_uses_future_information") != (c.get("catalog_policy") == "all_mapped"):
            raise ArtifactIntegrityError("invalid catalog policy")
        if (c.get("temporal_scope") != ("global" if strategy == "global_time_cutoffs" else "user_relative") or
                c.get("candidate_filter_uses_future_information") != (c.get("positive_filter_horizon") == "all_observed") or
                c.get("globally_time_causal") != (strategy == "global_time_cutoffs" and c.get("catalog_policy") == "train_observed" and c.get("positive_filter_horizon") == "as_of_split")):
            raise ArtifactIntegrityError("temporal provenance metadata is inconsistent")
        if strategy == "global_time_cutoffs":
            _int(c.get("validation_cutoff"), "validation cutoff"); _int(c.get("test_cutoff"), "test cutoff")
            if c["validation_cutoff"] >= c["test_cutoff"] or c.get("minimum_timestamp_groups") is not None: raise ArtifactIntegrityError("invalid global cutoffs")
        elif c.get("validation_cutoff") is not None or c.get("test_cutoff") is not None or c.get("minimum_timestamp_groups") != 3:
            raise ArtifactIntegrityError("invalid per-user split configuration")
        for name in ("catalog_uses_future_information", "candidate_filter_uses_future_information", "globally_time_causal"):
            if not isinstance(c.get(name), bool):
                raise ArtifactIntegrityError("temporal provenance metadata must be boolean")
        _int(c.get("sampled_negatives"), "sampled_negatives"); _int(c.get("seed"), "seed")
        if c["sampled_negatives"] < 0: raise ArtifactIntegrityError("sampled_negatives must not be negative")
        policies = c.get("candidate_policies")
        if not isinstance(policies, dict) or set(policies) != {"fixed_sampled", "full_catalog"}:
            raise ArtifactIntegrityError("candidate policy config is inconsistent")
        fixed = policies["fixed_sampled"]
        full = policies["full_catalog"]
        if (not isinstance(fixed, dict) or set(fixed) != {"materialized", "negatives", "seed", "without_replacement", "algorithm"} or
                fixed.get("materialized") is not True or fixed.get("algorithm") != SAMPLER_ALGORITHM or
                c.get("candidate_sampler_algorithm") != SAMPLER_ALGORITHM or fixed.get("negatives") != c["sampled_negatives"] or
                fixed.get("seed") != c["seed"] or fixed.get("without_replacement") is not True or
                not isinstance(full, dict) or set(full) != {"materialized"} or full.get("materialized") is not False):
            raise ArtifactIntegrityError("candidate policy config is inconsistent")
        if not isinstance(artifact["targets"], dict) or set(artifact["targets"]) != set(ITEM_TYPES): raise ArtifactIntegrityError("target payloads are incomplete")
        statistics = artifact["statistics"]
        if not isinstance(statistics, dict) or not isinstance(statistics.get("targets"), dict) or set(statistics["targets"]) != set(ITEM_TYPES): raise ArtifactIntegrityError("protocol statistics are incomplete")
        for name in ("total_users", "eligible_users"):
            _int(statistics.get(name), name)
            if statistics[name] < 0:
                raise ArtifactIntegrityError("%s must be non-negative" % name)
        insufficient = statistics.get("insufficient_users")
        if (not isinstance(insufficient, (list, tuple)) or
                any(isinstance(user, bool) or not isinstance(user, int)
                    for user in insufficient) or
                list(insufficient) != sorted(set(insufficient)) or
                statistics["total_users"] != statistics["eligible_users"] + len(insufficient)):
            raise ArtifactIntegrityError("user eligibility statistics are inconsistent")
        if strategy == "global_time_cutoffs" and (statistics["insufficient_users"] or
                                                   statistics["eligible_users"] != statistics["total_users"]):
            raise ArtifactIntegrityError("global split user eligibility is inconsistent")
        for name in ("repeated_validation_pairs_excluded", "repeated_test_pairs_excluded"):
            _int(statistics.get(name), name)
            if statistics[name] < 0:
                raise ArtifactIntegrityError("%s must be non-negative" % name)
        per_user_boundaries = {"validation": {}, "test": {}}
        per_user_train_last = {}
        raw_split_user_ids = set()
        repeat_counts = {"validation": 0, "test": 0}
        for kind in ITEM_TYPES:
            payload = artifact["targets"][kind]
            if not isinstance(payload, dict): raise ArtifactIntegrityError("target payload must be a mapping")
            catalog = payload.get("catalog")
            if not isinstance(catalog, (list, tuple)) or any(isinstance(x, bool) or not isinstance(x, int) for x in catalog) or list(catalog) != sorted(set(catalog)):
                raise ArtifactIntegrityError("catalog must be sorted integer IDs")
            catalog_set = set(catalog); known = _pairs(payload.get("known_positives"), "known positives")
            snapshots = payload.get("positive_snapshots")
            if not isinstance(snapshots, dict) or set(snapshots) != {"train", "validation", "test"}: raise ArtifactIntegrityError("positive snapshots missing")
            raw_splits = payload.get("raw_splits")
            if not isinstance(raw_splits, dict) or set(raw_splits) != {"train", "validation", "test"}:
                raise ArtifactIntegrityError("raw splits missing")
            raw_rows = {}
            for n, rows in raw_splits.items():
                if not isinstance(rows, (list, tuple)):
                    raise ArtifactIntegrityError("raw split rows must be a sequence")
                normalized = []
                for row in rows:
                    if not isinstance(row, dict) or set(row) != {"user_id", "item_id", "play_count", "first_timestamp", "last_timestamp"}:
                        raise ArtifactIntegrityError("invalid raw split row")
                    u, i = _int(row.get("user_id"), "user_id"), _int(row.get("item_id"), "item_id")
                    raw_split_user_ids.add(u)
                    play, first, last = (_int(row.get("play_count"), "play_count", True),
                                         _int(row.get("first_timestamp"), "timestamp"),
                                         _int(row.get("last_timestamp"), "timestamp"))
                    if first > last or (u, i) not in known:
                        raise ArtifactIntegrityError("invalid raw split row")
                    if strategy == "global_time_cutoffs":
                        v, t = c["validation_cutoff"], c["test_cutoff"]
                        if not ((n == "train" and last < v) or (n == "validation" and v <= first and last < t) or (n == "test" and first >= t)):
                            raise ArtifactIntegrityError("raw split row violates global cutoff")
                    elif n in per_user_boundaries:
                        if first != last:
                            raise ArtifactIntegrityError("raw timestamp boundaries must preserve ties")
                        previous = per_user_boundaries[n].setdefault(u, first)
                        if previous != first:
                            raise ArtifactIntegrityError("raw timestamp boundaries differ across targets")
                    if strategy == "per_user_last_timestamp_groups" and n == "train":
                        per_user_train_last[u] = max(per_user_train_last.get(u, last), last)
                    normalized.append(row)
                if [(r["user_id"], r["item_id"]) for r in normalized] != sorted((r["user_id"], r["item_id"]) for r in normalized) or len({(r["user_id"], r["item_id"]) for r in normalized}) != len(normalized):
                    raise ArtifactIntegrityError("raw split rows must be sorted and unique")
                raw_rows[n] = tuple(normalized)
            for n in snapshots:
                snapshot_values = _pairs(snapshots[n], n + " positive snapshots")
                if set(snapshot_values) != {(r["user_id"], r["item_id"]) for r in raw_rows[n]}:
                    raise ArtifactIntegrityError("positive snapshots do not match raw splits")
            splits = payload.get("splits")
            if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}: raise ArtifactIntegrityError("splits missing")
            split_pairs = {}
            for n, rows in splits.items():
                if not isinstance(rows, (list, tuple)): raise ArtifactIntegrityError("split rows must be a sequence")
                keys=[]
                for row in rows:
                    if not isinstance(row, dict) or set(row) != {"user_id", "item_id", "play_count", "first_timestamp", "last_timestamp"}: raise ArtifactIntegrityError("split row must be a mapping")
                    u,i = _int(row.get("user_id"), "user_id"), _int(row.get("item_id"), "item_id")
                    p,first,last = _int(row.get("play_count"), "play_count", True), _int(row.get("first_timestamp"), "timestamp"), _int(row.get("last_timestamp"), "timestamp")
                    if first > last or i not in catalog_set or (u,i) not in known: raise ArtifactIntegrityError("invalid split row")
                    if strategy == "global_time_cutoffs":
                        v,t=c["validation_cutoff"],c["test_cutoff"]
                        if not ((n == "train" and last < v) or (n == "validation" and v <= first and last < t) or (n == "test" and first >= t)):
                            raise ArtifactIntegrityError("split row violates global cutoff")
                    keys.append((u,i))
                if keys != sorted(set(keys)): raise ArtifactIntegrityError("split rows must be sorted and unique")
                split_pairs[n]=set(keys)
            if c["repeat_policy"] == "novel_only" and (split_pairs["train"] & split_pairs["validation"] or split_pairs["train"] & split_pairs["test"] or split_pairs["validation"] & split_pairs["test"]): raise ArtifactIntegrityError("novel-only split pairs overlap")
            raw_pairs = {name: set(_pairs(snapshots[name], name + " positive snapshots"))
                         for name in ("train", "validation", "test")}
            for name in ("validation", "test"):
                repeat_counts[name] += len(raw_pairs[name] - _retained_pairs(
                    raw_pairs, name, c["repeat_policy"]))
            for name in ("train", "validation", "test"):
                task_pairs = _retained_pairs(raw_pairs, name, c["repeat_policy"])
                if name == "train" or c["catalog_policy"] == "all_mapped":
                    expected_pairs = {pair for pair in task_pairs if pair[1] in catalog_set}
                else:
                    target_train_users = {user for user, item in raw_pairs["train"]}
                    expected_pairs = {pair for pair in task_pairs
                                       if pair[1] in catalog_set and pair[0] in target_train_users}
                expected_rows = tuple(row for row in raw_rows[name]
                                      if (row["user_id"], row["item_id"]) in expected_pairs)
                if split_pairs[name] != expected_pairs or tuple(splits[name]) != expected_rows:
                    if strategy == "per_user_last_timestamp_groups":
                        raise ArtifactIntegrityError("timestamp boundaries differ from raw splits")
                    raise ArtifactIntegrityError("retained split pairs do not match snapshots and policy")
            expected_catalog = {r["item_id"] for r in splits["train"]} if c["catalog_policy"] == "train_observed" else {i for u,i in known}
            if catalog_set != expected_catalog: raise ArtifactIntegrityError("catalog does not match policy")
            candidates = payload.get("candidates")
            if not isinstance(candidates, dict) or set(candidates) != {"validation", "test"}: raise ArtifactIntegrityError("candidates missing")
            for n in ("validation", "test"):
                expected_pos=defaultdict(set)
                for row in splits[n]: expected_pos[row["user_id"]].add(row["item_id"])
                rows=candidates[n]
                if not isinstance(rows, (list, tuple)) or [r.get("user_id") if isinstance(r,dict) else None for r in rows] != sorted(expected_pos): raise ArtifactIntegrityError("candidate users do not match targets")
                horizon=_candidate_exclusions(payload,n,c["repeat_policy"],c["positive_filter_horizon"]); index=defaultdict(set)
                for u,i in horizon:index[u].add(i)
                for row in rows:
                    if not isinstance(row,dict): raise ArtifactIntegrityError("candidate row must be a mapping")
                    u=_int(row.get("user_id"),"candidate user_id"); items=row.get("candidate_item_ids"); positives=row.get("positive_item_ids")
                    if not isinstance(items,(list,tuple)) or not isinstance(positives,(list,tuple)) or any(isinstance(i,bool) or not isinstance(i,int) for i in list(items)+list(positives)) or list(items)!=sorted(set(items)) or tuple(positives)!=tuple(sorted(expected_pos[u])) or not set(positives).issubset(items): raise ArtifactIntegrityError("invalid candidate row")
                    expected=_build_candidates_indexed(u,kind,n,expected_pos[u],catalog,"fixed_sampled",c["sampled_negatives"],c["seed"],index[u])
                    if tuple(items)!=expected.items or row.get("candidate_hash")!=candidate_digest(items): raise ArtifactIntegrityError("candidate rows do not match deterministic sampler")
            cold = statistics["targets"][kind].get("cold_start_exclusions") if isinstance(statistics["targets"][kind], dict) else None
            if not isinstance(cold, dict) or set(cold) != {"validation", "test"}: raise ArtifactIntegrityError("cold-start statistics are incomplete")
            for n in ("validation", "test"):
                value=cold[n]
                reason_names = {"no_same_target_training_history_only", "unseen_training_catalog_item_only", "no_same_target_training_history_and_unseen_training_catalog_item"}
                if not isinstance(value,dict) or set(value) != {"excluded_positive_pairs", "excluded_user_ids", "excluded_users"}.union(reason_names): raise ArtifactIntegrityError("cold-start statistics are inconsistent")
                task_pairs = _retained_pairs(raw_pairs, n, c["repeat_policy"])
                excluded = task_pairs - split_pairs[n]
                expected_ids = tuple(sorted({user for user, item in excluded}))
                expected_reason_pairs = {
                    "no_same_target_training_history_only": {p for p in excluded if p[0] not in {u for u, i in raw_pairs["train"]} and p[1] in catalog_set},
                    "unseen_training_catalog_item_only": {p for p in excluded if p[0] in {u for u, i in raw_pairs["train"]} and p[1] not in catalog_set},
                    "no_same_target_training_history_and_unseen_training_catalog_item": {p for p in excluded if p[0] not in {u for u, i in raw_pairs["train"]} and p[1] not in catalog_set}}
                def valid_summary(summary, pairs):
                    ids = tuple(sorted({user for user, item in pairs}))
                    return (isinstance(summary, dict) and set(summary) == {"excluded_positive_pairs", "excluded_user_ids", "excluded_users"} and
                            not isinstance(summary["excluded_positive_pairs"], bool) and isinstance(summary["excluded_positive_pairs"], int) and summary["excluded_positive_pairs"] >= 0 and
                            isinstance(summary["excluded_user_ids"], (list, tuple)) and
                            all(not isinstance(user, bool) and isinstance(user, int) for user in summary["excluded_user_ids"]) and
                            tuple(summary["excluded_user_ids"]) == ids and
                            not isinstance(summary["excluded_users"], bool) and isinstance(summary["excluded_users"], int) and summary["excluded_users"] >= 0 and
                            summary["excluded_positive_pairs"] == len(pairs) and summary["excluded_users"] == len(ids))
                if (not valid_summary({k: value[k] for k in ("excluded_positive_pairs", "excluded_user_ids", "excluded_users")}, excluded) or
                        any(not valid_summary(value[reason], pairs) for reason, pairs in expected_reason_pairs.items())):
                    raise ArtifactIntegrityError("cold-start statistics are inconsistent")
        if (statistics["repeated_validation_pairs_excluded"] != repeat_counts["validation"] or
                statistics["repeated_test_pairs_excluded"] != repeat_counts["test"]):
            raise ArtifactIntegrityError("repeat exclusion statistics are inconsistent")
        if set(insufficient).intersection(raw_split_user_ids):
            raise ArtifactIntegrityError("insufficient users cannot appear in raw split rows")
        for user_id, validation_time in per_user_boundaries["validation"].items():
            if per_user_train_last.get(user_id, validation_time - 1) >= validation_time:
                raise ArtifactIntegrityError("training must precede validation")
            test_time = per_user_boundaries["test"].get(user_id)
            if test_time is not None and validation_time >= test_time:
                raise ArtifactIntegrityError("validation must precede test")
        for user_id, test_time in per_user_boundaries["test"].items():
            if per_user_train_last.get(user_id, test_time - 1) >= test_time:
                raise ArtifactIntegrityError("training must precede test")
        return dict(artifact)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        if isinstance(error, ArtifactIntegrityError): raise
        raise ArtifactIntegrityError("malformed protocol artifact: %s" % error)


def _candidate_rows_for_validated_protocol(protocol: Mapping[str, Any], target: str,
                                           split: str, policy: str) -> Tuple[Dict[str, Any], ...]:
    """Internal only: callers hold a protocol just validated at their boundary."""
    if target not in ITEM_TYPES or split not in ("validation","test") or policy not in ("fixed_sampled","full_catalog"): raise ValueError("invalid candidate request")
    payload=protocol["targets"][target]
    if policy == "fixed_sampled": return tuple(dict(row) for row in payload["candidates"][split])
    positives=defaultdict(set)
    for row in payload["splits"][split]: positives[row["user_id"]].add(row["item_id"])
    horizon=_candidate_exclusions(payload,split,protocol["config"]["repeat_policy"],protocol["config"]["positive_filter_horizon"])
    index=defaultdict(set)
    for u,i in horizon:index[u].add(i)
    return tuple(_candidate_row(_build_candidates_indexed(u,target,split,positives[u],payload["catalog"],"full_catalog",protocol["config"]["sampled_negatives"],protocol["config"]["seed"],index[u])) for u in sorted(positives))


def candidate_rows_for_policy(artifact: Mapping[str, Any], target: str, split: str, policy: str) -> Tuple[Dict[str, Any], ...]:
    """Return candidate rows after strict validation of an external artifact."""
    return _candidate_rows_for_validated_protocol(
        validate_protocol_artifact(artifact), target, split, policy)


def save_protocol_artifact(directory: str, artifact: Mapping[str, Any]) -> Dict[str, Any]: return save_bundle(directory,{"protocol":validate_protocol_artifact(artifact)})
def load_protocol_artifact(directory: str, expected_protocol_hash: Optional[str]=None, expected_config_hash: Optional[str]=None) -> Dict[str,Any]:
    loaded=load_bundle(directory, max_artifact_count=1)
    if set(loaded)!={"protocol"}: raise ArtifactIntegrityError("protocol bundle must contain only protocol.json")
    return validate_protocol_artifact(loaded["protocol"],expected_protocol_hash,expected_config_hash)


def graph_input_from_protocol(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    protocol=validate_protocol_artifact(artifact); ids={x:set() for x in ("user",)+ITEM_TYPES}
    for kind in ITEM_TYPES:
        for row in protocol["targets"][kind]["splits"]["train"]:
            ids["user"].add(row["user_id"]); ids[kind].add(row["item_id"])
        if protocol["config"]["catalog_policy"] == "all_mapped":
            for row in protocol["targets"][kind]["known_positives"]:
                ids["user"].add(row["user_id"]); ids[kind].add(row["item_id"])
    mappings={kind:{str(value):n for n,value in enumerate(sorted(values))} for kind,values in ids.items()}
    edges={}
    for kind in ITEM_TYPES:
        edges[kind]=tuple({"user_id":mappings["user"][str(r["user_id"])],"item_id":mappings[kind][str(r["item_id"])],"play_count":r["play_count"],"first_timestamp":r["first_timestamp"],"last_timestamp":r["last_timestamp"]} for r in protocol["targets"][kind]["splits"]["train"])
    graph={"schema_version":GRAPH_INPUT_SCHEMA_VERSION,"protocol_hash":protocol["protocol_hash"],"node_counts":{k:len(v) for k,v in mappings.items()},"node_mappings":mappings,"train_interaction_edges":edges,"static_edges":[]}
    validate_graph_input(graph); return graph


def validate_graph_input(graph: Mapping[str, Any]) -> Dict[str,Any]:
    if (not isinstance(graph,dict) or set(graph) != {"schema_version", "protocol_hash", "node_counts", "node_mappings", "train_interaction_edges", "static_edges"} or
            isinstance(graph.get("schema_version"), bool) or not isinstance(graph.get("schema_version"), int) or graph.get("schema_version") != GRAPH_INPUT_SCHEMA_VERSION): raise ArtifactIntegrityError("unsupported graph input schema")
    if not isinstance(graph["protocol_hash"], str) or len(graph["protocol_hash"]) != 64 or any(ch not in "0123456789abcdef" for ch in graph["protocol_hash"]): raise ArtifactIntegrityError("invalid graph protocol hash")
    mappings=graph.get("node_mappings"); edges=graph.get("train_interaction_edges")
    if (not isinstance(mappings,dict) or set(mappings) != set(("user",)+ITEM_TYPES) or
            not isinstance(edges,dict) or set(edges) != set(ITEM_TYPES) or
            not isinstance(graph.get("node_counts"),dict) or set(graph["node_counts"]) != set(("user",)+ITEM_TYPES) or
            not isinstance(graph.get("static_edges"), list) or graph["static_edges"]): raise ArtifactIntegrityError("invalid graph input")
    for kind in ("user",)+ITEM_TYPES:
        mapping=mappings.get(kind)
        if (not isinstance(mapping,dict) or
                any(not isinstance(key, str) or str(int(key)) != key or
                    isinstance(value, bool) or not isinstance(value, int)
                    for key, value in mapping.items()) or
                len(set(mapping.values())) != len(mapping) or
                sorted(mapping.values()) != list(range(len(mapping))) or
                isinstance(graph["node_counts"][kind], bool) or
                not isinstance(graph["node_counts"][kind], int) or
                graph["node_counts"][kind] < 0 or
                graph["node_counts"][kind] != len(mapping)):
            raise ArtifactIntegrityError("graph mappings must be contiguous")
    for kind in ITEM_TYPES:
        if not isinstance(edges.get(kind), (list,tuple)): raise ArtifactIntegrityError("graph edges missing")
        for row in edges[kind]:
            if not isinstance(row,dict) or set(row) != {"user_id","item_id","play_count","first_timestamp","last_timestamp"}: raise ArtifactIntegrityError("graph edge must be mapping")
            for field in ("user_id","item_id","play_count","first_timestamp","last_timestamp"): _int(row.get(field),field,field=="play_count")
            if row["user_id"] not in mappings["user"].values() or row["item_id"] not in mappings[kind].values() or row["first_timestamp"]>row["last_timestamp"]: raise ArtifactIntegrityError("invalid graph edge")
        edge_keys = [(row["user_id"], row["item_id"], row["play_count"],
                      row["first_timestamp"], row["last_timestamp"])
                     for row in edges[kind]]
        if edge_keys != sorted(edge_keys) or len(edge_keys) != len(set(edge_keys)):
            raise ArtifactIntegrityError("graph edges must be sorted and unique")
    return dict(graph)


def save_graph_input(path: str, artifact: Mapping[str, Any]) -> Dict[str, Any]:
    graph=graph_input_from_protocol(artifact); parent=os.path.dirname(os.path.abspath(path)); os.makedirs(parent,exist_ok=True)
    with open(path,"wb") as f:f.write(canonical_json(graph))
    return graph


def _safe_artifact_name(name: Any) -> bool:
    return (isinstance(name, str) and bool(name) and name != "manifest" and
            all(character.isalnum() or character in "_-" for character in name))


def _safe_filename(filename: Any) -> bool:
    return (isinstance(filename, str) and filename.endswith(".json") and
            _safe_artifact_name(filename[:-5]))

def save_bundle(directory: str, artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(artifacts,Mapping) or not artifacts: raise ValueError("artifacts must not be empty")
    if any(not _safe_artifact_name(name) for name in artifacts):
        raise ValueError("artifact names must be simple and non-empty")
    os.makedirs(directory,exist_ok=True); files={}
    for name in sorted(artifacts):
        filename=name+".json"; content=canonical_json(artifacts[name])
        with open(os.path.join(directory,filename),"wb") as f:f.write(content)
        files[filename]=hashlib.sha256(content).hexdigest()
    identity={"schema_version":SCHEMA_VERSION,"files":files}; manifest=dict(identity,identity_sha256=sha256(identity))
    with open(os.path.join(directory,"manifest.json"),"wb") as f:f.write(canonical_json(manifest))
    return manifest


def load_bundle(directory: str, max_manifest_bytes: int = 1024 * 1024,
                max_artifact_bytes: int = 128 * 1024 * 1024,
                max_total_artifact_bytes: int = 128 * 1024 * 1024,
                max_artifact_count: int = 64) -> Dict[str, Any]:
    try:
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_fd = os.open(os.path.realpath(directory), directory_flags)
        try:
            def read_regular(name, byte_limit, limit_error="artifact exceeds size limit"):
                flags = os.O_RDONLY
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                if hasattr(os, "O_NONBLOCK"):
                    flags |= os.O_NONBLOCK
                descriptor = os.open(name, flags, dir_fd=directory_fd)
                try:
                    file_stat = os.fstat(descriptor)
                    if not stat.S_ISREG(file_stat.st_mode):
                        raise ArtifactIntegrityError("artifact must be a regular file")
                    if file_stat.st_size > byte_limit:
                        raise ArtifactIntegrityError(limit_error)
                    chunks = []
                    total = 0
                    while True:
                        # Read one detection byte beyond the applicable limit,
                        # never a full unbounded chunk after a stale fstat.
                        chunk = os.read(descriptor, min(1024 * 1024,
                                                        byte_limit - total + 1))
                        if not chunk:
                            return b"".join(chunks)
                        total += len(chunk)
                        if total > byte_limit:
                            raise ArtifactIntegrityError(limit_error)
                        chunks.append(chunk)
                finally:
                    os.close(descriptor)

            if (isinstance(max_manifest_bytes, bool) or not isinstance(max_manifest_bytes, int) or
                    isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int) or
                    isinstance(max_total_artifact_bytes, bool) or not isinstance(max_total_artifact_bytes, int) or
                    isinstance(max_artifact_count, bool) or not isinstance(max_artifact_count, int) or
                    max_manifest_bytes < 0 or max_artifact_bytes < 0 or
                    max_total_artifact_bytes < 0 or max_artifact_count < 0):
                raise ArtifactIntegrityError("bundle size limits must be non-negative integers")
            manifest = json.loads(read_regular("manifest.json", max_manifest_bytes).decode("utf-8"))
            if (not isinstance(manifest, dict) or
                    set(manifest) != {"schema_version", "files", "identity_sha256"} or
                    isinstance(manifest.get("schema_version"), bool) or not isinstance(manifest.get("schema_version"), int) or
                    manifest.get("schema_version") != SCHEMA_VERSION or
                    not isinstance(manifest.get("files"), dict) or
                    not isinstance(manifest.get("identity_sha256"), str)):
                raise ArtifactIntegrityError("malformed manifest")
            files = manifest["files"]
            if (not files or any(not _safe_filename(name) or not isinstance(digest, str) or
                                 len(digest) != 64 or
                                 any(ch not in "0123456789abcdef" for ch in digest)
                                 for name, digest in files.items())):
                raise ArtifactIntegrityError("unsafe manifest files")
            if len(files) > max_artifact_count:
                raise ArtifactIntegrityError("manifest exceeds artifact count limit")
            identity = {"schema_version": manifest["schema_version"], "files": files}
            if manifest["identity_sha256"] != sha256(identity):
                raise ArtifactIntegrityError("manifest identity hash mismatch")
            result = {}; total_artifact_bytes = 0
            for name, digest in sorted(files.items()):
                remaining = max_total_artifact_bytes - total_artifact_bytes
                content = read_regular(name, min(max_artifact_bytes, remaining),
                                       "bundle exceeds total artifact size limit"
                                       if remaining < max_artifact_bytes else
                                       "artifact exceeds size limit")
                total_artifact_bytes += len(content)
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ArtifactIntegrityError("artifact hash mismatch: " + name)
                logical = name[:-5]
                if logical in result:
                    raise ArtifactIntegrityError("duplicate logical artifact name")
                result[logical] = json.loads(content.decode("utf-8"))
            return result
        finally:
            os.close(directory_fd)
    except ArtifactIntegrityError: raise
    except (OSError, ValueError, TypeError, AttributeError) as error: raise ArtifactIntegrityError("cannot load bundle: %s" % error)
