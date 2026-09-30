"""Adapter-neutral bounded baseline scoring over a validated protocol artifact."""

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import asdict

from lfm1b_protocol.artifacts import (_candidate_rows_for_validated_protocol,
                                       validate_protocol_artifact)
from lfm1b_protocol.canonical import sha256
from lfm1b_protocol.metrics import (aggregate_ranking_contributions,
                                    evaluate_ranking_contributions)

from .plan import (CANDIDATE_POLICIES, MODELS, SPLITS, TARGETS,
                   ComparisonError, MAX_SIGNED_INTEGER)


MAX_ENTITY_ID = MAX_SIGNED_INTEGER


def _candidate_upper_bound(protocol, target, split, candidate_policy):
    payload = protocol["targets"][target]
    if candidate_policy == "fixed_sampled":
        return sum(len(row["candidate_item_ids"])
                   for row in payload["candidates"][split])
    users = {row["user_id"] for row in payload["splits"][split]}
    return len(users) * len(payload["catalog"])


def _candidate_user_upper_bound(protocol, target, split, candidate_policy):
    payload = protocol["targets"][target]
    if candidate_policy == "fixed_sampled":
        return len(payload["candidates"][split])
    return len({row["user_id"] for row in payload["splits"][split]})


def _validate_bounded_ids(protocol, target):
    payload = protocol["targets"][target]
    values = list(payload["catalog"])
    for split in ("train", "validation", "test"):
        for row in payload["splits"][split]:
            values.extend((row["user_id"], row["item_id"]))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or
           value > MAX_ENTITY_ID for value in values):
        raise ComparisonError("comparison entity IDs must fit unsigned 63-bit integers")


def comparison_work_upper_bounds(protocol, target, split, candidate_policy, k_values):
    protocol = validate_protocol_artifact(protocol)
    return _comparison_work_upper_bounds_validated(
        protocol, target, split, candidate_policy, k_values)


def _comparison_work_upper_bounds_validated(protocol, target, split,
                                             candidate_policy, k_values):
    """Internal only: use after one strict artifact-boundary validation."""
    if (target not in TARGETS or split not in SPLITS or
            candidate_policy not in CANDIDATE_POLICIES or
            not isinstance(k_values, (tuple, list)) or not k_values or
            any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                or value > MAX_SIGNED_INTEGER for value in k_values)):
        raise ComparisonError("invalid comparison work-bound request")
    _validate_bounded_ids(protocol, target)
    candidate_scores = _candidate_upper_bound(
        protocol, target, split, candidate_policy)
    users = _candidate_user_upper_bound(protocol, target, split, candidate_policy)
    catalog_items = len(protocol["targets"][target]["catalog"])
    return {"candidate_scores": candidate_scores, "users": users,
            "contribution_rows": users * len(k_values),
            "ranking_items": candidate_scores * len(k_values),
            # Candidate scores bound computation.  Serialized contributions
            # contain only the top-K recommendation IDs and novelty values.
            "serialized_ranking_items": users * sum(
                min(k, catalog_items) for k in k_values)}


def _training_state(protocol, target, model, max_itemknn_pairs):
    train = protocol["targets"][target]["splits"]["train"]
    popularity = Counter()
    histories = defaultdict(set)
    for row in train:
        popularity[row["item_id"]] += row["play_count"]
        histories[row["user_id"]].add(row["item_id"])
    similarities = defaultdict(dict)
    pair_operations = 0
    if model == "itemknn":
        pair_operations = sum(len(items) * (len(items) - 1)
                              for items in histories.values())
        if pair_operations > max_itemknn_pairs:
            raise ComparisonError(
                "ItemKNN requires %d ordered co-occurrence pairs; limit is %d" %
                (pair_operations, max_itemknn_pairs))
        counts = Counter()
        cooccurrences = Counter()
        for user_id in sorted(histories):
            items = sorted(histories[user_id])
            for item in items:
                counts[item] += 1
            for left in items:
                for right in items:
                    if left != right:
                        cooccurrences[(left, right)] += 1
        for pair in sorted(cooccurrences):
            left, right = pair
            similarities[left][right] = (
                cooccurrences[pair] / math.sqrt(counts[left] * counts[right]))
    return popularity, histories, similarities, pair_operations


def _random_priority(seed, target, split, user_id, item_id):
    payload = "protocol-comparison:sha256-priority:v1:%s:%s:%s:%s:%s" % (
        seed, target, split, user_id, item_id)
    return int(hashlib.sha256(payload.encode("ascii")).hexdigest(), 16)


def evaluate_protocol_baseline(protocol, target, split, candidate_policy, model,
                               k_values, random_seed, neighbor_limit,
                               max_candidate_scores, max_itemknn_pairs,
                               max_contribution_rows, max_ranking_items):
    if (target not in TARGETS or split not in SPLITS or
            candidate_policy not in CANDIDATE_POLICIES or model not in MODELS):
        raise ComparisonError("invalid protocol comparison request")
    if (not isinstance(k_values, (tuple, list)) or not k_values or
            any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in k_values)):
        raise ComparisonError("K values must be positive integers")
    if (isinstance(max_candidate_scores, bool) or
            not isinstance(max_candidate_scores, int) or max_candidate_scores <= 0 or
            isinstance(max_itemknn_pairs, bool) or
            not isinstance(max_itemknn_pairs, int) or max_itemknn_pairs <= 0):
        raise ComparisonError("comparison limits must be positive integers")
    if (isinstance(max_contribution_rows, bool) or
            not isinstance(max_contribution_rows, int) or max_contribution_rows <= 0 or
            isinstance(max_ranking_items, bool) or
            not isinstance(max_ranking_items, int) or max_ranking_items <= 0):
        raise ComparisonError("comparison output/work limits must be positive integers")
    if model == "random" and (isinstance(random_seed, bool) or
                              not isinstance(random_seed, int) or
                              random_seed < -MAX_SIGNED_INTEGER or
                              random_seed > MAX_SIGNED_INTEGER):
        raise ComparisonError("random seed must fit a signed 63-bit integer")
    if model == "itemknn" and (isinstance(neighbor_limit, bool) or
                               not isinstance(neighbor_limit, int) or
                               neighbor_limit <= 0 or
                               neighbor_limit > MAX_SIGNED_INTEGER):
        raise ComparisonError("neighbor limit must be a positive signed 63-bit integer")
    protocol = validate_protocol_artifact(protocol)
    return _evaluate_protocol_baseline_validated(
        protocol, target, split, candidate_policy, model, k_values, random_seed,
        neighbor_limit, max_candidate_scores, max_itemknn_pairs,
        max_contribution_rows, max_ranking_items)


def _evaluate_protocol_baseline_validated(
        protocol, target, split, candidate_policy, model, k_values, random_seed,
        neighbor_limit, max_candidate_scores, max_itemknn_pairs,
        max_contribution_rows, max_ranking_items):
    """Internal only: protocol must already have passed strict validation."""
    bounds = _comparison_work_upper_bounds_validated(
        protocol, target, split, candidate_policy, k_values)
    upper_bound = bounds["candidate_scores"]
    if upper_bound > max_candidate_scores:
        raise ComparisonError(
            "candidate scoring upper bound is %d; limit is %d" %
            (upper_bound, max_candidate_scores))
    contribution_upper_bound = bounds["contribution_rows"]
    if contribution_upper_bound > max_contribution_rows:
        raise ComparisonError(
            "contribution-row upper bound is %d; limit is %d" %
            (contribution_upper_bound, max_contribution_rows))
    ranking_upper_bound = bounds["ranking_items"]
    if ranking_upper_bound > max_ranking_items:
        raise ComparisonError(
            "ranking-item upper bound is %d; limit is %d" %
            (ranking_upper_bound, max_ranking_items))
    rows = _candidate_rows_for_validated_protocol(
        protocol, target, split, candidate_policy)
    if not rows:
        raise ComparisonError("comparison request has no eligible candidate rows")
    candidate_scores = sum(len(row["candidate_item_ids"]) for row in rows)
    if candidate_scores > max_candidate_scores:
        raise ComparisonError("candidate score count exceeds the comparison limit")
    contribution_rows = len(rows) * len(k_values)
    ranking_items = candidate_scores * len(k_values)
    popularity, histories, similarities, pair_operations = _training_state(
        protocol, target, model, max_itemknn_pairs)
    scores = {}
    for row in rows:
        user_id = row["user_id"]
        candidates = row["candidate_item_ids"]
        if model == "random":
            def score(item_id):
                return _random_priority(random_seed, target, split, user_id, item_id)
        elif model == "popularity":
            def score(item_id):
                return popularity[item_id]
        else:
            neighbors = defaultdict(list)
            for prior in sorted(histories[user_id]):
                for item_id, similarity in sorted(similarities[prior].items()):
                    neighbors[item_id].append((similarity, prior))

            def score(item_id):
                ranked = sorted(neighbors.get(item_id, ()),
                                key=lambda value: (-value[0], value[1]))
                return sum(similarity for similarity, unused_prior
                           in ranked[:neighbor_limit])
        scores[user_id] = [(item_id, score(item_id)) for item_id in candidates]
    positives = {row["user_id"]: tuple(row["positive_item_ids"]) for row in rows}
    expected = {row["user_id"]: tuple(row["candidate_item_ids"]) for row in rows}
    catalog = protocol["targets"][target]["catalog"]
    if model == "random":
        model_config = {"algorithm": "protocol_comparison_sha256_priority_v1",
                        "random_seed": random_seed}
    elif model == "popularity":
        model_config = {"algorithm": "train_play_count_popularity_v1"}
    else:
        model_config = {"algorithm": "binary_cosine_itemknn_v1",
                        "neighbor_limit": neighbor_limit,
                        "ordered_cooccurrence_pairs": pair_operations}
    evaluations = []
    for k in k_values:
        contributions = evaluate_ranking_contributions(
            scores, positives, expected, k, catalog, popularity)
        metric = aggregate_ranking_contributions(
            contributions, k, catalog, popularity)
        evaluations.append({"k": k, "metrics": asdict(metric),
                            "contributions": [asdict(value)
                                              for value in contributions]})
    return {
        "candidate_rows_hash": sha256({"target": target, "split": split,
                                       "candidate_policy": candidate_policy,
                                       "rows": rows}),
        "candidate_scores": candidate_scores,
        "catalog_items": len(catalog),
        "itemknn_pairs": pair_operations,
        "contribution_rows": contribution_rows,
        "ranking_items": ranking_items,
        "model_config": model_config,
        "evaluations": evaluations,
    }
