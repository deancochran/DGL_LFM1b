import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple


MAX_SIGNED_INTEGER = (1 << 63) - 1


def _validated_smoothing(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("smoothing must be positive and finite")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError("smoothing must be positive and finite")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("smoothing must be positive and finite")
    return value


def _novelty_denominator(training_popularity, catalog_set, smoothing):
    """Convert arbitrary-size counts to a finite floating novelty domain."""
    total_popularity = sum(training_popularity.get(item, 0) for item in catalog_set)
    try:
        denominator = float(total_popularity) + smoothing * len(catalog_set)
    except OverflowError:
        raise ValueError("training popularity total must produce a finite novelty denominator")
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("training popularity total must produce a finite novelty denominator")
    return denominator


def _item_novelty(popularity, smoothing, denominator):
    try:
        probability = (float(popularity) + smoothing) / denominator
    except OverflowError:
        raise ValueError("training popularity must produce finite novelty values")
    if not math.isfinite(probability) or probability <= 0:
        raise ValueError("training popularity must produce finite novelty values")
    return -math.log2(probability)


@dataclass(frozen=True)
class MetricResult:
    k: int
    users: int
    recall: float
    ndcg: float
    mrr: float
    mean_average_precision: float
    precision: float
    hit_rate: float
    catalog_coverage: float
    novelty: float


@dataclass(frozen=True)
class RankingContribution:
    user_id: int
    k: int
    relevant_items: int
    candidate_items: int
    recommended_items: int
    hit_items: int
    recall: float
    ndcg: float
    mrr: float
    mean_average_precision: float
    precision: float
    hit_rate: float
    novelty: float
    hit_ranks: Tuple[int, ...]
    recommended_item_ids: Tuple[int, ...]
    recommended_item_novelties: Tuple[float, ...]


def ranking_metric_values(relevant_items: int, recommended_items: int, k: int,
                          hit_ranks: Sequence[int],
                          recommended_item_novelties: Sequence[float]) -> Dict[str, float]:
    """Return the one canonical arithmetic contract for a ranked user row.

    Report validators deliberately call this same function.  Keep the operation
    order here stable: report hashes are reproducibility identities, not merely
    approximate displays of metrics.
    """
    hits = len(hit_ranks)
    recall = hits / float(relevant_items)
    precision = hits / float(k)
    dcg = math.fsum(1.0 / math.log2(rank + 1.0) for rank in hit_ranks)
    ideal_hits = min(relevant_items, recommended_items)
    idcg = math.fsum(1.0 / math.log2(rank + 1.0)
                     for rank in range(1, ideal_hits + 1))
    precision_sum = math.fsum(hit_number / float(rank)
                              for hit_number, rank in enumerate(hit_ranks, 1))
    return {
        "recall": recall,
        "ndcg": dcg / idcg if idcg else 0.0,
        "mrr": 1.0 / hit_ranks[0] if hit_ranks else 0.0,
        "mean_average_precision": precision_sum / float(min(relevant_items, k)),
        "precision": precision,
        "hit_rate": 1.0 if hit_ranks else 0.0,
        "novelty": (math.fsum(recommended_item_novelties) /
                    len(recommended_item_novelties)
                    if recommended_item_novelties else 0.0),
    }


def aggregate_metric_values(contributions: Iterable[RankingContribution],
                            catalog_items: int) -> Dict[str, float]:
    """Aggregate ranked-user rows with the canonical macro/weighted contract."""
    contributions = tuple(contributions)
    users = len(contributions)
    recommended = {item for value in contributions
                   for item in value.recommended_item_ids}
    novelty_values = [novelty for value in contributions
                      for novelty in value.recommended_item_novelties]
    return {
        "recall": math.fsum(value.recall for value in contributions) / users,
        "ndcg": math.fsum(value.ndcg for value in contributions) / users,
        "mrr": math.fsum(value.mrr for value in contributions) / users,
        "mean_average_precision": math.fsum(value.mean_average_precision for value in contributions) / users,
        "precision": math.fsum(value.precision for value in contributions) / users,
        "hit_rate": math.fsum(value.hit_rate for value in contributions) / users,
        "catalog_coverage": len(recommended) / float(catalog_items),
        "novelty": (math.fsum(novelty_values) / len(novelty_values)
                    if novelty_values else 0.0),
    }


def paired_metric_values(baseline: Sequence[float],
                         comparison: Sequence[float]) -> Dict[str, float]:
    """Return canonical paired means and deltas in common-user order."""
    if not baseline or len(baseline) != len(comparison):
        raise ValueError("paired metric values must be nonempty and aligned")
    deltas = tuple(right - left for left, right in zip(baseline, comparison))
    return {"baseline_mean": math.fsum(baseline) / len(baseline),
            "comparison_mean": math.fsum(comparison) / len(comparison),
            "mean_delta": math.fsum(deltas) / len(deltas)}


def evaluate_ranking_contributions(
        scores: Mapping[int, Sequence[Tuple[int, float]]],
        positives: Mapping[int, Iterable[int]], expected_candidates: Mapping[int, Iterable[int]], k: int,
        catalog: Iterable[int], training_popularity: Mapping[int, int],
        smoothing: float = 1.0) -> Tuple[RankingContribution, ...]:
    """Return deterministic per-user contributions for paired comparisons."""
    if (isinstance(k, bool) or not isinstance(k, int) or k <= 0 or
            k > MAX_SIGNED_INTEGER):
        raise ValueError("k must be a positive signed 63-bit integer")
    smoothing = _validated_smoothing(smoothing)
    if not scores:
        raise ValueError("scores must contain at least one user")
    for user_id in scores:
        if isinstance(user_id, bool) or not isinstance(user_id, int):
            raise ValueError("user IDs must be non-boolean integers")
    if set(scores) != set(positives) or set(scores) != set(expected_candidates):
        raise ValueError("scores, positives, and expected candidates must contain the same users")
    catalog_set = set(catalog)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in catalog_set):
        raise ValueError("catalog item IDs must be integers")
    if not catalog_set:
        raise ValueError("catalog must not be empty")
    for item in catalog_set:
        popularity = training_popularity.get(item, 0)
        if isinstance(popularity, bool) or not isinstance(popularity, int) or popularity < 0:
            raise ValueError("training popularity must be non-negative integers")
    denominator = _novelty_denominator(training_popularity, catalog_set, smoothing)
    contributions = []
    for user_id in sorted(scores):
        relevant = set(positives[user_id])
        expected = list(expected_candidates[user_id])
        if (len(expected) != len(set(expected)) or
                any(isinstance(item, bool) or not isinstance(item, int)
                    for item in expected) or
                not set(expected).issubset(catalog_set)):
            raise ValueError("expected candidates must be unique catalog items")
        if (any(isinstance(item, bool) or not isinstance(item, int)
                for item in relevant) or
                not relevant.issubset(set(expected))):
            raise ValueError("positives must be expected catalog items")
        if not relevant:
            raise ValueError("each user must have at least one positive")
        rows = list(scores[user_id])
        if any(isinstance(score, bool) or not isinstance(score, (int, float)) or
               not math.isfinite(score) for unused_item, score in rows):
            raise ValueError("scores must be finite")
        item_ids = [item for item, unused_score in rows]
        if any(isinstance(item, bool) or not isinstance(item, int)
               for item in item_ids):
            raise ValueError("scored item IDs must be integers")
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("a user's scored item IDs must be unique")
        if not set(item_ids).issubset(catalog_set):
            raise ValueError("scored items must belong to the catalog")
        if set(item_ids) != set(expected):
            raise ValueError("scored item IDs must exactly equal expected candidates")
        if not relevant.issubset(set(item_ids)):
            raise ValueError("all positives must have scores")
        ranked = sorted(rows, key=lambda row: (-row[1], row[0]))[:k]
        hits = [index + 1 for index, (item, unused_score) in enumerate(ranked)
                if item in relevant]
        novelty_values = []
        for item, unused_score in ranked:
            novelty_values.append(_item_novelty(training_popularity.get(item, 0),
                                                smoothing, denominator))
        metrics = ranking_metric_values(len(relevant), len(ranked), k, hits,
                                        novelty_values)
        contributions.append(RankingContribution(
            user_id=user_id, k=k, relevant_items=len(relevant),
            candidate_items=len(expected), recommended_items=len(ranked),
            hit_items=len(hits), **metrics,
            hit_ranks=tuple(hits),
            recommended_item_ids=tuple(item for item, unused_score in ranked),
            recommended_item_novelties=tuple(novelty_values)))
    return tuple(contributions)


def aggregate_ranking_contributions(
        contributions: Iterable[RankingContribution], k: int, catalog: Iterable[int],
        training_popularity: Mapping[int, int], smoothing: float = 1.0) -> MetricResult:
    """Aggregate validated per-user contributions without reranking candidates."""
    if (isinstance(k, bool) or not isinstance(k, int) or k <= 0 or
            k > MAX_SIGNED_INTEGER):
        raise ValueError("k must be a positive signed 63-bit integer")
    smoothing = _validated_smoothing(smoothing)
    contributions = tuple(contributions)
    if not contributions or any(not isinstance(value, RankingContribution)
                                for value in contributions):
        raise ValueError("contributions must contain ranking contributions")
    if tuple(value.user_id for value in contributions) != tuple(sorted(
            {value.user_id for value in contributions})):
        raise ValueError("contribution users must be unique and sorted")
    if any(value.k != k for value in contributions):
        raise ValueError("contribution K must match the aggregate K")
    catalog_values = tuple(catalog)
    catalog_set = set(catalog_values)
    if not catalog_set:
        raise ValueError("catalog must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in catalog_set):
        raise ValueError("catalog item IDs must be integers")
    if any(not set(value.recommended_item_ids).issubset(catalog_set)
           for value in contributions):
        raise ValueError("recommended contribution items must belong to the catalog")
    for item in catalog_set:
        popularity = training_popularity.get(item, 0)
        if isinstance(popularity, bool) or not isinstance(popularity, int) or popularity < 0:
            raise ValueError("training popularity must be non-negative integers")
    denominator = _novelty_denominator(training_popularity, catalog_set, smoothing)
    users = len(contributions)
    # Recompute novelty from the trusted catalog/popularity inputs rather than
    # blindly trusting a caller-constructed contribution object.
    for contribution in contributions:
        expected_novelties = tuple(_item_novelty(training_popularity.get(item, 0),
                                                  smoothing, denominator)
                                   for item in contribution.recommended_item_ids)
        if contribution.recommended_item_novelties != expected_novelties:
            raise ValueError("contribution novelty does not match training popularity")
    values = aggregate_metric_values(contributions, len(catalog_set))
    return MetricResult(k=k, users=users, **values)


def evaluate_rankings(
        scores: Mapping[int, Sequence[Tuple[int, float]]],
        positives: Mapping[int, Iterable[int]], expected_candidates: Mapping[int, Iterable[int]], k: int,
        catalog: Iterable[int], training_popularity: Mapping[int, int],
        smoothing: float = 1.0) -> MetricResult:
    """Compute macro ranking metrics with score-descending/item-ID ties."""
    catalog_values = tuple(catalog)
    contributions = evaluate_ranking_contributions(
        scores, positives, expected_candidates, k, catalog_values,
        training_popularity, smoothing)
    return aggregate_ranking_contributions(
        contributions, k, catalog_values, training_popularity, smoothing)
