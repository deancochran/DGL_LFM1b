import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple


@dataclass(frozen=True)
class MetricResult:
    k: int
    users: int
    recall: float
    ndcg: float
    mrr: float
    mean_average_precision: float
    catalog_coverage: float
    novelty: float


def evaluate_rankings(
        scores: Mapping[int, Sequence[Tuple[int, float]]],
        positives: Mapping[int, Iterable[int]], k: int,
        catalog: Iterable[int], training_popularity: Mapping[int, int],
        smoothing: float = 1.0) -> MetricResult:
    """Compute macro ranking metrics with score-descending/item-ID ties."""
    if k <= 0:
        raise ValueError("k must be positive")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    if not scores:
        raise ValueError("scores must contain at least one user")
    if set(scores) != set(positives):
        raise ValueError("scores and positives must contain the same users")
    catalog_set = set(catalog)
    if not catalog_set:
        raise ValueError("catalog must not be empty")
    total_popularity = sum(training_popularity.get(item, 0) for item in catalog_set)
    denominator = total_popularity + smoothing * len(catalog_set)
    totals = {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "map": 0.0}
    recommended = set()  # type: Set[int]
    novelty_values = []
    for user_id in sorted(scores):
        relevant = set(positives[user_id])
        if not relevant:
            raise ValueError("each user must have at least one positive")
        rows = list(scores[user_id])
        if any(not math.isfinite(score) for unused_item, score in rows):
            raise ValueError("scores must be finite")
        item_ids = [item for item, unused_score in rows]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("a user's scored item IDs must be unique")
        if not set(item_ids).issubset(catalog_set):
            raise ValueError("scored items must belong to the catalog")
        if not relevant.issubset(set(item_ids)):
            raise ValueError("all positives must have scores")
        ranked = sorted(rows, key=lambda row: (-row[1], row[0]))[:k]
        hits = [index + 1 for index, (item, unused_score) in enumerate(ranked)
                if item in relevant]
        totals["recall"] += len(hits) / float(len(relevant))
        dcg = sum(1.0 / math.log2(rank + 1.0) for rank in hits)
        ideal_hits = min(len(relevant), len(ranked))
        idcg = sum(1.0 / math.log2(rank + 1.0)
                   for rank in range(1, ideal_hits + 1))
        totals["ndcg"] += dcg / idcg if idcg else 0.0
        totals["mrr"] += 1.0 / hits[0] if hits else 0.0
        precision_sum = sum(
            hit_number / float(rank) for hit_number, rank in enumerate(hits, 1))
        totals["map"] += precision_sum / float(min(len(relevant), k))
        for item, unused_score in ranked:
            recommended.add(item)
            probability = (training_popularity.get(item, 0) + smoothing) / denominator
            novelty_values.append(-math.log2(probability))
    users = len(scores)
    novelty = sum(novelty_values) / len(novelty_values) if novelty_values else 0.0
    return MetricResult(k, users, totals["recall"] / users,
                        totals["ndcg"] / users, totals["mrr"] / users,
                        totals["map"] / users,
                        len(recommended) / float(len(catalog_set)), novelty)
