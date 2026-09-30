import hashlib
import heapq
from typing import Iterable, Mapping, Optional, Sequence, Set, Tuple

from .canonical import sha256
from .models import CandidateSet


def candidate_digest(items: Iterable[int]) -> str:
    return sha256(tuple(sorted(items)))


SAMPLER_ALGORITHM = "sha256_priority_v1"


def _priority(seed: int, user_id: int, item_type: str, split: str, item_id: int) -> bytes:
    """A stable random-oracle priority; lowest priorities form a uniform sample."""
    material = "%s:%s:%s:%s:%s" % (seed, user_id, item_type, split, item_id)
    return hashlib.sha256(material.encode("utf-8")).digest()


def _integer(value: object, name: str, minimum: object = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be a non-boolean integer" % name)
    if minimum is not None and value < minimum:
        raise ValueError("%s is below its minimum" % name)
    return value


def build_candidates(user_id: int, item_type: str, split: str,
                     positives: Iterable[int], catalog: Iterable[int],
                      exclusion_pairs: Iterable[Tuple[int, int]],
                     policy: str = "full_catalog", negatives: int = 100,
                     seed: int = 0,
                     known_positive_index: Optional[Mapping[int, Set[int]]] = None,
                     sorted_catalog: Optional[Sequence[int]] = None,
                     catalog_index: Optional[Set[int]] = None) -> CandidateSet:
    _integer(user_id, "user_id")
    _integer(seed, "seed")
    _integer(negatives, "negatives", 0)
    if item_type not in ("artist", "album", "track"):
        raise ValueError("invalid item type")
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    positive_set = set(positives)  # type: Set[int]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in positive_set):
        raise ValueError("positives must be integer IDs")
    if not positive_set:
        raise ValueError("candidate set requires at least one positive")
    supplied_catalog = tuple(catalog)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in supplied_catalog):
        raise ValueError("catalog must contain integer IDs")
    canonical_catalog = tuple(sorted(set(supplied_catalog)))
    if sorted_catalog is not None and tuple(sorted_catalog) != canonical_catalog:
        raise ValueError("sorted_catalog must exactly match the canonical catalog")
    catalog_items = canonical_catalog
    if catalog_index is not None and set(catalog_index) != set(catalog_items):
        raise ValueError("catalog_index must exactly match the catalog")
    catalog_set = set(catalog_items)
    if not positive_set.issubset(catalog_set):
        raise ValueError("positives must be in the catalog")
    authoritative = {item for user, item in exclusion_pairs if user == user_id}
    if known_positive_index is not None and set(known_positive_index.get(user_id, set())) != authoritative:
        raise ValueError("known_positive_index contradicts exclusion_pairs")
    return _build_candidates_indexed(user_id, item_type, split, positive_set, catalog_items,
                                     policy, negatives, seed, authoritative)


def _build_candidates_indexed(user_id: int, item_type: str, split: str, positives: Set[int],
                              catalog_items: Sequence[int], policy: str, negatives: int,
                              seed: int, user_exclusions: Set[int]) -> CandidateSet:
    """Internal optimized path; callers must have derived exclusions authoritatively."""
    positive_set = set(positives)
    catalog_set = set(catalog_items)
    user_known = set(user_exclusions)
    # Direct positives must never become negatives, even if a caller supplied an
    # incomplete horizon snapshot.
    user_known.update(positive_set)
    if policy == "full_catalog":
        sampled = [item for item in catalog_items if item not in user_known]
    elif policy == "fixed_sampled":
        known_in_catalog = len(user_known.intersection(catalog_set))
        eligible_count = len(catalog_items) - known_in_catalog
        sample_count = min(negatives, eligible_count)
        eligible = [item for item in catalog_items if item not in user_known]
        if sample_count == 0:
            sampled = []
        elif sample_count == eligible_count:
            sampled = eligible
        else:
            sampled = [item for unused_priority, item in heapq.nsmallest(
                sample_count,
                ((_priority(seed, user_id, item_type, split, item), item)
                 for item in eligible))]
    else:
        raise ValueError("policy must be full_catalog or fixed_sampled")
    items = tuple(sorted(positive_set.union(sampled)))
    return CandidateSet(user_id, item_type, split, policy, items,
                        tuple(sorted(positive_set)), candidate_digest(items))
