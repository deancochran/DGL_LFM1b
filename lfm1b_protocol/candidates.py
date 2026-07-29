import hashlib
import random
from typing import Iterable, Mapping, Optional, Sequence, Set, Tuple

from .canonical import sha256
from .models import CandidateSet


def candidate_digest(items: Iterable[int]) -> str:
    return sha256(tuple(sorted(items)))


def _seed(seed: int, user_id: int, item_type: str, split: str) -> int:
    material = "%s:%s:%s:%s" % (seed, user_id, item_type, split)
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def build_candidates(user_id: int, item_type: str, split: str,
                     positives: Iterable[int], catalog: Iterable[int],
                     known_positives: Iterable[Tuple[int, int]],
                     policy: str = "full_catalog", negatives: int = 100,
                     seed: int = 0,
                     known_positive_index: Optional[Mapping[int, Set[int]]] = None,
                     sorted_catalog: Optional[Sequence[int]] = None,
                     catalog_index: Optional[Set[int]] = None) -> CandidateSet:
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    positive_set = set(positives)  # type: Set[int]
    if not positive_set:
        raise ValueError("candidate set requires at least one positive")
    catalog_items = (tuple(sorted_catalog) if sorted_catalog is not None
                     else tuple(sorted(set(catalog))))
    catalog_set = set(catalog_items) if catalog_index is None else catalog_index
    if not positive_set.issubset(catalog_set):
        raise ValueError("positives must be in the catalog")
    if known_positive_index is None:
        user_known = {item for user, item in known_positives if user == user_id}
    else:
        user_known = set(known_positive_index.get(user_id, set()))
    # Every known positive is ineligible as a negative; current positives are
    # added back only after sampling.
    if policy == "full_catalog":
        sampled = [item for item in catalog_items if item not in user_known]
    elif policy == "fixed_sampled":
        if negatives < 0:
            raise ValueError("negatives must be non-negative")
        rng = random.Random(_seed(seed, user_id, item_type, split))
        known_in_catalog = len(user_known.intersection(catalog_set))
        eligible_count = len(catalog_items) - known_in_catalog
        sample_count = min(negatives, eligible_count)
        if (eligible_count * 2 < len(catalog_items) or
                sample_count * 3 > eligible_count):
            eligible = [item for item in catalog_items if item not in user_known]
            sampled = rng.sample(eligible, sample_count)
        else:
            selected = set()  # type: Set[int]
            while len(selected) < sample_count:
                item = catalog_items[rng.randrange(len(catalog_items))]
                if item not in user_known:
                    selected.add(item)
            sampled = list(selected)
    else:
        raise ValueError("policy must be full_catalog or fixed_sampled")
    items = tuple(sorted(positive_set.union(sampled)))
    return CandidateSet(user_id, item_type, split, policy, items,
                        tuple(sorted(positive_set)), candidate_digest(items))
