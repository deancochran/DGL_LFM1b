from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from .models import Event, ITEM_TYPES, PairInteraction, TemporalSplit


@dataclass(frozen=True)
class SplitStatistics:
    total_users: int
    eligible_users: int
    insufficient_users: Tuple[int, ...]
    repeated_validation_pairs_excluded: int
    repeated_test_pairs_excluded: int


@dataclass(frozen=True)
class ProtocolData:
    split: TemporalSplit
    known_positives: Mapping[str, Tuple[Tuple[int, int], ...]]
    catalogs: Mapping[str, Tuple[int, ...]]
    statistics: SplitStatistics


def _interaction(user_id: int, item_type: str, item_id: int,
                 timestamps: List[int]) -> PairInteraction:
    return PairInteraction(user_id, item_type, item_id, len(timestamps),
                           min(timestamps), max(timestamps))


def temporal_split(events: Iterable[Event]) -> ProtocolData:
    """Split each user's global timestamp groups, then project all item types.

    The latest timestamp group is test, the previous group validation, and all
    earlier groups train. Users with fewer than three distinct timestamps are
    excluded from all split windows but remain in catalogs/known positives.
    """
    by_user = defaultdict(list)  # type: DefaultDict[int, List[Event]]
    known = {kind: set() for kind in ITEM_TYPES}  # type: Dict[str, Set[Tuple[int, int]]]
    catalogs = {kind: set() for kind in ITEM_TYPES}  # type: Dict[str, Set[int]]
    for event in events:
        by_user[event.user_id].append(event)
        for kind in ITEM_TYPES:
            item_id = getattr(event, kind + "_id")
            if item_id is not None:
                known[kind].add((event.user_id, item_id))
                catalogs[kind].add(item_id)

    windows = {"train": [], "validation": [], "test": []}  # type: Dict[str, List[PairInteraction]]
    insufficient = []  # type: List[int]
    repeated_validation = 0
    repeated_test = 0
    for user_id in sorted(by_user):
        user_events = by_user[user_id]
        timestamps = sorted(set(event.timestamp for event in user_events))
        if len(timestamps) < 3:
            insufficient.append(user_id)
            continue
        validation_timestamp, test_timestamp = timestamps[-2], timestamps[-1]
        event_windows = {
            "train": [event for event in user_events if event.timestamp < validation_timestamp],
            "validation": [event for event in user_events if event.timestamp == validation_timestamp],
            "test": [event for event in user_events if event.timestamp == test_timestamp],
        }
        for kind in ITEM_TYPES:
            seen_train = set(
                getattr(event, kind + "_id") for event in event_windows["train"]
                if getattr(event, kind + "_id") is not None
            )
            seen_validation = set()  # type: Set[int]
            for window_name in ("train", "validation", "test"):
                grouped = defaultdict(list)  # type: DefaultDict[int, List[int]]
                for event in event_windows[window_name]:
                    item_id = getattr(event, kind + "_id")
                    if item_id is not None:
                        grouped[item_id].append(event.timestamp)
                for item_id in sorted(grouped):
                    if window_name == "validation" and item_id in seen_train:
                        repeated_validation += 1
                        continue
                    if window_name == "test" and (item_id in seen_train or item_id in seen_validation):
                        repeated_test += 1
                        continue
                    windows[window_name].append(
                        _interaction(user_id, kind, item_id, grouped[item_id]))
                if window_name == "validation":
                    seen_validation.update(grouped)

    sort_key = lambda value: (value.user_id, value.item_type, value.item_id)
    result = TemporalSplit(
        tuple(sorted(windows["train"], key=sort_key)),
        tuple(sorted(windows["validation"], key=sort_key)),
        tuple(sorted(windows["test"], key=sort_key)),
    )
    statistics = SplitStatistics(
        len(by_user), len(by_user) - len(insufficient), tuple(insufficient),
        repeated_validation, repeated_test,
    )
    frozen_known = {kind: tuple(sorted(values)) for kind, values in known.items()}
    frozen_catalogs = {kind: tuple(sorted(values)) for kind, values in catalogs.items()}
    return ProtocolData(result, frozen_known, frozen_catalogs, statistics)
