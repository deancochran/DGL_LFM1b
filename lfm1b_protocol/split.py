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
    raw_split: TemporalSplit
    known_positives: Mapping[str, Tuple[Tuple[int, int], ...]]
    window_positives: Mapping[str, Mapping[str, Tuple[Tuple[int, int], ...]]]
    catalogs: Mapping[str, Tuple[int, ...]]
    statistics: SplitStatistics


def _interaction(user_id: int, item_type: str, item_id: int,
                 timestamps: List[int]) -> PairInteraction:
    return PairInteraction(user_id, item_type, item_id, len(timestamps),
                           min(timestamps), max(timestamps))


def temporal_split(events: Iterable[Event], *, strategy: str,
                   validation_cutoff: Optional[int] = None,
                   test_cutoff: Optional[int] = None,
                   repeat_policy: str = "novel_only") -> ProtocolData:
    """Split each user's global timestamp groups, then project all item types.

    The latest timestamp group is test, the previous group validation, and all
    earlier groups train. Users with fewer than three distinct timestamps are
    excluded from all split windows but remain in catalogs/known positives.
    """
    if strategy not in ("per_user_last_timestamp_groups", "global_time_cutoffs"):
        raise ValueError("unsupported split strategy")
    if repeat_policy not in ("novel_only", "repeat_allowed"):
        raise ValueError("repeat_policy must be novel_only or repeat_allowed")
    if strategy == "global_time_cutoffs":
        if (isinstance(validation_cutoff, bool) or not isinstance(validation_cutoff, int) or
                isinstance(test_cutoff, bool) or not isinstance(test_cutoff, int) or
                validation_cutoff >= test_cutoff):
            raise ValueError("global cutoffs must be integers with validation_cutoff < test_cutoff")
    elif validation_cutoff is not None or test_cutoff is not None:
        raise ValueError("cutoffs are only valid for global_time_cutoffs")
    by_user = defaultdict(list)  # type: DefaultDict[int, List[Event]]
    known = {kind: set() for kind in ITEM_TYPES}  # type: Dict[str, Set[Tuple[int, int]]]
    catalogs = {kind: set() for kind in ITEM_TYPES}  # type: Dict[str, Set[int]]
    for event in events:
        if not isinstance(event, Event):
            raise ValueError("events must be Event instances")
        if (isinstance(event.user_id, bool) or not isinstance(event.user_id, int) or
                isinstance(event.timestamp, bool) or not isinstance(event.timestamp, int) or
                (event.source_row is not None and (isinstance(event.source_row, bool) or
                                                   not isinstance(event.source_row, int) or event.source_row <= 0)) or
                any(value is not None and (isinstance(value, bool) or not isinstance(value, int))
                    for value in (event.artist_id, event.album_id, event.track_id))):
            raise ValueError("event fields have invalid primitive types")
        by_user[event.user_id].append(event)
        for kind in ITEM_TYPES:
            item_id = getattr(event, kind + "_id")
            if item_id is not None:
                known[kind].add((event.user_id, item_id))
                catalogs[kind].add(item_id)

    windows = {"train": [], "validation": [], "test": []}  # type: Dict[str, List[PairInteraction]]
    raw_interactions = {"train": [], "validation": [], "test": []}
    raw_windows = {kind: {name: set() for name in windows} for kind in ITEM_TYPES}
    insufficient = []  # type: List[int]
    repeated_validation = 0
    repeated_test = 0
    for user_id in sorted(by_user):
        user_events = by_user[user_id]
        timestamps = sorted(set(event.timestamp for event in user_events))
        if strategy == "per_user_last_timestamp_groups" and len(timestamps) < 3:
            insufficient.append(user_id)
            continue
        if strategy == "per_user_last_timestamp_groups":
            validation_timestamp, test_timestamp = timestamps[-2], timestamps[-1]
            event_windows = {"train": [e for e in user_events if e.timestamp < validation_timestamp],
                             "validation": [e for e in user_events if e.timestamp == validation_timestamp],
                             "test": [e for e in user_events if e.timestamp == test_timestamp]}
        else:
            event_windows = {"train": [e for e in user_events if e.timestamp < validation_cutoff],
                             "validation": [e for e in user_events if validation_cutoff <= e.timestamp < test_cutoff],
                             "test": [e for e in user_events if e.timestamp >= test_cutoff]}
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
                    raw_windows[kind][window_name].add((user_id, item_id))
                    raw_interactions[window_name].append(
                        _interaction(user_id, kind, item_id, grouped[item_id]))
                    if repeat_policy == "novel_only" and window_name == "validation" and item_id in seen_train:
                        repeated_validation += 1
                        continue
                    if (repeat_policy == "novel_only" and window_name == "test" and
                            (item_id in seen_train or item_id in seen_validation)):
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
    raw_result = TemporalSplit(
        tuple(sorted(raw_interactions["train"], key=sort_key)),
        tuple(sorted(raw_interactions["validation"], key=sort_key)),
        tuple(sorted(raw_interactions["test"], key=sort_key)),
    )
    statistics = SplitStatistics(
        len(by_user), len(by_user) - len(insufficient), tuple(insufficient),
        repeated_validation, repeated_test,
    )
    frozen_known = {kind: tuple(sorted(values)) for kind, values in known.items()}
    frozen_catalogs = {kind: tuple(sorted(values)) for kind, values in catalogs.items()}
    frozen_windows = {kind: {name: tuple(sorted(values)) for name, values in by_window.items()}
                      for kind, by_window in raw_windows.items()}
    return ProtocolData(result, raw_result, frozen_known, frozen_windows, frozen_catalogs, statistics)
