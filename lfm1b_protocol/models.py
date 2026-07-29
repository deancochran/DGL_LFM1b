from dataclasses import dataclass
from typing import Optional, Tuple


ITEM_TYPES = ("artist", "album", "track")


@dataclass(frozen=True)
class Event:
    user_id: int
    artist_id: Optional[int]
    album_id: Optional[int]
    track_id: Optional[int]
    timestamp: int
    source_row: Optional[int] = None


@dataclass(frozen=True)
class PairInteraction:
    user_id: int
    item_type: str
    item_id: int
    play_count: int
    first_timestamp: int
    last_timestamp: int


@dataclass(frozen=True)
class TemporalSplit:
    train: Tuple[PairInteraction, ...]
    validation: Tuple[PairInteraction, ...]
    test: Tuple[PairInteraction, ...]


@dataclass(frozen=True)
class CandidateSet:
    user_id: int
    item_type: str
    split: str
    policy: str
    items: Tuple[int, ...]
    positives: Tuple[int, ...]
    candidate_hash: str
