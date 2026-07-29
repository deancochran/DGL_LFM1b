from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple, Union

from .models import PairInteraction


RELATIONS = {
    ("user", "listened_to_artist", "artist"): ("artist", False),
    ("artist", "artist_listened_by", "user"): ("artist", True),
    ("user", "listened_to_album", "album"): ("album", False),
    ("album", "album_listened_by", "user"): ("album", True),
    ("user", "listened_to_track", "track"): ("track", False),
    ("track", "track_listened_by", "user"): ("track", True),
}


@dataclass(frozen=True)
class LeakageFinding:
    relation: str
    user_id: int
    item_type: str
    item_id: int


@dataclass(frozen=True)
class LeakageReport:
    clean: bool
    findings: Tuple[LeakageFinding, ...]


def _relation_key(key: Union[str, Tuple[str, str, str]]) -> Tuple[str, str, str]:
    if isinstance(key, tuple) and len(key) == 3:
        return key
    matches = [canonical for canonical in RELATIONS if canonical[1] == key]
    if len(matches) != 1:
        raise ValueError("unknown or ambiguous target relation: %r" % (key,))
    return matches[0]


def verify_endpoint_mappings(
        edge_mappings: Mapping[Union[str, Tuple[str, str, str]], Iterable[Tuple[int, int]]],
        heldout: Iterable[PairInteraction]) -> LeakageReport:
    """Check plain endpoint pairs; no DGL object or import is required."""
    heldout_pairs = {(row.item_type, row.user_id, row.item_id) for row in heldout}
    findings = []
    for supplied_key, endpoints in edge_mappings.items():
        canonical = _relation_key(supplied_key)
        item_type, reverse = RELATIONS.get(canonical, (None, False))
        if item_type is None:
            raise ValueError("unknown target relation: %r" % (supplied_key,))
        for source, destination in endpoints:
            user_id, item_id = ((destination, source) if reverse
                                else (source, destination))
            if (item_type, user_id, item_id) in heldout_pairs:
                findings.append(LeakageFinding(canonical[1], user_id,
                                               item_type, item_id))
    ordered = tuple(sorted(set(findings), key=lambda row: (
        row.relation, row.user_id, row.item_type, row.item_id)))
    return LeakageReport(not ordered, ordered)
