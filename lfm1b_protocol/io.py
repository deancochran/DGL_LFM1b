import csv
from typing import Iterator, Optional

from .models import Event


LFM_LISTENING_EVENT_COLUMNS = (
    "user_id", "artist_id", "album_id", "track_id", "timestamp")


def _optional_id(value: str) -> Optional[int]:
    value = value.strip()
    return None if value == "" else int(value)


def read_listening_events(path: str, has_header: bool = False) -> Iterator[Event]:
    """Stream tab-separated user, artist, album, track, timestamp rows.

    This is the raw LFM-1b listening-event order. Empty artist, album, or track
    IDs become ``None``. User IDs and timestamps are required integers.
    """
    with open(path, "r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        if has_header:
            header = next(reader, None)
            if tuple(header or ()) != LFM_LISTENING_EVENT_COLUMNS:
                raise ValueError("unexpected listening-event header")
        for source_row, row in enumerate(reader, 2 if has_header else 1):
            if len(row) != 5:
                raise ValueError("row %d has %d columns; expected 5" %
                                 (source_row, len(row)))
            yield Event(int(row[0]), _optional_id(row[1]), _optional_id(row[2]),
                        _optional_id(row[3]), int(row[4]), source_row)
