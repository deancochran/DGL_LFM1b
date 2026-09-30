import csv
import hashlib
import io
import os
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
    if not isinstance(has_header, bool):
        raise ValueError("has_header must be boolean")
    return _read_listening_events(path, has_header)


def _read_listening_events(path: str, has_header: bool) -> Iterator[Event]:
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


class _HashingReader(io.RawIOBase):
    def __init__(self, source, digest, max_bytes=None):
        self.source, self.digest, self.size = source, digest, 0
        self.max_bytes = max_bytes
    def readable(self): return True
    def readinto(self, buffer):
        amount = len(buffer)
        if self.max_bytes is not None:
            amount = min(amount, self.max_bytes - self.size + 1)
        data = self.source.read(amount)
        if not data: return 0
        if self.max_bytes is not None and self.size + len(data) > self.max_bytes:
            raise ValueError("listening-event input exceeds its byte limit")
        buffer[:len(data)] = data
        self.digest.update(data); self.size += len(data)
        return len(data)


class ListeningEventProvenanceStream(object):
    """One-shot parser stream whose provenance exists only after success."""
    def __init__(self, path: str, has_header: bool, max_bytes=None):
        if not isinstance(has_header, bool):
            raise ValueError("has_header must be boolean")
        if (max_bytes is not None and
                (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or
                 max_bytes <= 0)):
            raise ValueError("max_bytes must be a positive integer or None")
        self.path, self.has_header = os.path.abspath(path), has_header
        self.max_bytes = max_bytes
        self._started = self._completed = self._failed = False
        self._record = None

    def __iter__(self):
        if self._started:
            raise ValueError("parser-bound stream is one-shot and cannot be reused")
        self._started = True
        digest = hashlib.sha256(); rows_parsed = 0
        try:
            with open(self.path, "rb") as binary:
                raw = _HashingReader(binary, digest, self.max_bytes)
                with io.TextIOWrapper(io.BufferedReader(raw), encoding="utf-8", newline="") as source:
                    reader = csv.reader(source, delimiter="\t")
                    if self.has_header:
                        header = next(reader, None)
                        if tuple(header or ()) != LFM_LISTENING_EVENT_COLUMNS:
                            raise ValueError("unexpected listening-event header")
                    for source_row, row in enumerate(reader, 2 if self.has_header else 1):
                        if len(row) != 5:
                            raise ValueError("row %d has %d columns; expected 5" % (source_row, len(row)))
                        rows_parsed += 1
                        yield Event(int(row[0]), _optional_id(row[1]), _optional_id(row[2]),
                                    _optional_id(row[3]), int(row[4]), source_row)
            self._record = {"provenance_binding": "parser_bound", "source_kind": "listening_events",
                            "parser_version": "lfm_listening_events_tsv_v1", "bytes": raw.size,
                            "sha256": digest.hexdigest(), "has_header": self.has_header,
                            "column_order": LFM_LISTENING_EVENT_COLUMNS, "parsed_row_count": rows_parsed}
            self._completed = True
        except Exception:
            self._failed = True
            raise

    def completed_record(self):
        if not self._completed:
            raise ValueError("parser-bound stream was not fully consumed successfully")
        return dict(self._record)


def read_listening_events_with_provenance(
        path: str, has_header: bool, max_bytes=None) -> ListeningEventProvenanceStream:
    return ListeningEventProvenanceStream(path, has_header, max_bytes)
