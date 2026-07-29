import os
import tempfile
import unittest

from lfm1b_protocol.io import read_listening_events
from lfm1b_protocol.models import Event
from lfm1b_protocol.split import temporal_split


class SplitTests(unittest.TestCase):
    def test_ties_repeats_three_relations_and_order_independence(self):
        events = [
            Event(1, 10, 100, 1000, 1, 8),
            Event(1, 10, 100, 1000, 1, 1),
            Event(1, 11, 101, 1001, 2, 3),
            Event(1, 12, 102, 1002, 2, 4),
            Event(1, 10, 100, 1000, 2, 9),
            Event(1, 11, 103, 1003, 3, 5),
            Event(1, 13, 101, 1004, 3, 6),
        ]
        result = temporal_split(events)
        reversed_result = temporal_split(reversed(events))
        self.assertEqual(result, reversed_result)
        train_artist = next(row for row in result.split.train
                            if row.item_type == "artist")
        self.assertEqual((train_artist.play_count, train_artist.first_timestamp,
                          train_artist.last_timestamp), (2, 1, 1))
        self.assertEqual({row.item_type for row in result.split.validation},
                         {"artist", "album", "track"})
        self.assertNotIn(10, {row.item_id for row in result.split.validation
                             if row.item_type == "artist"})
        self.assertEqual(result.statistics.repeated_validation_pairs_excluded, 3)
        self.assertNotIn(11, {row.item_id for row in result.split.test
                             if row.item_type == "artist"})
        self.assertNotIn(101, {row.item_id for row in result.split.test
                              if row.item_type == "album"})
        self.assertEqual(result.statistics.repeated_test_pairs_excluded, 2)

    def test_records_are_immutable(self):
        event = Event(1, 2, 3, 4, 5)
        with self.assertRaises(AttributeError):
            event.timestamp = 6

    def test_insufficient_history_still_contributes_known_and_catalog(self):
        result = temporal_split([Event(9, 7, None, None, 1),
                                 Event(9, 8, None, None, 2)])
        self.assertEqual(result.statistics.insufficient_users, (9,))
        self.assertEqual(result.split.train, ())
        self.assertEqual(result.catalogs["artist"], (7, 8))
        self.assertEqual(result.known_positives["artist"], ((9, 7), (9, 8)))

    def test_stream_reader_missing_ids(self):
        descriptor, path = tempfile.mkstemp()
        try:
            with os.fdopen(descriptor, "w") as destination:
                destination.write("1\t10\t\t\t99\n")
            event = next(read_listening_events(path))
            self.assertEqual((event.artist_id, event.album_id, event.track_id),
                             (10, None, None))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
