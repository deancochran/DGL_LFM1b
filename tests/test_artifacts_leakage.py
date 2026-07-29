import json
import os
import tempfile
import unittest

from lfm1b_protocol.artifacts import (ArtifactIntegrityError, load_bundle,
                                      save_bundle)
from lfm1b_protocol.leakage import verify_endpoint_mappings
from lfm1b_protocol.models import PairInteraction


class ArtifactAndLeakageTests(unittest.TestCase):
    def test_artifact_stable_identity_and_tampering(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = save_bundle(first, {"example": {"b": 2, "a": 1}})
            two = save_bundle(second, {"example": {"a": 1, "b": 2}})
            self.assertEqual(one["identity_sha256"], two["identity_sha256"])
            self.assertEqual(load_bundle(first)["example"], {"a": 1, "b": 2})
            with open(os.path.join(first, "example.json"), "w") as destination:
                json.dump({"a": 9}, destination)
            with self.assertRaises(ArtifactIntegrityError):
                load_bundle(first)

    def test_leakage_forward_reverse_and_clean(self):
        heldout = (PairInteraction(1, "artist", 7, 1, 3, 3),
                   PairInteraction(2, "album", 8, 1, 3, 3),
                   PairInteraction(3, "track", 9, 1, 3, 3))
        report = verify_endpoint_mappings({
            ("user", "listened_to_artist", "artist"): ((1, 7),),
            "artist_listened_by": ((7, 1),),
            "listened_to_album": ((2, 8),),
            "album_listened_by": ((8, 2),),
            "listened_to_track": ((3, 9),),
            "track_listened_by": ((9, 3),),
        }, heldout)
        self.assertFalse(report.clean)
        self.assertEqual(len(report.findings), 6)
        self.assertTrue(verify_endpoint_mappings({"listened_to_artist": ((1, 8),)},
                                                 heldout).clean)


if __name__ == "__main__":
    unittest.main()
