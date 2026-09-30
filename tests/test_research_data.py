import copy
import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lfm1b_protocol.canonical import sha256
from research_data.acquisition import (AcquisitionError, HTTPSDataTransport,
                                       MAX_CHECKSUM_BYTES, available_datasets,
                                       cache_directory, discover, fetch,
                                       listenbrainz_protocol_inputs, load_lock,
                                       save_lock, validate_lock,
                                       verify_downloads,
                                       verify_musicbrainz_signature)
from research_data.hygiene import (ALLOWED_TRACKED_FIXTURES,
                                   FORBIDDEN_TRACKED_BASENAMES,
                                   FORBIDDEN_TRACKED_COMPONENTS,
                                   FORBIDDEN_TRACKED_SUFFIXES,
                                   RepositoryHygieneError,
                                   assert_repository_hygiene,
                                   ensure_generated_path,
                                   tracked_hygiene_violations)


ROOT = Path(__file__).resolve().parent.parent


class FakeTransport:
    def __init__(self, metadata, downloads, sizes=None):
        self.metadata = dict(metadata)
        self.downloads = dict(downloads)
        self.sizes = dict(sizes or {})
        self.starts = []

    def read_bytes(self, url, limit):
        value = self.metadata[url]
        if len(value) > limit:
            raise AcquisitionError("fake metadata exceeds limit")
        return value

    def size(self, url):
        return self.sizes.get(url, len(self.downloads[url]))

    def open_download(self, url, start, expected_size):
        value = self.downloads[url]
        if len(value) != expected_size:
            raise AcquisitionError("fake size contradiction")
        self.starts.append((url, start))
        return io.BytesIO(value[start:])


def index(*links):
    return ("<html><body>%s</body></html>" % "".join(
        '<a href="%s">%s</a>' % (value, value) for value in links)).encode("utf-8")


def sample_transport(archive=b"sample archive bytes"):
    root = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/sample/"
    old = "listenbrainz-sample-20250530-204040-full"
    snapshot = "listenbrainz-sample-20250610-094311-full"
    snapshot_url = root + snapshot + "/"
    archive_name = "listenbrainz-sample-dump-20250610-094311.tar.zst"
    archive_url = snapshot_url + archive_name
    checksum_url = archive_url + ".sha256"
    checksum = hashlib.sha256(archive).hexdigest().encode("ascii")
    metadata = {
        root: index("../", old + "/", snapshot + "/", "LATEST", "https://evil.invalid/"),
        snapshot_url: index("../", archive_name, archive_name + ".sha256"),
        checksum_url: checksum,
    }
    downloads = {archive_url: archive, checksum_url: checksum}
    return FakeTransport(metadata, downloads), snapshot, archive_name


def musicbrainz_transport(archive=b"musicbrainz core"):
    root = "https://data.metabrainz.org/pub/musicbrainz/data/fullexport/"
    snapshot = "20260902-002507"
    snapshot_url = root + snapshot + "/"
    archive_name = "mbdump.tar.bz2"
    archive_url = snapshot_url + archive_name
    checksum_url = snapshot_url + "SHA256SUMS"
    checksum = (hashlib.sha256(archive).hexdigest() + " *" + archive_name + "\n").encode("ascii")
    checksum_signature = b"signed checksum metadata"
    archive_signature = b"signed archive metadata"
    metadata = {
        root: index("../", "20260829-002439/", snapshot + "/", "LATEST"),
        snapshot_url: index("../", "SHA256SUMS", "SHA256SUMS.asc",
                            archive_name, archive_name + ".asc"),
        checksum_url: checksum,
        snapshot_url + "SHA256SUMS.asc": checksum_signature,
        archive_url + ".asc": archive_signature,
    }
    downloads = {
        checksum_url: checksum,
        snapshot_url + "SHA256SUMS.asc": checksum_signature,
        archive_url + ".asc": archive_signature,
        archive_url: archive,
    }
    return FakeTransport(metadata, downloads), snapshot, archive_name


class ResearchDataTests(unittest.TestCase):
    def test_supported_products_exclude_unassembled_incrementals(self):
        names = {value["dataset"] for value in available_datasets()}
        self.assertEqual(names, {"listenbrainz-sample", "listenbrainz-full",
                                 "musicbrainz-canonical", "musicbrainz-core"})

    def test_protocol_inputs_are_derived_from_the_verified_lock(self):
        lock = discover("listenbrainz-sample", transport=sample_transport()[0])
        inputs = listenbrainz_protocol_inputs(lock)
        self.assertEqual(inputs["lock_hash"], lock["lock_hash"])
        self.assertEqual(inputs["dump_id"], lock["snapshot"]["id"])
        self.assertEqual(inputs["dump_type"], "full")
        archive = next(value for value in lock["files"]
                       if value["role"] == "archive")
        self.assertEqual(inputs["archive_filename"], archive["filename"])
        self.assertEqual(inputs["archive_sha256"], archive["sha256"])
        other = discover("musicbrainz-core", transport=musicbrainz_transport()[0])
        with self.assertRaisesRegex(AcquisitionError, "not a supported ListenBrainz"):
            listenbrainz_protocol_inputs(other)

    def test_sample_discovery_is_deterministic_and_pins_latest(self):
        transport, snapshot, archive_name = sample_transport()
        first = discover("listenbrainz-sample", transport=transport)
        second = discover("listenbrainz-sample", snapshot=snapshot,
                          transport=sample_transport()[0])
        self.assertEqual(first, second)
        self.assertEqual(first["snapshot"]["id"], snapshot)
        self.assertEqual(first["files"][-1]["filename"], archive_name)
        self.assertEqual(first["files"][-1]["sha256"],
                         hashlib.sha256(b"sample archive bytes").hexdigest())
        self.assertEqual(first["lock_hash"], sha256(
            {key: value for key, value in first.items() if key != "lock_hash"}))

    def test_musicbrainz_discovery_includes_signed_integrity_files(self):
        transport, snapshot, archive_name = musicbrainz_transport()
        lock = discover("musicbrainz-core", transport=transport)
        self.assertEqual(lock["snapshot"]["id"], snapshot)
        self.assertEqual([value["role"] for value in lock["files"]],
                         ["checksum", "checksum_signature", "archive_signature", "archive"])
        self.assertEqual(lock["files"][-1]["filename"], archive_name)
        self.assertEqual(lock["signing_fingerprint"],
                         "D5E63B4BDCCE195642948684B8FC2375C777580F")

    def test_discovery_rejects_missing_ambiguous_and_unlisted_snapshots(self):
        transport, snapshot, unused_archive = sample_transport()
        with self.assertRaisesRegex(AcquisitionError, "not present"):
            discover("listenbrainz-sample",
                     snapshot="listenbrainz-sample-20250101-000000-full",
                     transport=transport)
        broken = sample_transport()[0]
        snapshot_url = (
            "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/sample/" +
            snapshot + "/")
        broken.metadata[snapshot_url] = index("unrelated.tar.zst")
        with self.assertRaisesRegex(AcquisitionError, "exactly one"):
            discover("listenbrainz-sample", transport=broken)

    def test_lock_round_trip_expected_hash_and_rehashed_semantic_tampering(self):
        lock = discover("listenbrainz-sample", transport=sample_transport()[0])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "lock.json")
            save_lock(path, lock)
            self.assertEqual(load_lock(path, lock["lock_hash"]), lock)
            with self.assertRaisesRegex(AcquisitionError, "expected"):
                load_lock(path, "0" * 64)
            outside = Path(directory) / "outside"
            outside.mkdir()
            redirect = Path(directory) / "redirect"
            redirect.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(AcquisitionError, "symbolic links"):
                save_lock(redirect / "nested" / "lock.json", lock)
            self.assertFalse((outside / "nested" / "lock.json").exists())
        evil = copy.deepcopy(lock)
        evil["files"][-1]["url"] = "https://evil.invalid/archive"
        evil["lock_hash"] = sha256(
            {key: value for key, value in evil.items() if key != "lock_hash"})
        with self.assertRaisesRegex(AcquisitionError, "snapshot-local"):
            validate_lock(evil)

    def test_fetch_verify_reuse_and_resume_without_repository_outputs(self):
        transport, unused_snapshot, unused_archive = sample_transport()
        lock = discover("listenbrainz-sample", transport=transport)
        with tempfile.TemporaryDirectory() as directory:
            fresh_transport = sample_transport()[0]
            receipt = fetch(lock, directory, max_bytes=1024,
                            transport=fresh_transport)
            self.assertEqual(receipt["verified_files"], 2)
            self.assertEqual([value["status"] for value in receipt["files"]],
                             ["downloaded", "downloaded"])
            no_network = FakeTransport({}, {})
            reused = fetch(lock, directory, max_bytes=1024,
                           transport=no_network)
            self.assertEqual([value["status"] for value in reused["files"]],
                             ["reused", "reused"])
            self.assertEqual(no_network.starts, [])

        with tempfile.TemporaryDirectory() as directory:
            resumed_transport = sample_transport()[0]
            cache = cache_directory(directory, lock)
            cache.mkdir(parents=True)
            archive_record = lock["files"][-1]
            partial = cache / (archive_record["filename"] + ".part")
            partial.write_bytes(b"sample")
            receipt = fetch(lock, directory, max_bytes=1024,
                            transport=resumed_transport)
            archive_starts = [start for url, start in resumed_transport.starts
                              if url == archive_record["url"]]
            self.assertEqual(archive_starts, [6])
            self.assertEqual(receipt["verified_files"], 2)

    def test_fetch_size_guard_corruption_and_checksum_relationship(self):
        lock = discover("listenbrainz-sample", transport=sample_transport()[0])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AcquisitionError, "raise --max-bytes"):
                fetch(lock, directory, max_bytes=1, transport=sample_transport()[0])
        with tempfile.TemporaryDirectory() as directory:
            fetch(lock, directory, max_bytes=1024, transport=sample_transport()[0])
            archive = cache_directory(directory, lock) / lock["files"][-1]["filename"]
            archive.write_bytes(b"corrupt")
            with self.assertRaisesRegex(AcquisitionError, "size mismatch"):
                verify_downloads(lock, directory)

        relationship = copy.deepcopy(lock)
        false_checksum = b"0" * 64
        relationship["files"][0]["sha256"] = hashlib.sha256(false_checksum).hexdigest()
        relationship["lock_hash"] = sha256(
            {key: value for key, value in relationship.items() if key != "lock_hash"})
        with tempfile.TemporaryDirectory() as directory:
            cache = cache_directory(directory, relationship)
            cache.mkdir(parents=True)
            for record in relationship["files"]:
                content = (false_checksum if record["role"] == "checksum" else
                           b"sample archive bytes")
                (cache / record["filename"]).write_bytes(content)
            with self.assertRaisesRegex(AcquisitionError, "contradicts"):
                verify_downloads(relationship, directory)

        oversized = copy.deepcopy(lock)
        oversized["files"][0]["size_bytes"] = MAX_CHECKSUM_BYTES + 1
        oversized["total_size_bytes"] = sum(
            value["size_bytes"] for value in oversized["files"])
        oversized["lock_hash"] = sha256(
            {key: value for key, value in oversized.items() if key != "lock_hash"})
        with self.assertRaisesRegex(AcquisitionError, "metadata exceeds"):
            validate_lock(oversized)

    def test_external_snapshot_mirror_is_hash_verified_without_exposing_url(self):
        lock = discover("listenbrainz-sample", transport=sample_transport()[0])
        mirror = "https://mirror.example.invalid/snapshots/pinned/"
        official = sample_transport()[0]
        mirrored_downloads = {
            mirror + record["filename"]: official.downloads[record["url"]]
            for record in lock["files"]}
        transport = FakeTransport({}, mirrored_downloads)
        with tempfile.TemporaryDirectory() as directory:
            receipt = fetch(lock, directory, max_bytes=1024, transport=transport,
                            mirror_base_url=mirror)
        self.assertTrue(all(value["source"] == "external_mirror"
                            for value in receipt["files"]))
        self.assertNotIn("mirror.example.invalid", repr(receipt))
        self.assertEqual({url for url, unused_start in transport.starts},
                         set(mirrored_downloads))

    def test_musicbrainz_signature_requires_exact_documented_fingerprint(self):
        transport, unused_snapshot, unused_archive = musicbrainz_transport()
        lock = discover("musicbrainz-core", transport=transport)
        with tempfile.TemporaryDirectory() as directory:
            fetch(lock, directory, max_bytes=1024,
                  transport=musicbrainz_transport()[0])
            keyring = os.path.join(directory, "trusted.gpg")
            Path(keyring).write_bytes(b"keyring")
            valid = SimpleNamespace(
                returncode=0,
                stdout="[GNUPG:] VALIDSIG D5E63B4BDCCE195642948684B8FC2375C777580F 0 0\n",
                stderr="")
            with mock.patch("research_data.acquisition.subprocess.run",
                            return_value=valid) as run:
                output = verify_musicbrainz_signature(lock, directory, keyring)
            self.assertTrue(output["pgp_verified"])
            self.assertIn("--no-default-keyring", run.call_args[0][0])
            wrong = SimpleNamespace(
                returncode=0,
                stdout="[GNUPG:] VALIDSIG 0000000000000000000000000000000000000000 0 0\n",
                stderr="")
            with mock.patch("research_data.acquisition.subprocess.run",
                            return_value=wrong):
                with self.assertRaisesRegex(AcquisitionError, "verification failed"):
                    verify_musicbrainz_signature(lock, directory, keyring)

            subkey = "1111111111111111111111111111111111111111"
            primary = "D5E63B4BDCCE195642948684B8FC2375C777580F"
            subkey_status = SimpleNamespace(
                returncode=0,
                stdout=("[GNUPG:] VALIDSIG %s 2026-09-02 0 0 4 0 1 10 00 %s\n" %
                        (subkey, primary)), stderr="")
            with mock.patch("research_data.acquisition.subprocess.run",
                            return_value=subkey_status):
                output = verify_musicbrainz_signature(lock, directory, keyring)
            self.assertEqual(output["signing_fingerprint"], subkey)
            self.assertEqual(output["primary_fingerprint"], primary)

    def test_https_transport_rejects_unsafe_sources(self):
        transport = HTTPSDataTransport()
        for url in ("http://data.metabrainz.org/file",
                    "https://evil.invalid/file",
                    "https://user:secret@data.metabrainz.org/file",
                    "https://data.metabrainz.org:444/file",
                    "https://data.metabrainz.org:not-a-port/file"):
            with self.assertRaises(AcquisitionError):
                transport._validate_url(url)
        with self.assertRaisesRegex(AcquisitionError, "authorization header"):
            HTTPSDataTransport(authorization="Bearer secret\r\nInjected: yes")

    def test_https_transport_uses_ephemeral_authorization_header(self):
        class Response(io.BytesIO):
            def __init__(self):
                io.BytesIO.__init__(self, b"index")
                self.headers = {"Content-Length": "5"}

            def getcode(self):
                return 200

        transport = HTTPSDataTransport(
            allowed_hosts=("mirror.example.invalid",),
            authorization="Bearer secret")
        with mock.patch.object(transport.opener, "open",
                               return_value=Response()) as opened:
            self.assertEqual(
                transport.read_bytes("https://mirror.example.invalid/index", 5),
                b"index")
        request = opened.call_args[0][0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertNotIn("secret", repr(transport.allowed_hosts))

        lock = discover("listenbrainz-sample", transport=sample_transport()[0])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AcquisitionError, "only for an explicit mirror"):
                fetch(lock, directory, max_bytes=1024,
                      mirror_authorization="Bearer secret")

    def test_https_transport_requires_exact_range_response(self):
        class Response(io.BytesIO):
            def __init__(self, status, headers):
                io.BytesIO.__init__(self, b"remaining")
                self.status = status
                self.headers = headers

            def getcode(self):
                return self.status

        transport = HTTPSDataTransport()
        valid = Response(206, {"Content-Range": "bytes 3-9/10", "Content-Length": "7"})
        with mock.patch.object(transport, "_open", return_value=valid):
            self.assertIs(transport.open_download(
                "https://data.metabrainz.org/file", 3, 10), valid)
        for headers in (
                {"Content-Range": "bytes 3-8/10", "Content-Length": "6"},
                {"Content-Range": "bytes 2-9/10", "Content-Length": "8"},
                {"Content-Range": "bytes 3-9/10", "Content-Length": "6"}):
            response = Response(206, headers)
            with mock.patch.object(transport, "_open", return_value=response):
                with self.assertRaisesRegex(AcquisitionError, "range request"):
                    transport.open_download(
                        "https://data.metabrainz.org/file", 3, 10)
        response = Response(200, {"Content-Length": "9"})
        with mock.patch.object(transport, "_open", return_value=response):
            with self.assertRaisesRegex(AcquisitionError, "size/status"):
                transport.open_download("https://data.metabrainz.org/file", 0, 10)

    def test_cache_rejects_symlinks_and_world_writable_directories(self):
        lock = discover("listenbrainz-sample", transport=sample_transport()[0])
        with tempfile.TemporaryDirectory() as directory:
            cache = cache_directory(directory, lock)
            cache.mkdir(parents=True)
            outside = Path(directory) / "outside"
            outside.write_bytes(b"not the checksum")
            os.symlink(str(outside), str(cache / lock["files"][0]["filename"]))
            with self.assertRaisesRegex(AcquisitionError, "single-link regular"):
                fetch(lock, directory, max_bytes=1024, transport=sample_transport()[0])
        with tempfile.TemporaryDirectory() as directory:
            cache = cache_directory(directory, lock)
            cache.mkdir(parents=True)
            outside = Path(directory) / "hardlinked"
            outside.write_bytes(b"must not be truncated")
            partial = cache / (lock["files"][0]["filename"] + ".part")
            os.link(str(outside), str(partial))
            with self.assertRaisesRegex(AcquisitionError, "single-link regular"):
                fetch(lock, directory, max_bytes=1024, restart=True,
                      transport=sample_transport()[0])
            self.assertEqual(outside.read_bytes(), b"must not be truncated")
        with tempfile.TemporaryDirectory() as directory:
            cache = cache_directory(directory, lock)
            cache.mkdir(parents=True)
            os.chmod(str(cache), 0o777)
            try:
                with self.assertRaisesRegex(AcquisitionError, "group/world writable"):
                    verify_downloads(lock, directory)
            finally:
                os.chmod(str(cache), 0o700)
        with tempfile.TemporaryDirectory() as directory:
            cache = cache_directory(directory, lock)
            cache.mkdir(parents=True)
            checksum = cache / lock["files"][0]["filename"]
            checksum.write_bytes(b"unsafe")
            os.chmod(str(checksum), 0o666)
            with self.assertRaisesRegex(AcquisitionError, "group/world writable"):
                verify_downloads(lock, directory)

    def test_repository_hygiene_guards_outputs_and_tracked_data(self):
        self.assertEqual(assert_repository_hygiene(ROOT), ())
        generated = ensure_generated_path(ROOT / "data" / "locks" / "test.json")
        self.assertEqual(generated, (ROOT / "data" / "locks" / "test.json").resolve())
        with self.assertRaisesRegex(RepositoryHygieneError, "generated output"):
            ensure_generated_path(ROOT / "research_data" / "generated.json")
        with mock.patch("research_data.hygiene._tracked_paths", return_value=(
                "research_data/acquisition.py", "downloads/archive.tar.zst",
                "examples/listenbrainz_synthetic.listens", "dist/package.whl",
                "inputs/real.csv", "inputs/real.dat", "inputs/generated.json")):
            self.assertEqual(
                tracked_hygiene_violations(ROOT),
                ("dist/package.whl", "downloads/archive.tar.zst",
                 "inputs/generated.json", "inputs/real.csv", "inputs/real.dat"))

        suffix_paths = tuple("forced/value" + suffix
                             for suffix in FORBIDDEN_TRACKED_SUFFIXES)
        with mock.patch("research_data.hygiene._tracked_paths",
                        return_value=suffix_paths):
            self.assertEqual(tracked_hygiene_violations(ROOT),
                             tuple(sorted(suffix_paths)))

        component_paths = tuple("nested/%s/payload" % component
                                for component in FORBIDDEN_TRACKED_COMPONENTS)
        basename_paths = tuple("nested/" + value
                               for value in FORBIDDEN_TRACKED_BASENAMES)
        forced = component_paths + basename_paths + (
            "nested/package.egg-info/PKG-INFO", "nested/.coverage.worker")
        with mock.patch("research_data.hygiene._tracked_paths", return_value=forced):
            self.assertEqual(tracked_hygiene_violations(ROOT), tuple(sorted(forced)))

        near_fixture = "other/listenbrainz_synthetic.listens"
        fixtures = tuple(sorted(ALLOWED_TRACKED_FIXTURES)) + (near_fixture,)
        with mock.patch("research_data.hygiene._tracked_paths", return_value=fixtures):
            self.assertEqual(tracked_hygiene_violations(ROOT), (near_fixture,))

        ignore_rules = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
        for suffix in FORBIDDEN_TRACKED_SUFFIXES:
            self.assertIn("*" + suffix, ignore_rules)


if __name__ == "__main__":
    unittest.main()
