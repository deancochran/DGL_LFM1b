"""Hash-locked acquisition of public MetaBrainz research data.

Discovery reads only small official indexes/checksum files. Fetching is always
an explicit separate action, writes to ignored or external paths, streams to a
partial file, verifies size and SHA256, and atomically promotes valid content.
"""

import hashlib
import json
import os
import re
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from lfm1b_protocol.canonical import canonical_json, sha256

from .hygiene import (atomic_write_generated, ensure_generated_path,
                      open_generated_directory)


LOCK_SCHEMA_VERSION = 1
DEFAULT_MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
MAX_INDEX_BYTES = 2 * 1024 * 1024
MAX_CHECKSUM_BYTES = 2 * 1024 * 1024
MUSICBRAINZ_SIGNING_FINGERPRINT = "D5E63B4BDCCE195642948684B8FC2375C777580F"
_HEX = set("0123456789abcdef")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class AcquisitionError(ValueError):
    pass


_SPECS = {
    "listenbrainz-sample": {
        "description": "ListenBrainz public sample listens dump",
        "index_url": "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/sample/",
        "snapshot_pattern": r"^listenbrainz-sample-(?P<timestamp>\d{8}-\d{6})-full$",
        "archive_pattern": r"^listenbrainz-sample-dump-(?P<timestamp>\d{8}-\d{6})\.tar\.zst$",
        "checksum_mode": "sidecar",
        "license_spdx": "CC0-1.0",
        "documentation_urls": (
            "https://listenbrainz.readthedocs.io/en/latest/users/listenbrainz-dumps.html",
            "https://metabrainz.org/datasets/postgres-dumps#listenbrainz",
        ),
        "caveats": (
            "sample_only_not_a_population_benchmark",
            "sha256_sidecar_has_no_documented_pgp_signature",
        ),
        "signing_fingerprint": None,
    },
    "listenbrainz-full": {
        "description": "ListenBrainz public full listens dump",
        "index_url": "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/fullexport/",
        "snapshot_pattern": r"^listenbrainz-dump-(?P<dump_id>\d+)-(?P<timestamp>\d{8}-\d{6})-full$",
        "archive_pattern": r"^listenbrainz-listens-dump-(?P<dump_id>\d+)-(?P<timestamp>\d{8}-\d{6})-full\.tar\.zst$",
        "checksum_mode": "sidecar",
        "license_spdx": "CC0-1.0",
        "documentation_urls": (
            "https://listenbrainz.readthedocs.io/en/latest/users/listenbrainz-dumps.html",
            "https://listenbrainz.readthedocs.io/en/latest/maintainers/dumps.html",
            "https://metabrainz.org/datasets/postgres-dumps#listenbrainz",
        ),
        "caveats": (
            "upstream_current_full_payload_retention_is_two_snapshots",
            "sha256_sidecar_has_no_documented_pgp_signature",
            "large_download_requires_explicit_size_override",
        ),
        "signing_fingerprint": None,
    },
    "musicbrainz-canonical": {
        "description": "Canonical MusicBrainz mappings and metadata",
        "index_url": "https://data.metabrainz.org/pub/musicbrainz/canonical_data/",
        "snapshot_pattern": r"^musicbrainz-canonical-dump-(?P<timestamp>\d{8}-\d{6})$",
        "archive_pattern": r"^musicbrainz-canonical-dump-(?P<timestamp>\d{8}-\d{6})\.tar\.zst$",
        "checksum_mode": "sidecar",
        "license_spdx": "CC0-1.0",
        "documentation_urls": (
            "https://musicbrainz.org/doc/Canonical_MusicBrainz_data",
            "https://metabrainz.org/datasets/derived-dumps#canonical",
        ),
        "caveats": (
            "canonical_mbids_can_change_between_snapshots",
            "sha256_sidecar_has_no_documented_pgp_signature",
            "large_download_requires_explicit_size_override",
        ),
        "signing_fingerprint": None,
    },
    "musicbrainz-core": {
        "description": "MusicBrainz core PostgreSQL data dump",
        "index_url": "https://data.metabrainz.org/pub/musicbrainz/data/fullexport/",
        "snapshot_pattern": r"^(?P<timestamp>\d{8}-\d{6})$",
        "archive_pattern": r"^mbdump\.tar\.bz2$",
        "checksum_mode": "sha256sums",
        "license_spdx": "CC0-1.0",
        "documentation_urls": (
            "https://musicbrainz.org/doc/MusicBrainz_Database/Download",
            "https://metabrainz.org/datasets/postgres-dumps#musicbrainz",
        ),
        "caveats": (
            "postgresql_import_is_the_supported_consumption_path",
            "supplementary_dumps_have_different_licenses_and_are_not_included",
            "large_download_requires_explicit_size_override",
        ),
        "signing_fingerprint": MUSICBRAINZ_SIGNING_FINGERPRINT,
    },
}


def available_datasets():
    return tuple({"dataset": name, "description": _SPECS[name]["description"]}
                 for name in sorted(_SPECS))


def _lower_digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise AcquisitionError(label + " must be a lowercase SHA256 digest")
    return value


def _nonnegative_integer(value, label, positive=False):
    if (isinstance(value, bool) or not isinstance(value, int) or value < 0 or
            (positive and value == 0)):
        raise AcquisitionError(label + " must be a %snon-negative integer" %
                               ("positive " if positive else ""))
    return value


def _safe_name(value, label):
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise AcquisitionError(label + " is not a safe filename")
    return value


class _IndexLinks(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and isinstance(value, str):
                self.hrefs.append(value)


def _index_entries(content):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AcquisitionError("index is not UTF-8: %s" % error)
    parser = _IndexLinks()
    parser.feed(text)
    entries = []
    for href in parser.hrefs:
        parsed = urllib.parse.urlsplit(href)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            continue
        decoded = urllib.parse.unquote(parsed.path)
        is_directory = decoded.endswith("/")
        value = decoded[:-1] if is_directory else decoded
        if "/" in value or not _SAFE_NAME.fullmatch(value or ""):
            continue
        entries.append((value, is_directory))
    return tuple(sorted(set(entries)))


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect rejected", headers, fp)


class HTTPSDataTransport:
    """Small HTTPS transport with normal certificate checks and no redirects."""
    def __init__(self, allowed_hosts=None, timeout=60, authorization=None):
        self.allowed_hosts = set(allowed_hosts or ("data.metabrainz.org",))
        self.timeout = timeout
        if (authorization is not None and
                (not isinstance(authorization, str) or not authorization or
                 "\r" in authorization or "\n" in authorization)):
            raise AcquisitionError("mirror authorization header is invalid")
        self.authorization = authorization
        self.opener = urllib.request.build_opener(_RejectRedirects())

    def _validate_url(self, url):
        parsed = urllib.parse.urlsplit(url)
        try:
            port = parsed.port
        except ValueError:
            raise AcquisitionError("URL contains an invalid port")
        if (parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts or
                parsed.username is not None or parsed.password is not None or
                port not in (None, 443) or parsed.query or parsed.fragment):
            raise AcquisitionError("URL must use approved credential-free HTTPS host")

    def _open(self, url, method="GET", headers=None):
        self._validate_url(url)
        request_headers = {"Accept-Encoding": "identity",
                           "User-Agent": "dgl-lfm1b-research-data/1"}
        if self.authorization is not None:
            request_headers["Authorization"] = self.authorization
        request_headers.update(headers or {})
        request = urllib.request.Request(url, headers=request_headers, method=method)
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            raise AcquisitionError("cannot read %s: %s" % (url, error))
        encoding = response.headers.get("Content-Encoding")
        if encoding not in (None, "identity"):
            response.close()
            raise AcquisitionError("encoded HTTP responses cannot be byte-verified")
        return response

    def read_bytes(self, url, limit):
        with self._open(url) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    if int(declared) > limit:
                        raise AcquisitionError("remote metadata exceeds size limit")
                except ValueError:
                    raise AcquisitionError("invalid remote Content-Length")
            value = response.read(limit + 1)
        if len(value) > limit:
            raise AcquisitionError("remote metadata exceeds size limit")
        return value

    def size(self, url):
        with self._open(url, method="HEAD") as response:
            value = response.headers.get("Content-Length")
        try:
            size = int(value)
        except (TypeError, ValueError):
            raise AcquisitionError("archive HEAD response lacks a valid Content-Length")
        return _nonnegative_integer(size, "remote size", positive=True)

    def open_download(self, url, start, expected_size):
        headers = {"Range": "bytes=%d-" % start} if start else None
        response = self._open(url, headers=headers)
        status = response.getcode()
        if start:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            declared_length = response.headers.get("Content-Length")
            invalid_length = False
            if declared_length is not None:
                try:
                    invalid_length = int(declared_length) != expected_size - start
                except ValueError:
                    invalid_length = True
            if (status != 206 or match is None or int(match.group(1)) != start or
                    int(match.group(2)) != expected_size - 1 or
                    int(match.group(3)) != expected_size or invalid_length):
                response.close()
                raise AcquisitionError(
                    "server did not honor the resumable range request; use --restart")
        else:
            declared_length = response.headers.get("Content-Length")
            invalid_length = False
            if declared_length is not None:
                try:
                    invalid_length = int(declared_length) != expected_size
                except ValueError:
                    invalid_length = True
            if status != 200 or invalid_length:
                response.close()
                raise AcquisitionError("archive download response has invalid size/status")
        return response


def _snapshot_values(pattern, value):
    match = re.fullmatch(pattern, value)
    return match.groupdict() if match is not None else None


def _snapshot_sort_key(spec, value):
    groups = _snapshot_values(spec["snapshot_pattern"], value)
    dump_id = int(groups.get("dump_id") or 0)
    return groups["timestamp"], dump_id, value


def _parse_direct_checksum(content):
    try:
        value = content.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise AcquisitionError("checksum sidecar is not ASCII: %s" % error)
    return _lower_digest(value, "archive checksum")


def _parse_sha256sums(content, filename):
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as error:
        raise AcquisitionError("SHA256SUMS is not ASCII: %s" % error)
    matches = []
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if match is not None and match.group(2) == filename:
            matches.append(match.group(1))
    if len(matches) != 1:
        raise AcquisitionError("SHA256SUMS must contain the archive exactly once")
    return matches[0]


def _remote_file(role, filename, snapshot_url, digest, size):
    _safe_name(filename, role + " filename")
    return {"role": role, "filename": filename,
            "url": urllib.parse.urljoin(snapshot_url, filename),
            "size_bytes": size, "sha256": digest}


def discover(dataset, snapshot="latest", transport=None):
    """Resolve an official index to a deterministic, hash-locked manifest."""
    if dataset not in _SPECS:
        raise AcquisitionError("unknown dataset: %s" % dataset)
    if not isinstance(snapshot, str) or not snapshot:
        raise AcquisitionError("snapshot must be 'latest' or an exact snapshot ID")
    spec = _SPECS[dataset]
    transport = transport or HTTPSDataTransport()
    root_entries = _index_entries(transport.read_bytes(spec["index_url"], MAX_INDEX_BYTES))
    snapshots = [name for name, is_directory in root_entries
                 if is_directory and _snapshot_values(spec["snapshot_pattern"], name)]
    if not snapshots:
        raise AcquisitionError("official index contains no matching snapshots")
    if snapshot == "latest":
        snapshot_id = max(snapshots, key=lambda value: _snapshot_sort_key(spec, value))
    else:
        if _snapshot_values(spec["snapshot_pattern"], snapshot) is None:
            raise AcquisitionError("snapshot ID does not match the dataset format")
        if snapshot not in snapshots:
            raise AcquisitionError("requested snapshot is not present in the official index")
        snapshot_id = snapshot
    snapshot_url = urllib.parse.urljoin(spec["index_url"], snapshot_id + "/")
    entries = _index_entries(transport.read_bytes(snapshot_url, MAX_INDEX_BYTES))
    filenames = {name for name, is_directory in entries if not is_directory}
    snapshot_groups = _snapshot_values(spec["snapshot_pattern"], snapshot_id)
    candidates = []
    for filename in filenames:
        groups = _snapshot_values(spec["archive_pattern"], filename)
        if groups is None:
            continue
        if any(groups.get(key) != value for key, value in snapshot_groups.items()
               if key in groups):
            continue
        candidates.append(filename)
    if len(candidates) != 1:
        raise AcquisitionError("snapshot must contain exactly one matching archive")
    archive_name = candidates[0]
    archive_size = transport.size(urllib.parse.urljoin(snapshot_url, archive_name))

    files = []
    if spec["checksum_mode"] == "sidecar":
        checksum_name = archive_name + ".sha256"
        if checksum_name not in filenames:
            raise AcquisitionError("snapshot lacks the required SHA256 sidecar")
        checksum_content = transport.read_bytes(
            urllib.parse.urljoin(snapshot_url, checksum_name), MAX_CHECKSUM_BYTES)
        archive_digest = _parse_direct_checksum(checksum_content)
        files.append(_remote_file("checksum", checksum_name, snapshot_url,
                                  hashlib.sha256(checksum_content).hexdigest(),
                                  len(checksum_content)))
    else:
        checksum_name = "SHA256SUMS"
        signature_name = "SHA256SUMS.asc"
        archive_signature_name = archive_name + ".asc"
        for required in (checksum_name, signature_name, archive_signature_name):
            if required not in filenames:
                raise AcquisitionError("snapshot lacks required integrity file: " + required)
        checksum_content = transport.read_bytes(
            urllib.parse.urljoin(snapshot_url, checksum_name), MAX_CHECKSUM_BYTES)
        archive_digest = _parse_sha256sums(checksum_content, archive_name)
        files.append(_remote_file("checksum", checksum_name, snapshot_url,
                                  hashlib.sha256(checksum_content).hexdigest(),
                                  len(checksum_content)))
        for role, filename in (("checksum_signature", signature_name),
                               ("archive_signature", archive_signature_name)):
            content = transport.read_bytes(
                urllib.parse.urljoin(snapshot_url, filename), MAX_CHECKSUM_BYTES)
            files.append(_remote_file(role, filename, snapshot_url,
                                      hashlib.sha256(content).hexdigest(), len(content)))

    files.append(_remote_file("archive", archive_name, snapshot_url,
                              archive_digest, archive_size))
    role_order = {"checksum": 0, "checksum_signature": 1,
                  "archive_signature": 2, "archive": 3}
    files.sort(key=lambda value: role_order[value["role"]])
    identity = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "dataset": dataset,
        "description": spec["description"],
        "license_spdx": spec["license_spdx"],
        "snapshot": {"id": snapshot_id, "index_url": snapshot_url,
                     "discovered_from": spec["index_url"]},
        "documentation_urls": list(spec["documentation_urls"]),
        "caveats": list(spec["caveats"]),
        "signing_fingerprint": spec["signing_fingerprint"],
        "files": files,
        "total_size_bytes": sum(value["size_bytes"] for value in files),
    }
    lock = dict(identity, lock_hash=sha256(identity))
    return validate_lock(lock)


def _lock_identity(lock):
    return {key: value for key, value in lock.items() if key != "lock_hash"}


def _expected_roles(spec):
    if spec["checksum_mode"] == "sidecar":
        return ("checksum", "archive")
    return ("checksum", "checksum_signature", "archive_signature", "archive")


def validate_lock(lock, expected_lock_hash=None):
    required = {"schema_version", "dataset", "description", "license_spdx", "snapshot",
                "documentation_urls", "caveats", "signing_fingerprint", "files",
                "total_size_bytes", "lock_hash"}
    if not isinstance(lock, dict) or set(lock) != required:
        raise AcquisitionError("acquisition lock has an invalid shape")
    if (isinstance(lock["schema_version"], bool) or
            not isinstance(lock["schema_version"], int) or
            lock["schema_version"] != LOCK_SCHEMA_VERSION):
        raise AcquisitionError("unsupported acquisition lock schema")
    dataset = lock["dataset"]
    if dataset not in _SPECS:
        raise AcquisitionError("unknown dataset in acquisition lock")
    spec = _SPECS[dataset]
    if (lock["description"] != spec["description"] or
            lock["license_spdx"] != spec["license_spdx"] or
            lock["documentation_urls"] != list(spec["documentation_urls"]) or
            lock["caveats"] != list(spec["caveats"]) or
            lock["signing_fingerprint"] != spec["signing_fingerprint"]):
        raise AcquisitionError("acquisition lock contradicts the dataset recipe")
    snapshot = lock["snapshot"]
    if not isinstance(snapshot, dict) or set(snapshot) != {"id", "index_url", "discovered_from"}:
        raise AcquisitionError("acquisition snapshot has an invalid shape")
    if (_snapshot_values(spec["snapshot_pattern"], snapshot["id"]) is None or
            snapshot["discovered_from"] != spec["index_url"] or
            snapshot["index_url"] != urllib.parse.urljoin(
                spec["index_url"], snapshot["id"] + "/")):
        raise AcquisitionError("acquisition snapshot is not pinned to the official index")
    files = lock["files"]
    roles = _expected_roles(spec)
    if (not isinstance(files, list) or tuple(value.get("role")
                                             if isinstance(value, dict) else None
                                             for value in files) != roles):
        raise AcquisitionError("acquisition files are incomplete or out of order")
    by_role = {}
    for value in files:
        if set(value) != {"role", "filename", "url", "size_bytes", "sha256"}:
            raise AcquisitionError("acquisition file has an invalid shape")
        filename = _safe_name(value["filename"], value["role"] + " filename")
        if value["url"] != urllib.parse.urljoin(snapshot["index_url"], filename):
            raise AcquisitionError("acquisition file URL is not snapshot-local")
        _nonnegative_integer(value["size_bytes"], value["role"] + " size", positive=True)
        if value["role"] != "archive" and value["size_bytes"] > MAX_CHECKSUM_BYTES:
            raise AcquisitionError("integrity metadata exceeds its size limit")
        _lower_digest(value["sha256"], value["role"] + " SHA256")
        by_role[value["role"]] = value
    archive = by_role["archive"]
    archive_groups = _snapshot_values(spec["archive_pattern"], archive["filename"])
    snapshot_groups = _snapshot_values(spec["snapshot_pattern"], snapshot["id"])
    if (archive_groups is None or
            any(archive_groups.get(key) != expected
                for key, expected in snapshot_groups.items() if key in archive_groups)):
        raise AcquisitionError("archive filename contradicts the snapshot")
    if spec["checksum_mode"] == "sidecar":
        if by_role["checksum"]["filename"] != archive["filename"] + ".sha256":
            raise AcquisitionError("checksum filename contradicts the archive")
    elif (by_role["checksum"]["filename"] != "SHA256SUMS" or
          by_role["checksum_signature"]["filename"] != "SHA256SUMS.asc" or
          by_role["archive_signature"]["filename"] != archive["filename"] + ".asc"):
        raise AcquisitionError("MusicBrainz integrity filenames are inconsistent")
    expected_total = sum(value["size_bytes"] for value in files)
    if (_nonnegative_integer(lock["total_size_bytes"], "total size", positive=True) !=
            expected_total):
        raise AcquisitionError("acquisition total size is inconsistent")
    _lower_digest(lock["lock_hash"], "lock hash")
    if lock["lock_hash"] != sha256(_lock_identity(lock)):
        raise AcquisitionError("acquisition lock hash mismatch")
    if expected_lock_hash is not None and lock["lock_hash"] != expected_lock_hash:
        raise AcquisitionError("expected acquisition lock hash does not match")
    return lock


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AcquisitionError("duplicate JSON key in acquisition lock: " + key)
        result[key] = value
    return result


def _json_constant(value):
    raise AcquisitionError("non-finite JSON value in acquisition lock: " + value)


def save_lock(path, lock):
    lock = validate_lock(lock)
    try:
        atomic_write_generated(path, canonical_json(lock))
    except ValueError as error:
        raise AcquisitionError("cannot write acquisition lock: %s" % error)
    return lock


def load_lock(path, expected_lock_hash=None, max_bytes=1024 * 1024):
    _nonnegative_integer(max_bytes, "lock byte limit", positive=True)
    try:
        with open(path, "rb") as stream:
            content = stream.read(max_bytes + 1)
    except OSError as error:
        raise AcquisitionError("cannot read acquisition lock: %s" % error)
    if len(content) > max_bytes:
        raise AcquisitionError("acquisition lock exceeds size limit")
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_json_pairs,
                           parse_constant=_json_constant)
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        if isinstance(error, AcquisitionError):
            raise
        raise AcquisitionError("malformed acquisition lock: %s" % error)
    return validate_lock(value, expected_lock_hash)


def cache_directory(root, lock):
    lock = validate_lock(lock)
    lexical_root = Path(os.path.abspath(str(Path(root).expanduser())))
    return lexical_root / lock["dataset"] / lock["snapshot"]["id"]


def listenbrainz_protocol_inputs(lock):
    """Return acquisition-locked arguments required by ListenBrainz prepare."""
    lock = validate_lock(lock)
    if lock["dataset"] not in ("listenbrainz-sample", "listenbrainz-full"):
        raise AcquisitionError("lock is not a supported ListenBrainz full/sample dump")
    archive = next(value for value in lock["files"] if value["role"] == "archive")
    return {"lock_hash": lock["lock_hash"], "dump_id": lock["snapshot"]["id"],
            "dump_type": "full", "archive_filename": archive["filename"],
            "archive_sha256": archive["sha256"],
            "snapshot_index_url": lock["snapshot"]["index_url"]}


def _regular_file_size(path):
    try:
        value = os.lstat(str(path))
    except FileNotFoundError:
        return None
    except OSError as error:
        raise AcquisitionError("cannot inspect local file: %s" % error)
    if not stat.S_ISREG(value.st_mode):
        raise AcquisitionError("download path must be a regular file, not a link/device")
    return value.st_size


def _open_cache_directory(directory, create=False):
    descriptor = None
    try:
        unused_directory, descriptor = open_generated_directory(directory, create=create)
        value = os.fstat(descriptor)
    except (OSError, ValueError) as error:
        if descriptor is not None:
            os.close(descriptor)
        raise AcquisitionError("cannot open cache directory safely: %s" % error)
    if not stat.S_ISDIR(value.st_mode):
        os.close(descriptor)
        raise AcquisitionError("cache path must be a directory")
    if hasattr(os, "getuid") and value.st_uid != os.getuid():
        os.close(descriptor)
        raise AcquisitionError("cache directory must be owned by the current user")
    if value.st_mode & 0o022:
        os.close(descriptor)
        raise AcquisitionError("cache directory must not be group/world writable")
    return descriptor


def _validate_cache_file_stat(value, label):
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise AcquisitionError(label + " must be a single-link regular file")
    if hasattr(os, "getuid") and value.st_uid != os.getuid():
        raise AcquisitionError(label + " must be owned by the current user")
    if value.st_mode & 0o022:
        raise AcquisitionError(label + " must not be group/world writable")
    return value


def _stat_at(directory_fd, filename):
    _safe_name(filename, "cache filename")
    try:
        value = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise AcquisitionError("cannot inspect cache file: %s" % error)
    return _validate_cache_file_stat(value, "cache file")


def _file_identity(value):
    return (value.st_dev, value.st_ino, value.st_size,
            getattr(value, "st_mtime_ns", int(value.st_mtime * 1000000000)))


def _hash_file_at(directory_fd, filename, expected_size=None):
    digest = hashlib.sha256()
    total = 0
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            _validate_cache_file_stat(opened, "cache file")
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if expected_size is not None and total > expected_size:
                    raise AcquisitionError("local file exceeds expected size")
                digest.update(chunk)
    except OSError as error:
        raise AcquisitionError("cannot hash local file: %s" % error)
    return total, digest.hexdigest(), _file_identity(opened)


def _verify_file_at(directory_fd, record):
    value = _stat_at(directory_fd, record["filename"])
    if value is None:
        raise AcquisitionError("missing downloaded file: " + record["filename"])
    if value.st_size != record["size_bytes"]:
        raise AcquisitionError("downloaded size mismatch: " + record["filename"])
    measured_size, measured_digest, unused_identity = _hash_file_at(
        directory_fd, record["filename"], record["size_bytes"])
    if measured_size != record["size_bytes"] or measured_digest != record["sha256"]:
        raise AcquisitionError("downloaded SHA256 mismatch: " + record["filename"])


def _open_partial_at(directory_fd, filename, append):
    flags = os.O_WRONLY | os.O_CREAT
    if append:
        flags |= os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
    value = os.fstat(descriptor)
    try:
        _validate_cache_file_stat(value, "partial download")
    except AcquisitionError:
        os.close(descriptor)
        raise
    if not append:
        os.ftruncate(descriptor, 0)
    return os.fdopen(descriptor, "ab" if append else "wb"), _file_identity(value)


def _promote_at(directory_fd, partial_name, final_name, expected_identity):
    current = _stat_at(directory_fd, partial_name)
    if current is None or _file_identity(current) != expected_identity:
        raise AcquisitionError("partial download changed before atomic promotion")
    if _stat_at(directory_fd, final_name) is not None:
        raise AcquisitionError("final download appeared before atomic promotion")
    try:
        os.replace(partial_name, final_name,
                   src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as error:
        raise AcquisitionError("cannot atomically promote download: %s" % error)


def _fetch_file(record, directory_fd, transport, restart, source_url):
    final_name = record["filename"]
    partial_name = record["filename"] + ".part"
    final_value = _stat_at(directory_fd, final_name)
    if final_value is not None:
        _verify_file_at(directory_fd, record)
        return "reused"
    partial_value = _stat_at(directory_fd, partial_name)
    if restart and partial_value is not None:
        start = 0
    else:
        start = partial_value.st_size if partial_value is not None else 0
    if start > record["size_bytes"]:
        raise AcquisitionError("partial file exceeds expected size; use --restart")
    digest = hashlib.sha256()
    partial_identity = None
    if start:
        measured_size, measured_digest, partial_identity = _hash_file_at(
            directory_fd, partial_name, start)
        if measured_size != start:
            raise AcquisitionError("partial file size changed during verification")
        if start == record["size_bytes"]:
            if measured_digest != record["sha256"]:
                raise AcquisitionError("complete partial file has wrong SHA256; use --restart")
            _promote_at(directory_fd, partial_name, final_name, partial_identity)
            return "resumed"
        flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
        descriptor = os.open(partial_name, flags, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb") as stream:
            if _file_identity(os.fstat(stream.fileno())) != partial_identity:
                raise AcquisitionError("partial file changed before resume")
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    total = start
    completed_identity = None
    try:
        with transport.open_download(source_url, start, record["size_bytes"]) as response:
            output, opened_identity = _open_partial_at(
                directory_fd, partial_name, append=bool(start))
            with output:
                if start and opened_identity != partial_identity:
                    raise AcquisitionError("partial file changed before append")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > record["size_bytes"]:
                        raise AcquisitionError("download exceeded its locked size")
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
                completed_identity = _file_identity(os.fstat(output.fileno()))
    except OSError as error:
        raise AcquisitionError("download write failed: %s" % error)
    if total != record["size_bytes"]:
        raise AcquisitionError("download ended before its locked size")
    if digest.hexdigest() != record["sha256"]:
        raise AcquisitionError("downloaded SHA256 mismatch; use --restart")
    _promote_at(directory_fd, partial_name, final_name, completed_identity)
    return "resumed" if start else "downloaded"


def _mirror_urls(lock, mirror_base_url):
    if mirror_base_url is None:
        return {value["role"]: value["url"] for value in lock["files"]}, "official"
    if not isinstance(mirror_base_url, str) or not mirror_base_url:
        raise AcquisitionError("mirror base URL must be a nonempty HTTPS URL")
    parsed = urllib.parse.urlsplit(mirror_base_url)
    try:
        port = parsed.port
    except ValueError:
        raise AcquisitionError("mirror base URL contains an invalid port")
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None or
            parsed.password is not None or port not in (None, 443) or
            parsed.query or parsed.fragment):
        raise AcquisitionError("mirror base URL must be credential-free HTTPS")
    base = mirror_base_url if mirror_base_url.endswith("/") else mirror_base_url + "/"
    return ({value["role"]: urllib.parse.urljoin(base, value["filename"])
             for value in lock["files"]}, "external_mirror")


def fetch(lock, root, max_bytes=DEFAULT_MAX_DOWNLOAD_BYTES, restart=False, transport=None,
          mirror_base_url=None, mirror_authorization=None):
    """Fetch every locked file into ``root/dataset/snapshot``.

    The one-GiB default prevents an accidental full-dump transfer. Callers must
    deliberately raise ``max_bytes`` for larger snapshots.
    """
    lock = validate_lock(lock)
    _nonnegative_integer(max_bytes, "maximum download bytes", positive=True)
    if not isinstance(restart, bool):
        raise AcquisitionError("restart must be boolean")
    if lock["total_size_bytes"] > max_bytes:
        raise AcquisitionError(
            "locked download is %d bytes; raise --max-bytes deliberately" %
            lock["total_size_bytes"])
    source_urls, source_kind = _mirror_urls(lock, mirror_base_url)
    if mirror_authorization is not None and mirror_base_url is None:
        raise AcquisitionError("authorization is accepted only for an explicit mirror")
    if transport is not None and mirror_authorization is not None:
        raise AcquisitionError("custom transports must configure their own authorization")
    if transport is None:
        hosts = {"data.metabrainz.org"}
        if mirror_base_url is not None:
            hosts.add(urllib.parse.urlsplit(mirror_base_url).hostname)
        transport = HTTPSDataTransport(hosts, authorization=mirror_authorization)
    directory = cache_directory(root, lock)
    ensure_generated_path(directory / "sentinel")
    directory_fd = _open_cache_directory(directory, create=True)
    try:
        statuses = []
        for record in lock["files"]:
            status = _fetch_file(record, directory_fd, transport, restart,
                                 source_urls[record["role"]])
            statuses.append({"role": record["role"], "filename": record["filename"],
                             "status": status, "source": source_kind})
    finally:
        os.close(directory_fd)
    verification = verify_downloads(lock, root)
    return {"schema_version": 1, "lock_hash": lock["lock_hash"],
            "dataset": lock["dataset"], "snapshot_id": lock["snapshot"]["id"],
            "cache_directory": str(directory), "files": statuses,
            "verified_files": verification["verified_files"]}


def _read_local_bytes_at(directory_fd, filename, limit):
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb") as stream:
            value = os.fstat(stream.fileno())
            _validate_cache_file_stat(value, "local integrity metadata")
            if value.st_size > limit:
                raise AcquisitionError("local integrity metadata exceeds its size limit")
            content = stream.read(limit + 1)
    except OSError as error:
        raise AcquisitionError("cannot read local metadata file: %s" % error)
    if len(content) > limit:
        raise AcquisitionError("local integrity metadata exceeds its size limit")
    return content


def _verify_checksum_relationship(lock, directory_fd):
    records = {value["role"]: value for value in lock["files"]}
    content = _read_local_bytes_at(
        directory_fd, records["checksum"]["filename"], MAX_CHECKSUM_BYTES)
    archive = records["archive"]
    if _SPECS[lock["dataset"]]["checksum_mode"] == "sidecar":
        declared = _parse_direct_checksum(content)
    else:
        declared = _parse_sha256sums(content, archive["filename"])
    if declared != archive["sha256"]:
        raise AcquisitionError("downloaded checksum file contradicts the acquisition lock")


def verify_downloads(lock, root):
    lock = validate_lock(lock)
    directory = cache_directory(root, lock)
    directory_fd = _open_cache_directory(directory)
    try:
        for record in lock["files"]:
            _verify_file_at(directory_fd, record)
        _verify_checksum_relationship(lock, directory_fd)
    finally:
        os.close(directory_fd)
    return {"schema_version": 1, "lock_hash": lock["lock_hash"],
            "dataset": lock["dataset"], "snapshot_id": lock["snapshot"]["id"],
            "cache_directory": str(directory),
            "verified_files": len(lock["files"]),
            "pgp_signature_available": lock["signing_fingerprint"] is not None}


def verify_musicbrainz_signature(lock, root, keyring, gpgv="gpgv"):
    """Verify MusicBrainz SHA256SUMS with a caller-trusted dedicated keyring."""
    lock = validate_lock(lock)
    if lock["signing_fingerprint"] is None:
        raise AcquisitionError("this dataset has no documented PGP signature recipe")
    verify_downloads(lock, root)
    keyring_path = Path(keyring).expanduser().resolve()
    if _regular_file_size(keyring_path) is None:
        raise AcquisitionError("PGP keyring does not exist")
    directory = cache_directory(root, lock)
    records = {value["role"]: value for value in lock["files"]}
    checksum = directory / records["checksum"]["filename"]
    signature = directory / records["checksum_signature"]["filename"]
    try:
        result = subprocess.run(
            [gpgv, "--no-default-keyring", "--keyring", str(keyring_path),
             "--status-fd", "1",
             str(signature), str(checksum)], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
    except OSError as error:
        raise AcquisitionError("cannot run gpgv: %s" % error)
    signatures = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[:2] == ["[GNUPG:]", "VALIDSIG"]:
            signing = fields[2].upper()
            if re.fullmatch(r"[0-9A-F]{40}", signing) is None:
                continue
            primary = fields[-1].upper()
            if re.fullmatch(r"[0-9A-F]{40}", primary) is None:
                primary = signing
            signatures.append((signing, primary))
    if (result.returncode != 0 or len(signatures) != 1 or
            signatures[0][1] != lock["signing_fingerprint"]):
        raise AcquisitionError("MusicBrainz checksum signature verification failed")
    return {"schema_version": 1, "lock_hash": lock["lock_hash"],
            "signed_file": records["checksum"]["filename"],
            "signing_fingerprint": signatures[0][0],
            "primary_fingerprint": signatures[0][1], "pgp_verified": True}
