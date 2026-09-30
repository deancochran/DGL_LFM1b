"""Reproducible acquisition logic; downloaded bytes remain outside Git."""

from .acquisition import (AcquisitionError, available_datasets, discover,
                          fetch, load_lock, save_lock, validate_lock,
                          listenbrainz_protocol_inputs, verify_downloads,
                          verify_musicbrainz_signature)
from .hygiene import (RepositoryHygieneError, assert_repository_hygiene,
                      atomic_write_generated, ensure_generated_path,
                      open_generated_directory, tracked_hygiene_violations)

__all__ = [
    "AcquisitionError", "RepositoryHygieneError", "available_datasets",
    "assert_repository_hygiene", "atomic_write_generated", "discover",
    "ensure_generated_path", "fetch",
    "listenbrainz_protocol_inputs", "load_lock", "save_lock",
    "open_generated_directory", "tracked_hygiene_violations", "validate_lock", "verify_downloads",
    "verify_musicbrainz_signature",
]
