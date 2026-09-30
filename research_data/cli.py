import argparse
import json
import os

from .acquisition import (DEFAULT_MAX_DOWNLOAD_BYTES, available_datasets,
                          discover, fetch, load_lock, save_lock,
                          listenbrainz_protocol_inputs, verify_downloads,
                          verify_musicbrainz_signature)
from .hygiene import assert_repository_hygiene


def build_parser():
    parser = argparse.ArgumentParser(
        prog="research-data",
        description="Discover, lock, fetch, and verify external research data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list supported public data products")

    discover_parser = subparsers.add_parser(
        "discover", help="resolve official metadata into an ignored local lock")
    discover_parser.add_argument("dataset", choices=tuple(
        value["dataset"] for value in available_datasets()))
    discover_parser.add_argument("lock")
    discover_parser.add_argument("--snapshot", default="latest",
                                 help="exact dated snapshot ID or latest")

    show_parser = subparsers.add_parser("show", help="validate and print a lock")
    show_parser.add_argument("lock")
    show_parser.add_argument("--expected-lock-hash")

    inputs_parser = subparsers.add_parser(
        "protocol-inputs", help="print locked arguments for ListenBrainz prepare")
    inputs_parser.add_argument("lock")
    inputs_parser.add_argument("--expected-lock-hash", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch", help="explicitly download a lock into an ignored cache")
    fetch_parser.add_argument("lock")
    fetch_parser.add_argument("root")
    fetch_parser.add_argument("--expected-lock-hash", required=True)
    fetch_parser.add_argument("--max-bytes", type=int,
                              default=DEFAULT_MAX_DOWNLOAD_BYTES)
    fetch_parser.add_argument("--restart", action="store_true",
                              help="restart partial files instead of resuming")
    fetch_parser.add_argument(
        "--mirror-base-url-env",
        help="environment variable containing a credential-free HTTPS snapshot mirror")
    fetch_parser.add_argument(
        "--mirror-authorization-env",
        help="environment variable containing an Authorization header for the mirror")

    verify_parser = subparsers.add_parser(
        "verify", help="rehash every downloaded file and checksum relationship")
    verify_parser.add_argument("lock")
    verify_parser.add_argument("root")
    verify_parser.add_argument("--expected-lock-hash", required=True)

    pgp_parser = subparsers.add_parser(
        "verify-pgp", help="verify signed MusicBrainz SHA256SUMS with gpgv")
    pgp_parser.add_argument("lock")
    pgp_parser.add_argument("root")
    pgp_parser.add_argument("--keyring", required=True,
                            help="dedicated keyring obtained through a trusted channel")
    pgp_parser.add_argument("--expected-lock-hash", required=True)
    pgp_parser.add_argument("--gpgv", default="gpgv")

    hygiene_parser = subparsers.add_parser(
        "hygiene", help="fail if Git tracks generated data/build output")
    hygiene_parser.add_argument("--root", default=os.getcwd())
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            output = {"datasets": available_datasets()}
        elif args.command == "discover":
            lock = discover(args.dataset, args.snapshot)
            save_lock(args.lock, lock)
            output = lock
        elif args.command == "show":
            output = load_lock(args.lock, args.expected_lock_hash)
        elif args.command == "protocol-inputs":
            output = listenbrainz_protocol_inputs(
                load_lock(args.lock, args.expected_lock_hash))
        elif args.command == "fetch":
            lock = load_lock(args.lock, args.expected_lock_hash)
            mirror = None
            if args.mirror_base_url_env:
                if args.mirror_base_url_env not in os.environ:
                    raise ValueError("mirror URL environment variable is not set")
                mirror = os.environ[args.mirror_base_url_env]
            authorization = None
            if args.mirror_authorization_env:
                if args.mirror_authorization_env not in os.environ:
                    raise ValueError("mirror authorization environment variable is not set")
                authorization = os.environ[args.mirror_authorization_env]
            output = fetch(lock, args.root, args.max_bytes, args.restart,
                           mirror_base_url=mirror,
                           mirror_authorization=authorization)
        elif args.command == "verify":
            lock = load_lock(args.lock, args.expected_lock_hash)
            output = verify_downloads(lock, args.root)
        elif args.command == "verify-pgp":
            lock = load_lock(args.lock, args.expected_lock_hash)
            output = verify_musicbrainz_signature(
                lock, args.root, args.keyring, args.gpgv)
        else:
            assert_repository_hygiene(args.root)
            output = {"repository_hygiene": "passed"}
        print(json.dumps(output, sort_keys=True))
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    main()
