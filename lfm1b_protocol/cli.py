import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Sequence

from .artifacts import (ArtifactIntegrityError, candidate_rows_for_policy,
                        load_protocol_artifact, prepare_protocol_artifact,
                        save_graph_input, save_protocol_artifact)
from .io import LFM_LISTENING_EVENT_COLUMNS, read_listening_events
from .metrics import evaluate_rankings


def _source_provenance(path: str, has_header: bool) -> Dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {
        "bytes": size,
        "column_order": LFM_LISTENING_EVENT_COLUMNS,
        "has_header": has_header,
        "path": os.path.abspath(path),
        "sha256": digest.hexdigest(),
    }


def _prepare(args: argparse.Namespace) -> int:
    provenance = _source_provenance(args.input, args.header)
    protocol = prepare_protocol_artifact(
        read_listening_events(args.input, args.header), args.sampled_negatives,
        args.seed, args.catalog_policy, sources=(provenance,))
    save_protocol_artifact(args.output, protocol)
    print(json.dumps({"config_hash": protocol["config_hash"],
                      "protocol_hash": protocol["protocol_hash"]}, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    protocol = load_protocol_artifact(
        args.artifact, args.expected_protocol_hash, args.expected_config_hash)
    print("verified protocol %s" % protocol["protocol_hash"])
    return 0


def _graph_input(args: argparse.Namespace) -> int:
    protocol = load_protocol_artifact(args.artifact)
    graph_input = save_graph_input(args.output, protocol)
    print(json.dumps({"node_counts": graph_input["node_counts"],
                      "protocol_hash": protocol["protocol_hash"]}, sort_keys=True))
    return 0


def _baseline(args: argparse.Namespace) -> int:
    protocol = load_protocol_artifact(args.artifact)
    target = protocol["targets"][args.item_type]
    candidate_rows = candidate_rows_for_policy(
        protocol, args.item_type, args.split, args.policy)
    positives = {row["user_id"]: tuple(row["positive_item_ids"])
                 for row in candidate_rows}
    if not positives:
        raise ValueError("no positives for requested split and item type")
    popularity = Counter()  # type: Counter
    for row in target["splits"]["train"]:
        popularity[row["item_id"]] += row["play_count"]
    catalog = target["catalog"]
    scores = {}
    for row in candidate_rows:
        scores[row["user_id"]] = [
            (item, float(popularity[item])) for item in row["candidate_item_ids"]]
    result = evaluate_rankings(scores, positives, args.k, catalog, popularity)
    output = dict(result.__dict__)
    output.update({"item_type": args.item_type, "policy": args.policy,
                   "protocol_hash": protocol["protocol_hash"],
                   "split": args.split})
    print(json.dumps(output, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lfm1b-protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="prepare a protocol bundle")
    prepare.add_argument("input", help="listening-event TSV")
    prepare.add_argument("output", help="output bundle directory")
    prepare.add_argument("--header", action="store_true")
    prepare.add_argument("--sampled-negatives", type=int, default=1000)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--catalog-policy",
                         choices=("train_observed", "all_mapped"),
                         default="train_observed")
    prepare.set_defaults(action=_prepare)
    verify = subparsers.add_parser("verify", help="verify bundle integrity")
    verify.add_argument("artifact", help="bundle directory")
    verify.add_argument("--expected-protocol-hash")
    verify.add_argument("--expected-config-hash")
    verify.set_defaults(action=_verify)
    graph_input = subparsers.add_parser(
        "graph-input", help="export interaction-only thesis graph mappings")
    graph_input.add_argument("artifact", help="bundle directory")
    graph_input.add_argument("output", help="output graph JSON")
    graph_input.set_defaults(action=_graph_input)
    baseline = subparsers.add_parser("baseline", help="evaluate training popularity")
    baseline.add_argument("artifact", help="prepared bundle directory")
    baseline.add_argument("--item-type", choices=("artist", "album", "track"),
                          default="artist")
    baseline.add_argument("--split", choices=("validation", "test"), default="test")
    baseline.add_argument("--policy", choices=("full_catalog", "fixed_sampled"),
                          default="full_catalog")
    baseline.add_argument("-k", type=int, default=10)
    baseline.set_defaults(action=_baseline)
    return parser


def main(argv: Sequence[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.action(args)
    except (ArtifactIntegrityError, OSError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    main()
