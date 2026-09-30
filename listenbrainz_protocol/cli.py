import argparse
import json

from .core import (evaluate_baseline, graph_input, load, load_graph, prepare,
                   save, save_graph)


def _split_options(parser):
    parser.add_argument("--split-strategy", required=True,
                        choices=("per_user_last_timestamp_groups", "global_time_cutoffs"))
    parser.add_argument("--validation-cutoff", type=int)
    parser.add_argument("--test-cutoff", type=int)
    parser.add_argument("--repeat-policy", choices=("novel_only", "repeat_allowed"),
                        default="novel_only")
    parser.add_argument("--positive-filter-horizon", choices=("as_of_split", "all_observed"),
                        default="as_of_split")
    parser.add_argument("--catalog-policy", choices=("train_observed", "all_mapped"),
                        default="train_observed")
    parser.add_argument("--sampled-negatives", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)


def _artifact_hashes(wrapper):
    return {"artifact_hash": wrapper["artifact_hash"],
            "protocol_hash": wrapper["protocol"]["protocol_hash"],
            "config_hash": wrapper["config_hash"],
            "protocol_config_hash": wrapper["protocol"]["config_hash"]}


def build_parser():
    parser = argparse.ArgumentParser(prog="listenbrainz-protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("input")
    prepare_parser.add_argument("output")
    prepare_parser.add_argument("--dump-id", required=True)
    prepare_parser.add_argument("--dump-type", required=True, choices=("full", "incremental"))
    prepare_parser.add_argument("--archive-sha256", required=True)
    prepare_parser.add_argument("--member-path", required=True)
    prepare_parser.add_argument("--track-identity-policy", required=True,
                                choices=("server_mapped_musicbrainz", "recording_msid"))
    prepare_parser.add_argument("--album-entity", required=True,
                                choices=("release_group", "release"))
    prepare_parser.add_argument("--include-local-path", action="store_true")
    _split_options(prepare_parser)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("artifact")
    verify_parser.add_argument("--expected-wrapper-hash")
    verify_parser.add_argument("--expected-protocol-hash")
    verify_parser.add_argument("--expected-config-hash")

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("artifact")
    baseline_parser.add_argument("--model", required=True,
                                 choices=("random", "popularity", "itemknn"))
    baseline_parser.add_argument("--target", default="track",
                                 choices=("artist", "album", "track"))
    baseline_parser.add_argument("--split", default="test",
                                 choices=("validation", "test"))
    baseline_parser.add_argument("--candidate-policy", default="full_catalog",
                                 choices=("fixed_sampled", "full_catalog"))
    baseline_parser.add_argument("-k", type=int, default=10)
    baseline_parser.add_argument("--random-seed", type=int, default=0)
    baseline_parser.add_argument("--neighbor-limit", type=int, default=50)

    graph_parser = subparsers.add_parser("graph-input")
    graph_parser.add_argument("artifact")
    graph_parser.add_argument("output")

    graph_verify_parser = subparsers.add_parser("graph-verify")
    graph_verify_parser.add_argument("artifact")
    graph_verify_parser.add_argument("graph")
    graph_verify_parser.add_argument("--expected-wrapper-hash")
    graph_verify_parser.add_argument("--expected-protocol-hash")
    graph_verify_parser.add_argument("--expected-config-hash")
    graph_verify_parser.add_argument("--expected-graph-hash")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            wrapper = prepare(
                args.input, dump_id=args.dump_id, dump_type=args.dump_type,
                archive_sha256=args.archive_sha256, member_path=args.member_path,
                track_identity_policy=args.track_identity_policy,
                album_entity=args.album_entity, include_local_path=args.include_local_path,
                split_strategy=args.split_strategy, validation_cutoff=args.validation_cutoff,
                test_cutoff=args.test_cutoff, repeat_policy=args.repeat_policy,
                positive_filter_horizon=args.positive_filter_horizon,
                catalog_policy=args.catalog_policy, sampled_negatives=args.sampled_negatives,
                seed=args.seed)
            save(args.output, wrapper)
            output = _artifact_hashes(wrapper)
        elif args.command == "verify":
            wrapper = load(args.artifact, args.expected_wrapper_hash,
                           args.expected_protocol_hash, args.expected_config_hash)
            output = _artifact_hashes(wrapper)
        elif args.command == "baseline":
            output = evaluate_baseline(
                load(args.artifact), model=args.model, target=args.target,
                split=args.split, candidate_policy=args.candidate_policy, k=args.k,
                random_seed=args.random_seed, neighbor_limit=args.neighbor_limit)
        elif args.command == "graph-input":
            wrapper = load(args.artifact)
            graph = graph_input(wrapper)
            save_graph(args.output, wrapper)
            output = dict(_artifact_hashes(wrapper), graph_hash=graph["graph_hash"])
        else:
            wrapper = load(args.artifact, args.expected_wrapper_hash,
                           args.expected_protocol_hash, args.expected_config_hash)
            graph = load_graph(args.graph, wrapper, args.expected_graph_hash)
            output = dict(_artifact_hashes(wrapper), graph_hash=graph["graph_hash"])
        print(json.dumps(output, sort_keys=True))
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    main()
