"""Command-line interface for local, bounded protocol comparisons."""

import argparse
import json

from .adapters import artifact_summary, case_from_artifact
from .contrasts import (contrast_pair, create_contrast_analysis,
                        load_contrast_analysis, result_summaries,
                        save_contrast_analysis)
from .matrix import (create_contrast_analysis_from_matrix_plan,
                     create_mask_matrix_spec, create_matrix_comparison_plan,
                     load_mask_matrix_spec, load_matrix_contrast_plan,
                     load_matrix_index, matrix_preview, prepare_mask_matrix,
                     resolve_matrix_contrast_plan, save_mask_matrix_spec,
                     save_matrix_contrast_plan, save_matrix_index,
                     validate_matrix_output_paths)
from .plan import (ADAPTERS, CANDIDATE_POLICIES, MODELS, SPLITS, TARGETS,
                    create_plan, load_plan, save_plan)
from .runner import load_report, run_plan, save_report


def _repeat(parser, flag, choices, help_text):
    parser.add_argument(flag, action="append", choices=choices, help=help_text)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="protocol-compare",
        description=("Plan, run, and analyze bounded comparisons over local "
                     "protocol artifacts"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="validate a local artifact and print its self-attested hashes")
    inspect_parser.add_argument("adapter", choices=ADAPTERS)
    inspect_parser.add_argument("artifact")

    init_parser = subparsers.add_parser(
        "init", help="create an ignored/external comparison plan from local artifacts")
    init_parser.add_argument("output")
    init_parser.add_argument(
        "--artifact", action="append", nargs=4, required=True,
        metavar=("ADAPTER", "DATASET", "CASE", "PATH"),
        help="local artifact descriptor; repeat for multiple datasets")
    _repeat(init_parser, "--target", TARGETS, "target to include; default: track")
    _repeat(init_parser, "--split", SPLITS, "split to include; default: test")
    _repeat(init_parser, "--candidate-policy", CANDIDATE_POLICIES,
            "candidate policy to include; default: full_catalog")
    _repeat(init_parser, "--model", MODELS,
            "baseline to include; default: random, popularity, and itemknn")
    init_parser.add_argument("-k", "--k", action="append", type=int,
                             dest="k_values", help="ranking cutoff; default: 10")
    init_parser.add_argument("--random-seed", action="append", type=int,
                             help="random baseline seed; default: 0")
    init_parser.add_argument("--neighbor-limit", type=int, default=50)
    init_parser.add_argument("--max-candidate-scores", type=int, default=1000000)
    init_parser.add_argument("--max-itemknn-pairs", type=int, default=1000000)
    init_parser.add_argument("--max-results", type=int, default=1000)
    init_parser.add_argument("--max-contribution-rows", type=int, default=100000)
    init_parser.add_argument("--max-ranking-items", type=int, default=5000000)
    init_parser.add_argument("--max-report-bytes", type=int, default=67108864)

    validate_parser = subparsers.add_parser(
        "validate-plan", help="validate and summarize a comparison plan")
    validate_parser.add_argument("plan")
    validate_parser.add_argument("--expected-plan-hash")

    run_parser = subparsers.add_parser(
        "run", help="execute a pinned plan without network access")
    run_parser.add_argument("plan")
    run_parser.add_argument("output")
    run_parser.add_argument("--expected-plan-hash", required=True)

    report_parser = subparsers.add_parser(
        "verify-report", help="verify a saved report against an external hash")
    report_parser.add_argument("report")
    report_parser.add_argument("--expected-report-hash", required=True)

    list_parser = subparsers.add_parser(
        "list-results", help="list result hashes available for explicit contrasts")
    list_parser.add_argument("report")
    list_parser.add_argument("--expected-report-hash", required=True)

    contrast_parser = subparsers.add_parser(
        "contrast", help="create paired descriptive contrasts from a report")
    contrast_parser.add_argument("report")
    contrast_parser.add_argument("output")
    contrast_parser.add_argument("--expected-report-hash", required=True)
    contrast_parser.add_argument(
        "--pair", action="append", nargs=4, required=True,
        metavar=("NAME", "KIND", "BASELINE_HASH", "COMPARISON_HASH"),
        help="explicit model, seed, or mask contrast; repeat as needed")
    contrast_parser.add_argument("--max-contrasts", type=int, default=100)
    contrast_parser.add_argument("--max-paired-rows", type=int, default=100000)
    contrast_parser.add_argument("--max-analysis-bytes", type=int,
                                 default=67108864)

    verify_contrast_parser = subparsers.add_parser(
        "verify-contrast",
        help="recompute a contrast artifact from its externally pinned report")
    verify_contrast_parser.add_argument("report")
    verify_contrast_parser.add_argument("analysis")
    verify_contrast_parser.add_argument("--expected-report-hash", required=True)
    verify_contrast_parser.add_argument("--expected-analysis-hash", required=True)

    matrix_init = subparsers.add_parser(
        "matrix-init", help="create a bounded local mask-matrix specification")
    matrix_init.add_argument("output")
    matrix_init.add_argument("--name", required=True)
    matrix_init.add_argument("--dataset", required=True)
    matrix_init.add_argument("--adapter", required=True, choices=ADAPTERS)
    matrix_init.add_argument("--source", required=True)
    matrix_init.add_argument("--header", action="store_true")
    matrix_init.add_argument("--dump-id")
    matrix_init.add_argument("--dump-type", choices=("full", "incremental"))
    matrix_init.add_argument("--archive-sha256")
    matrix_init.add_argument("--member-path")
    matrix_init.add_argument(
        "--track-identity-policy",
        choices=("server_mapped_musicbrainz", "recording_msid"))
    matrix_init.add_argument("--album-entity", choices=("release_group", "release"))
    matrix_init.add_argument("--split-name", default="primary")
    matrix_init.add_argument("--split-strategy", required=True,
                             choices=("per_user_last_timestamp_groups",
                                      "global_time_cutoffs"))
    matrix_init.add_argument("--validation-cutoff", type=int)
    matrix_init.add_argument("--test-cutoff", type=int)
    _repeat(matrix_init, "--repeat-policy",
            ("novel_only", "repeat_allowed"),
            "repeat-policy axis; default: both")
    _repeat(matrix_init, "--positive-filter-horizon",
            ("as_of_split", "all_observed"),
            "candidate-filter horizon axis; default: both")
    _repeat(matrix_init, "--catalog-policy",
            ("train_observed", "all_mapped"),
            "catalog-policy axis; default: both")
    _repeat(matrix_init, "--target", TARGETS, "evaluation target; default: track")
    _repeat(matrix_init, "--split", SPLITS,
            "evaluation split; default: validation and test")
    _repeat(matrix_init, "--candidate-policy", CANDIDATE_POLICIES,
            "evaluation candidate policy; default: full_catalog")
    _repeat(matrix_init, "--model", MODELS,
            "evaluation model; default: random, popularity, and itemknn")
    matrix_init.add_argument("-k", "--k", action="append", type=int,
                             dest="k_values")
    matrix_init.add_argument("--random-seed", action="append", type=int)
    matrix_init.add_argument("--neighbor-limit", type=int, default=50)
    matrix_init.add_argument("--sampled-negatives", type=int, default=1000)
    matrix_init.add_argument("--preparation-seed", type=int, default=0)
    matrix_init.add_argument("--max-variants", type=int, default=32)
    matrix_init.add_argument("--max-source-bytes", type=int, default=268435456)
    matrix_init.add_argument("--max-events", type=int, default=1000000)
    matrix_init.add_argument("--max-users", type=int, default=100000)
    matrix_init.add_argument("--max-items-per-target", type=int, default=1000000)
    matrix_init.add_argument("--max-preparation-candidate-rows", type=int,
                             default=100000)
    matrix_init.add_argument("--max-preparation-candidate-items", type=int,
                             default=5000000)
    matrix_init.add_argument("--max-artifact-bytes", type=int,
                             default=134217728)
    matrix_init.add_argument("--max-total-artifact-bytes", type=int,
                             default=536870912)
    matrix_init.add_argument("--max-candidate-scores", type=int, default=1000000)
    matrix_init.add_argument("--max-itemknn-pairs", type=int, default=1000000)
    matrix_init.add_argument("--max-results", type=int, default=1000)
    matrix_init.add_argument("--max-contribution-rows", type=int, default=100000)
    matrix_init.add_argument("--max-ranking-items", type=int, default=5000000)
    matrix_init.add_argument("--max-report-bytes", type=int, default=67108864)
    matrix_init.add_argument("--max-contrasts", type=int, default=100)
    matrix_init.add_argument("--max-paired-rows", type=int, default=100000)
    matrix_init.add_argument("--max-analysis-bytes", type=int, default=67108864)

    matrix_preview_parser = subparsers.add_parser(
        "matrix-preview", help="validate and preview matrix expansion without reading data")
    matrix_preview_parser.add_argument("spec")
    matrix_preview_parser.add_argument("--expected-spec-hash", required=True)

    matrix_prepare_parser = subparsers.add_parser(
        "matrix-prepare", help="prepare local matrix artifacts and their pinned index")
    matrix_prepare_parser.add_argument("spec")
    matrix_prepare_parser.add_argument("artifact_root")
    matrix_prepare_parser.add_argument("index")
    matrix_prepare_parser.add_argument("--expected-spec-hash", required=True)

    matrix_plan_parser = subparsers.add_parser(
        "matrix-plan", help="create a comparison plan from a matrix artifact index")
    matrix_plan_parser.add_argument("spec")
    matrix_plan_parser.add_argument("index")
    matrix_plan_parser.add_argument("output")
    matrix_plan_parser.add_argument("--expected-spec-hash", required=True)
    matrix_plan_parser.add_argument("--expected-index-hash", required=True)

    matrix_resolve_parser = subparsers.add_parser(
        "matrix-resolve-contrasts",
        help="resolve one-axis matrix selectors to report result hashes")
    matrix_resolve_parser.add_argument("spec")
    matrix_resolve_parser.add_argument("index")
    matrix_resolve_parser.add_argument("plan")
    matrix_resolve_parser.add_argument("report")
    matrix_resolve_parser.add_argument("output")
    matrix_resolve_parser.add_argument("--expected-spec-hash", required=True)
    matrix_resolve_parser.add_argument("--expected-index-hash", required=True)
    matrix_resolve_parser.add_argument("--expected-plan-hash", required=True)
    matrix_resolve_parser.add_argument("--expected-report-hash", required=True)

    matrix_contrast_parser = subparsers.add_parser(
        "matrix-contrast", help="execute a fully verified resolved matrix contrast plan")
    matrix_contrast_parser.add_argument("spec")
    matrix_contrast_parser.add_argument("index")
    matrix_contrast_parser.add_argument("plan")
    matrix_contrast_parser.add_argument("report")
    matrix_contrast_parser.add_argument("contrast_plan")
    matrix_contrast_parser.add_argument("output")
    matrix_contrast_parser.add_argument("--expected-spec-hash", required=True)
    matrix_contrast_parser.add_argument("--expected-index-hash", required=True)
    matrix_contrast_parser.add_argument("--expected-plan-hash", required=True)
    matrix_contrast_parser.add_argument("--expected-report-hash", required=True)
    matrix_contrast_parser.add_argument("--expected-contrast-plan-hash", required=True)
    return parser


def _init(args):
    targets = args.target or ("track",)
    splits = args.split or ("test",)
    candidate_policies = args.candidate_policy or ("full_catalog",)
    models = args.model or MODELS
    k_values = args.k_values or (10,)
    random_seeds = args.random_seed or (0,)
    cases = []
    for adapter, dataset, name, path in args.artifact:
        cases.append(case_from_artifact(
            name, dataset, adapter, path, targets=targets, splits=splits,
            candidate_policies=candidate_policies, models=models,
            k_values=k_values, random_seeds=random_seeds,
            neighbor_limit=args.neighbor_limit))
    plan = create_plan(cases, args.max_candidate_scores, args.max_itemknn_pairs,
                       args.max_results, args.max_contribution_rows,
                       args.max_ranking_items, args.max_report_bytes)
    save_plan(args.output, plan)
    return {"cases": len(plan["cases"]), "plan_hash": plan["plan_hash"]}


def _matrix_init(args):
    listenbrainz_values = (args.dump_id, args.dump_type, args.archive_sha256,
                           args.member_path, args.track_identity_policy,
                           args.album_entity)
    if args.adapter == "lfm1b":
        if any(value is not None for value in listenbrainz_values):
            raise ValueError("ListenBrainz source fields are invalid for an LFM matrix")
        source_config = {"has_header": args.header}
    else:
        if args.header:
            raise ValueError("--header is valid only for an LFM matrix")
        if any(value is None for value in listenbrainz_values):
            raise ValueError("all ListenBrainz source fields are required")
        source_config = {
            "dump_id": args.dump_id,
            "dump_type": args.dump_type,
            "archive_sha256": args.archive_sha256,
            "member_path": args.member_path,
            "track_identity_policy": args.track_identity_policy,
            "album_entity": args.album_entity,
        }
    split = {"name": args.split_name, "strategy": args.split_strategy,
             "validation_cutoff": args.validation_cutoff,
             "test_cutoff": args.test_cutoff}
    limits = {
        "max_variants": args.max_variants,
        "max_source_bytes": args.max_source_bytes,
        "max_events": args.max_events,
        "max_users": args.max_users,
        "max_items_per_target": args.max_items_per_target,
        "max_preparation_candidate_rows": args.max_preparation_candidate_rows,
        "max_preparation_candidate_items": args.max_preparation_candidate_items,
        "max_artifact_bytes": args.max_artifact_bytes,
        "max_total_artifact_bytes": args.max_total_artifact_bytes,
        "max_candidate_scores": args.max_candidate_scores,
        "max_itemknn_pairs": args.max_itemknn_pairs,
        "max_results": args.max_results,
        "max_contribution_rows": args.max_contribution_rows,
        "max_ranking_items": args.max_ranking_items,
        "max_report_bytes": args.max_report_bytes,
        "max_contrasts": args.max_contrasts,
        "max_paired_rows": args.max_paired_rows,
        "max_analysis_bytes": args.max_analysis_bytes,
    }
    spec = create_mask_matrix_spec(
        args.name, args.dataset, args.adapter, args.source, source_config, split,
        repeat_policies=args.repeat_policy or ("novel_only", "repeat_allowed"),
        positive_filter_horizons=(args.positive_filter_horizon or
                                  ("as_of_split", "all_observed")),
        catalog_policies=args.catalog_policy or ("train_observed", "all_mapped"),
        targets=args.target or ("track",),
        splits=args.split or ("validation", "test"),
        candidate_policies=args.candidate_policy or ("full_catalog",),
        models=args.model or MODELS, k_values=args.k_values or (10,),
        random_seeds=args.random_seed or (0,), neighbor_limit=args.neighbor_limit,
        sampled_negatives=args.sampled_negatives, seed=args.preparation_seed,
        limits=limits)
    save_mask_matrix_spec(args.output, spec)
    output = matrix_preview(spec)
    return output


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            output = artifact_summary(args.adapter, args.artifact)
            output["authenticity"] = "self_attested_until_hashes_are_recorded_externally"
        elif args.command == "init":
            output = _init(args)
        elif args.command == "matrix-init":
            output = _matrix_init(args)
        elif args.command == "matrix-preview":
            spec = load_mask_matrix_spec(args.spec, args.expected_spec_hash)
            output = matrix_preview(spec, args.expected_spec_hash)
        elif args.command == "matrix-prepare":
            spec = load_mask_matrix_spec(args.spec, args.expected_spec_hash)
            validate_matrix_output_paths(args.index, args.artifact_root)
            index = prepare_mask_matrix(spec, args.artifact_root,
                                        args.expected_spec_hash)
            save_matrix_index(args.index, index, spec)
            output = {"spec_hash": spec["spec_hash"],
                      "index_hash": index["index_hash"],
                      "artifacts": len(index["entries"])}
        elif args.command == "matrix-plan":
            spec = load_mask_matrix_spec(args.spec, args.expected_spec_hash)
            index = load_matrix_index(
                args.index, spec, args.expected_index_hash,
                args.expected_spec_hash)
            plan = create_matrix_comparison_plan(
                spec, index, args.expected_spec_hash, args.expected_index_hash)
            save_plan(args.output, plan)
            output = {"spec_hash": spec["spec_hash"],
                      "index_hash": index["index_hash"],
                      "plan_hash": plan["plan_hash"],
                      "cases": len(plan["cases"])}
        elif args.command == "matrix-resolve-contrasts":
            spec = load_mask_matrix_spec(args.spec, args.expected_spec_hash)
            index = load_matrix_index(
                args.index, spec, args.expected_index_hash,
                args.expected_spec_hash)
            plan = load_plan(args.plan, args.expected_plan_hash)
            report = load_report(args.report, args.expected_report_hash)
            contrast_plan = resolve_matrix_contrast_plan(
                spec, index, plan, report, args.expected_spec_hash,
                args.expected_index_hash, args.expected_plan_hash,
                args.expected_report_hash)
            save_matrix_contrast_plan(
                args.output, contrast_plan, spec, index, plan, report)
            output = {"contrast_plan_hash": contrast_plan["contrast_plan_hash"],
                      "input_report_hash": contrast_plan["input_report_hash"],
                      "contrasts": len(contrast_plan["pairs"])}
        elif args.command == "matrix-contrast":
            spec = load_mask_matrix_spec(args.spec, args.expected_spec_hash)
            index = load_matrix_index(
                args.index, spec, args.expected_index_hash,
                args.expected_spec_hash)
            plan = load_plan(args.plan, args.expected_plan_hash)
            report = load_report(args.report, args.expected_report_hash)
            contrast_plan = load_matrix_contrast_plan(
                args.contrast_plan, spec, index, plan, report,
                args.expected_contrast_plan_hash, args.expected_report_hash)
            analysis = create_contrast_analysis_from_matrix_plan(
                contrast_plan, spec, index, plan, report,
                args.expected_contrast_plan_hash, args.expected_report_hash)
            save_contrast_analysis(args.output, analysis, report)
            output = {"analysis_hash": analysis["analysis_hash"],
                      "contrast_plan_hash": contrast_plan["contrast_plan_hash"],
                      "contrasts": len(analysis["contrasts"])}
        elif args.command == "validate-plan":
            plan = load_plan(args.plan, args.expected_plan_hash)
            output = {"cases": len(plan["cases"]), "plan_hash": plan["plan_hash"]}
        elif args.command == "run":
            plan = load_plan(args.plan, args.expected_plan_hash)
            report = run_plan(plan, args.expected_plan_hash)
            save_report(args.output, report)
            output = {"report_hash": report["report_hash"],
                      "results": len(report["results"])}
        elif args.command == "verify-report":
            report = load_report(args.report, args.expected_report_hash)
            output = {"plan_hash": report["plan_hash"],
                      "report_hash": report["report_hash"],
                      "results": len(report["results"])}
        elif args.command == "list-results":
            report = load_report(args.report, args.expected_report_hash)
            output = {"report_hash": report["report_hash"],
                      "results": result_summaries(report)}
        elif args.command == "contrast":
            report = load_report(args.report, args.expected_report_hash)
            pairs = [contrast_pair(*value) for value in args.pair]
            analysis = create_contrast_analysis(
                report, pairs, args.expected_report_hash,
                args.max_contrasts, args.max_paired_rows,
                args.max_analysis_bytes)
            save_contrast_analysis(args.output, analysis, report)
            output = {"analysis_hash": analysis["analysis_hash"],
                      "contrasts": len(analysis["contrasts"]),
                      "input_report_hash": analysis["input_report_hash"]}
        else:
            report = load_report(args.report, args.expected_report_hash)
            analysis = load_contrast_analysis(
                args.analysis, report, args.expected_analysis_hash,
                args.expected_report_hash)
            output = {"analysis_hash": analysis["analysis_hash"],
                      "contrasts": len(analysis["contrasts"]),
                      "input_report_hash": analysis["input_report_hash"]}
        print(json.dumps(output, sort_keys=True))
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    main()
