# Thesis evaluation audit

## Scope

This note records a methodological review of the 2022 thesis
[*Heterogeneous Graph Neural Network Music Recommendation*](https://doi.org/10.5281/zenodo.7116043)
and its associated historical implementation. It distinguishes the published
experiment from the corrected protocol now available in this repository.

This is a code-and-method audit, not a rerun. The full LFM-1b corpus is no longer
available from its historical official download because of licensing, and this
repository has neither a lawful copy nor a complete current model-training
pipeline. The repository now also contains a bounded-subset ListenBrainz adapter
for an independent replication; that does not convert new-dataset results into
an exact reproduction of the thesis tables.

## Interpretation of the historical results

The thesis tables should not be interpreted as unbiased estimates of temporal,
full-catalog recommendation performance. The main threats are:

1. **Held-out graph leakage.** The historical path constructs the heterogeneous
   graph before downstream edge splitting. A held-out interaction can therefore
   affect topology, embeddings, or message passing before it is scored.
2. **Task mismatch.** Sampled binary edge classification is not the same
   estimand as ranking every eligible catalog item for a user. Results depend on
   the sampled-negative distribution and cannot be compared directly with
   full-catalog Recall/NDCG/MRR/MAP.
3. **Metric-definition ambiguity.** Historical code and report labels do not
   consistently establish standard recommender denominators and candidate-set
   semantics. Metric names alone are insufficient evidence of equivalence.
4. **Incomplete result provenance.** The reported tables do not fully pin raw
   input identity, split policy, candidate policy, seeds, and executable
   artifacts needed to reconstruct each number.
5. **Limited uncertainty evidence.** Small or uncontrolled subsets and the lack
   of repeated-seed uncertainty estimates prevent strong comparative claims.

These issues do not show that a particular model has no value. They mean the
published measurements cannot support the stronger claim that one model will
rank future music better under a leakage-controlled catalog recommendation
task.

## What protocol v2 corrects

The separate `lfm1b_protocol` package now provides:

- explicit global or user-relative temporal splitting with ties preserved;
- a globally causal interaction configuration with no future-conditioned user
  inclusion;
- target-specific warm-start eligibility and auditable cold-start strata;
- explicit novel-only versus repeat-inclusive tasks;
- task-aware full and deterministic sampled candidate construction;
- strict ranking metrics, including conventional Precision@K = hits / K;
- train-only graph-input interaction edges;
- identity-bearing configuration, raw aggregates, candidates, and source
  provenance; and
- semantic validators and deterministic regression/property tests.

The phrase "globally causal" is deliberately narrow. It applies only to
`global_time_cutoffs + train_observed + as_of_split` and only to the interaction
protocol. It does not certify external metadata, features, normalization,
negative training samples, model selection, or message passing.

## What remains before a defensible thesis rerun

This repository still does not provide an end-to-end modern thesis experiment.
A rerun requires at least:

1. lawful access to the raw event and metadata files;
2. predeclared global cutoffs and target/repeat estimands;
3. graph construction from training information only;
4. temporally valid metadata/features and training negatives;
5. validation-only model and hyperparameter selection, followed by one final
   test evaluation;
6. full-catalog primary metrics, with sampled metrics labeled as diagnostics;
7. competitive non-neural and popularity baselines;
8. multiple seeds or resamples with uncertainty intervals;
9. pinned protocol/config/source hashes and machine-readable score artifacts;
   and
10. sensitivity analyses for user/item cold start, repeat policy, cutoff choice,
    and any transductive assumptions.

The current `graph-input` export is interaction-only. It has no metadata edges,
DGL adapter, training loop, checkpoint protocol, arbitrary-model score ingestion
CLI, or confidence-interval pipeline. Those omissions must remain explicit in
any new claim.

## Independent replication on ListenBrainz

The unavailability of LFM-1b blocks an exact rerun, but it does not block testing
the broader research question on another population. `listenbrainz_protocol`
normalizes a pinned local `.listens` member, pseudonymizes usernames, records
mapping/missingness/duplicate diagnostics, embeds protocol v2, evaluates three
reference baselines, and exports a wrapper-bound interaction graph.

The proposed headline replication is global-cutoff, repeat-inclusive,
train-observed track ranking with as-of-split filtering and full-catalog NDCG@10.
See [`dataset-migration.md`](dataset-migration.md) for the data decision and
[`benchmark-plan.md`](benchmark-plan.md) for the preregistration and model
ladder.

Before that study can make empirical claims, it still needs full-dump-capable
ETL, a pinned real dump and cohort, real-format mapping-coverage validation,
temporally declared MusicBrainz enrichment, modern model implementations, and
paired uncertainty analysis. ListenBrainz and LFM-1b results must be labeled as
different datasets and cannot be compared as if only the model changed.

## Current verification boundary

The combined corrected preparation code was locally checked with 66 checked-in
tests under Python 3.11 and 3.14. The suite includes 10 focused ListenBrainz
tests, a reproducible randomized 1,920-configuration policy matrix,
hand-calculated split/candidate/metric expectations, and
rehash-after-tampering checks. This establishes expected behavior for synthetic
and subset inputs; it does not reproduce the thesis tables, establish model
performance on LFM-1b, or establish compatibility with every real ListenBrainz
dump variant.
