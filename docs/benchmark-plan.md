# ListenBrainz recommendation benchmark plan

## Evidence status

This is a preregistration-oriented implementation plan. The repository contains
no real ListenBrainz dump, trained modern-model checkpoint, or empirical model
comparison. A local comparison runner can execute a deterministic random
diagnostic, train-only popularity, and binary ItemKNN over bounded LFM or
ListenBrainz protocol artifacts. Do not turn its synthetic test metrics into
performance claims.

## Research question and primary estimand

The independent replication asks:

> Given a known listener's recording history before a global cutoff, how well
> can a model rank training-observed recordings that the listener plays in a
> later test window?

The proposed confirmatory task is track recommendation with:

- `global_time_cutoffs`;
- `repeat_allowed`, because repeated music consumption is part of the intended
  behavior rather than a nuisance to remove;
- `train_observed`, so every scored recording and evaluated user has same-target
  training support;
- `as_of_split`, so validation candidate filtering does not inspect test data;
- `server_mapped_musicbrainz` recording identities;
- every distinct eligible user--recording pair in the held-out window as a
  positive; and
- exact `full_catalog` ranking with NDCG@10 as the primary metric.

This estimand deliberately differs from novel-item discovery. Report a
`novel_only` run as a named sensitivity analysis, not pooled with the headline
result. A one-first-event-per-user or fixed-horizon next-item task would be a
different estimand and requires a new protocol version before it can replace the
one above.

## Freeze before inspecting results

Fill and publish this record before model tuning:

| Decision | Required value |
| --- | --- |
| ListenBrainz full dump ID, acquisition lock hash, and official index | TBD |
| Downloaded archive SHA256, verification method, and durable mirror receipt | TBD |
| Included member paths and exact member hashes | TBD |
| Observation start/end | TBD |
| Validation cutoff | TBD |
| Test cutoff | TBD |
| User-history/activity thresholds | TBD; compute from training only |
| Mapping coverage threshold and exclusion policy | TBD |
| MusicBrainz snapshot ID/hash for enriched runs | TBD |
| Primary K and metric | 10 and NDCG@10 unless amended before results |
| Stochastic seeds | At least five, fixed in advance |
| Hyperparameter budget per model family | TBD, equalized and recorded |
| Hardware/runtime budget | TBD |

The test window ends at the pinned input snapshot's declared observation end;
the protocol's `timestamp >= test_cutoff` rule does not itself impose an upper
bound. Assemble the input window before preparation and hash that assembly.

Do not require a user to have a future event in order to place the user's earlier
events in training. Evaluate only users with eligible positives in the requested
held-out window, and report target-specific cold-start exclusions separately.

Resolve a dated snapshot with `research-data discover`, record the printed lock
hash outside Git, fetch with that hash pinned, and run `research-data verify`
before extraction. Because the publisher retains only a small rolling set of
full ListenBrainz archives, copy every locked file to immutable external storage
before analysis. A later URL that returns 404 does not invalidate a hash, but it
does make the experiment irreproducible unless the pinned bytes were retained.
Do not publish `latest` as a dataset identity. Do not model an incremental as a
standalone snapshot; a future incremental workflow must prove a full base plus a
contiguous, deletion-aware chain.

## Candidate and metric contract

Every compared model must score exactly the candidate IDs returned by the same
protocol artifact. It may not silently add model-specific negatives, remove
hard items, or use a different mapping.

- `full_catalog` is the confirmatory policy.
- `fixed_sampled` is a compute/debug diagnostic and must be labeled with sample
  size, seed, and `sha256_priority_v1` algorithm.
- Validation selects models and hyperparameters; the test split is evaluated
  once after selection.
- Ranking uses score descending and item ID ascending for ties.
- Primary: macro NDCG@10.
- Secondary: Recall@10, Precision@10, MRR@10, MAP@10, hit rate, catalog
  coverage, and popularity-smoothed novelty.
- Also report users, catalog size, positives per user, candidate counts,
  train/validation/test time ranges, mapping coverage, and every exclusion
  stratum. A metric without its denominator/cohort is incomplete.

Metrics under sampled candidates, different datasets, different repeat tasks,
or different catalogs are not numerically interchangeable.

## Model comparison ladder

All models receive the same training interactions and candidates. Side-data
models add only the explicitly pinned MusicBrainz snapshot; they must also be
compared with an interaction-only capacity control.

| Model | Purpose | Inputs | Status |
| --- | --- | --- | --- |
| Deterministic random | Pipeline and metric diagnostic, not a competitive recommender | Candidate IDs plus fixed seed | Implemented (`sha256_priority_v1`) |
| MostPopular | Non-personalized lower/reference baseline | Training play counts only | Implemented (`train_play_count_popularity_v1`) |
| ItemKNN | Classical personalized neighborhood baseline | Binary training histories; cosine similarity | Implemented (`binary_cosine_itemknn_v1`) |
| Implicit ALS | Strong latent-factor control for weighted implicit feedback | Training interactions only | Planned |
| BPR-MF | Pairwise-ranking latent-factor control | Training interactions and train-only sampled negatives | Planned |
| LightGCN | Homogeneous interaction-graph neural baseline | Train-only user--recording graph | Planned |
| SASRec | Sequential baseline testing whether order adds value | Timestamp-ordered training sequences only | Planned |
| Side-information control | Tests metadata value without heterogeneous message passing | Same interaction backbone plus frozen MusicBrainz features | Planned |
| R-GCN | Established relation-aware graph baseline | Train-only interactions plus declared MusicBrainz relations | Planned |
| HGT | Attention-based heterogeneous graph baseline | Same relation set and snapshot as R-GCN | Planned |
| Reconstructed thesis model | Historical-method comparison under the corrected protocol | Explicitly documented reconstruction | Planned; must not be called exact without matching code/data |

### Implemented bounded runner

`protocol-compare` creates canonical plans over already-prepared local artifacts.
Every case pins the adapter-specific and embedded protocol/config hashes. Plan
identity excludes relocatable local paths but includes targets, splits,
candidate policies, models, random seeds, K values, and workload limits. Reports
contain canonical result hashes, aggregate metrics, and sorted per-user
contributions; report identity likewise excludes only local path provenance.
Report schema v3 retains the normalized population hash introduced in v2, so separately prepared mask
variants can be paired only when their declared source and normalized user--item
identity agree. LFM includes per-target known-positive fingerprints even for a
self-attested source record; external hash retention remains the authenticity
boundary for such claims.

Schema v3 additionally binds explicit numerical semantics using `math.fsum`
reductions and platform `libm` logarithms, and retains per-recommendation novelty
values. Producer and verifier share arithmetic. Regenerate older reports and
analyses rather than reinterpreting their pinned hashes. Exact cross-runtime
hash agreement has been verified on CPython 3.11 and 3.14 on the same machine;
universal cross-platform bit reproducibility is not claimed.

The same random, popularity, and ItemKNN implementations are used for both
adapters. Candidate rows always come from the validated shared protocol
artifact, and training state uses only that target's training rows. Default
limits cap total candidate scoring, repeated ranking work, per-user contribution
rows, ItemKNN co-occurrence construction, result count, and report bytes, so this
runner is evidence for orchestration and parity on bounded inputs—not a
full-dump execution engine.

A masking comparison requires a separately prepared artifact for each
identity-bearing combination. `all_observed` is useful only as an oracle
validation sensitivity because its test masking equals `as_of_split` by design.
`novel_only` changes the recommendation task and positive cohort, while
`all_mapped` changes the catalog/cohort and uses future information; neither is
merely a harmless implementation toggle. Report within-dataset native cohorts
and common-user paired deltas separately.

The bounded analyzer now accepts explicit result-hash pairs and emits canonical
descriptive `model`, random-`seed`, or `mask` contrasts. Model and seed contrasts
require identical candidate rows and user cohorts. Mask contrasts require the
same normalized population, target, split, model request, K, and `full_catalog`
policy, then preserve native metrics while computing deltas on the common-user
intersection. Every delta is oriented comparison minus baseline. The resulting
artifact binds its input report hash and can be semantically recomputed during
verification; a new internal hash without an externally recorded expected hash
is still self-attested.

The bounded mask-matrix orchestrator now closes the local preparation gap. A
canonical, path-independent spec declares one split regime and selected repeat,
horizon, and catalog axes. Preparation emits one separately validated artifact
per Cartesian variant plus an index that pins every artifact, protocol, config,
and normalized-population identity. The ordinary comparison plan remains the
execution boundary. After execution, a resolved contrast plan selects exact
result hashes for one-axis, full-catalog mask edges and is itself bound to the
spec, index, comparison plan, and report hashes. Test-split horizon edges are
omitted because they are equivalent by construction.

Matrix generation is intentionally subset-scale: source bytes and parsed rows,
user/item cardinality, materialized candidate rows/items, per-artifact and
aggregate artifact bytes, variant count, planned results, resolved contrasts,
paired rows, and report/analysis bytes are bounded. It does not make the
in-memory protocol implementation suitable for a full dump, and it leaves
partial output visible rather than performing destructive cleanup.

Recommended foundational references are ItemKNN (Sarwar et al., 2001), implicit
ALS (Hu, Koren, and Volinsky, 2008), BPR (Rendle et al., 2009), SASRec (Kang and
McAuley, 2018), R-GCN (Schlichtkrull et al., 2018), LightGCN (He et al., 2020),
and HGT (Hu et al., 2020). The historical comparison target is the
[2022 thesis](https://doi.org/10.5281/zenodo.7116043). Pin actual implementations,
versions, and licenses in the experiment manifest rather than relying on model
names alone.

Model references:

- Sarwar et al., [Item-based collaborative filtering recommendation algorithms](https://doi.org/10.1145/371920.372071)
- Hu, Koren, and Volinsky, [Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22)
- Rendle et al., [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)
- Kang and McAuley, [Self-Attentive Sequential Recommendation](https://doi.org/10.1109/ICDM.2018.00035)
- Schlichtkrull et al., [Modeling Relational Data with Graph Convolutional Networks](https://doi.org/10.1007/978-3-319-93417-4_38)
- He et al., [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://doi.org/10.1145/3397271.3401063)
- Hu et al., [Heterogeneous Graph Transformer](https://doi.org/10.1145/3366423.3380027)

## Leakage controls

1. Fit popularity, neighborhoods, embeddings, sequence vocabularies, scalers,
   and all learned features on training rows only.
2. Do not place validation/test interactions in a graph used for message
   passing, degree normalization, or negative sampling.
3. Tune on validation only. Never choose a checkpoint, threshold, epoch, or
   feature set from test metrics.
4. Record whether current ListenBrainz-to-MusicBrainz mappings are treated as
   transductive identity resolution. Do not call them historically point-in-time
   mappings without evidence.
5. For metadata edges, pin one MusicBrainz snapshot and classify every relation
   as static, snapshot-as-of, or future-aware. If a historically valid snapshot
   cannot be reconstructed, label the run transductive.
6. Keep test positives in the candidate set. Never use them as training
   negatives.
7. Do not condition the training partition on future user participation.

## Statistical comparison

- Run each stochastic model with at least five predeclared seeds; deterministic
  models run once unless another source of randomness exists.
- Preserve per-user metric contributions. Deterministic paired descriptive
  model, seed, and mask differences are implemented. Add a paired user-level
  bootstrap confidence interval before making inferential claims.
- Apply the same user bootstrap samples to every model being compared.
- Correct or clearly scope multiple comparisons when making more than one
  confirmatory superiority claim.
- Report failures, timeout/OOM rates, wall time, peak memory, accelerator type,
  parameter count, and inference cost. A model excluded because it exceeded a
  budget remains part of the study record.
- Include effect sizes and intervals; do not infer practical superiority from a
  rounded mean alone.

## Required sensitivity analyses

1. `novel_only` versus `repeat_allowed`.
2. `server_mapped_musicbrainz` versus `recording_msid`, with mapping coverage and
   changed-cohort statistics.
3. At least one earlier/later global cutoff pair.
4. Warm-start headline cohort versus reported user/item cold-start strata.
5. Interaction-only graph versus each MusicBrainz relation family added
   separately.
6. Full catalog versus fixed sampled candidates, explicitly to show how sampling
   changes the metric rather than to substitute for the primary result.
7. User-activity and item-popularity strata.

## Result artifact checklist

Each published row must resolve to machine-readable records containing:

- dataset/dump/member identity and verified hashes;
- acquisition-lock hash, official source index, durable mirror object identity,
  and integrity/PGP verification receipt;
- wrapper, wrapper-config, protocol, protocol-config, graph, code revision, and
  environment hashes;
- complete normalization/split/candidate/model configuration;
- model seed, selected hyperparameters, checkpoint hash, and training budget;
- aggregate and per-user metrics with exact candidate policy;
- cohort, catalog, mapping-coverage, missingness, duplicate, and exclusion
  statistics; and
- a statement of which inputs or relations were transductive/future-aware.

Until those records and real runs exist, documentation should say *planned* or
*implemented*, never *outperforms*, *reproduces*, or *state of the art*.

Reports now supply deterministic per-user contributions and the analyzer
supplies common-user paired descriptive deltas. Paired bootstrap intervals and
multiple-comparison correction remain analysis-stage work rather than
implemented statistical claims.
