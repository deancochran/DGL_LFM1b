# Protocol v2

`lfm1b_protocol` prepares reproducible temporal ranking-evaluation artifacts. It
does not claim that an arbitrary downstream model, feature pipeline, or graph is
leakage-free. Listening-event input order is exactly:

```text
user_id, artist_id, album_id, track_id, timestamp
```

## Required split design

The CLI, `temporal_split`, and `prepare_protocol_artifact` require an explicit
strategy.

### `global_time_cutoffs`

Given `validation_cutoff = v` and `test_cutoff = t`, where `v < t`:

- train: timestamp `< v`
- validation: `v <= timestamp < t`
- test: timestamp `>= t`

Every event is assigned from its timestamp alone. Users are **not** required to
appear in every window: train-only users remain in training, and validation and
test cohorts are determined independently. Consequently, adding or removing a
future event cannot change an existing pre-cutoff event's partition.

### `per_user_last_timestamp_groups`

For each user with at least three distinct timestamps, the final timestamp group
is test, the penultimate group is validation, and all earlier groups are train.
Timestamp ties remain together across artist, album, and track projections.
Users with fewer than three groups are excluded from split windows but remain in
all-observed identity catalogs and known-positive records.

This strategy is user-relative: different users can have training events later
than other users' test events. It must not be described as globally
time-causal.

## Repeat task

- `novel_only`: remove a validation pair seen in train and a test pair seen in
  train or validation. This estimates ranking of new user-item pairs.
- `repeat_allowed`: retain those pairs. This estimates future listens where
  repeated consumption is part of the target.

`positive_snapshots` retain raw pair membership for each window. `raw_splits`
retain the corresponding deterministic pair aggregates (`play_count`, first
timestamp, and last timestamp). Artifact validation reconstructs retained split
rows from these identity-bearing raw aggregates.

## Catalog and held-out eligibility

`train_observed` is the warm-start policy used by the primary globally temporal
configuration. For each target relation independently:

1. its catalog contains only items in that target's raw training window;
2. a held-out user must have training history for that same target; and
3. a held-out item must belong to that target's training catalog.

Validation eligibility does not require a test event, and test eligibility does
not require a validation event. Exclusion statistics report deterministic user
IDs and positive-pair counts in three mutually exclusive strata:

- no same-target training history only;
- unseen training-catalog item only; and
- both conditions.

`all_mapped` is explicitly transductive. Its catalogs use every non-missing ID in
the supplied stream, including IDs first observed after the training cutoff, and
it may retain relation-cold users/items.

## Candidate semantics

Candidates always contain all current held-out positives. The exclusion set then
depends on the task and horizon:

| Task | `as_of_split` | `all_observed` |
| --- | --- | --- |
| `novel_only`, validation | mask train positives; do not inspect test | also mask test positives (oracle) |
| `novel_only`, test | mask train and validation positives | same; no later window exists |
| `repeat_allowed`, validation | keep historical items as competitors | mask only truly future-only test items |
| `repeat_allowed`, test | keep train and validation items as competitors | same; no later window exists |

Current positives are restored after masking. Therefore `repeat_allowed` does
not collapse to an artificially easy singleton merely because the user consumed
other catalog items previously. `all_observed` is an oracle, future-aware
sensitivity analysis and must not be presented as an online estimate.

`full_catalog` and `fixed_sampled` use exactly the same exclusion semantics.
Full-catalog ranking is the primary estimate. Fixed sampling is without
replacement and selects the lowest stable SHA256 priorities under
`sha256_priority_v1`; sampled and full-catalog metrics are not interchangeable.

## Causality metadata

The identity-bearing config records:

- `temporal_scope`: `global` or `user_relative`;
- `catalog_uses_future_information`;
- `candidate_filter_uses_future_information`; and
- `globally_time_causal`.

`globally_time_causal` is true exactly for:

```text
global_time_cutoffs + train_observed + as_of_split
```

It is false for per-user splitting, `all_mapped`, or `all_observed`. This flag
describes the interaction protocol only; downstream features and message passing
still require their own temporal controls.

## Ranking metrics

Scores must cover exactly the declared candidate IDs. Ranking is descending by
score with ascending item ID as the deterministic tie-breaker. Metrics are
macro-averaged over evaluated users:

- Recall@K: hits divided by the user's number of relevant candidates;
- Precision@K: hits divided by **K**, even when fewer than K candidates exist;
- binary NDCG@K;
- MRR@K;
- MAP@K with denominator `min(number of relevant items, K)`;
- hit-rate@K;
- catalog coverage; and
- novelty from smoothed training popularity.

When fewer than K candidates exist, absent ranks are treated as nonrelevant for
Precision@K. Candidate policy and task configuration must accompany every
reported result.

## Provenance and integrity

CLI preparation uses a dedicated one-shot parser-bound stream. A source record
is published only after the stream is completely and successfully consumed and
contains:

- `provenance_binding = parser_bound`;
- source kind and parser/format version;
- parsed row count;
- exact byte count and lowercase SHA256;
- header flag; and
- canonical LFM column order.

Generic source records supplied to the library are normalized to
`self_attested`; they cannot claim parser binding. Direct `Event` input without a
source produces an empty source list and is explicitly unprovenanced. Source
paths are aligned one-for-one with source records but excluded from protocol
identity so moving an identical file does not change its hash.

Bundle, config, protocol, candidate, and raw-aggregate validators detect
corruption and internal contradictions. Graph validation checks schema and
internal shape only; a standalone graph export is not cryptographically bound to
the protocol artifact named by its `protocol_hash`. Self-contained hashes do not
prove authenticity against an actor who can rewrite an artifact and all hashes.
Pin `protocol_hash` and `config_hash` externally, and separately hash the graph
export, when authenticity matters.

## Graph and scalability boundary

Graph-input schema v2 contains contiguous mappings and aggregated **train-only**
user--artist, user--album, and user--track edges. `static_edges` is empty.
Consumers remain responsible for metadata enrichment, temporal feature
construction, message-passing leakage controls, model training, and inference.

The standard-library implementation groups events and protocol state in memory
and targets synthetic data and subsets, not direct processing of the billion-row
corpus. Fixed sampling scans each relevant catalog and uses bounded priority
selection. Full-catalog derivation and scoring inherently cost
O(users x catalog).

## Verification evidence

For this protocol-v2 correction:

- 55 checked-in unit, integration, and property tests passed locally under
  Python 3.11 and 3.14;
- a hand-calculated validation candidate matrix covers both repeat policies and
  both horizons, with separate repeat-inclusive test-window checks;
- exact metric checks include the `candidate_count < K` case;
- rehashed semantic-tampering tests cover raw aggregates, temporal metadata,
  user eligibility, candidates, and cold-start strata, while separate
  graph-schema tests cover mappings; and
- the checked-in randomized 1,920-configuration cross-product passed for both split
  strategies, both repeat policies, both catalog policies, both horizons, two
  sample sizes, and two seeds.

This evidence verifies implementation invariants on synthetic inputs. It is not
a model-quality result or a rerun on the unavailable full LFM-1b corpus.
