# DGL LFM-1b research bridge

This repository preserves a historical DGL LFM-1b loader, provides a corrected
temporal evaluation protocol, and now includes a local ListenBrainz adapter for
an open-data successor study. The opt-in `research-data` utility can acquire
public MetaBrainz snapshots into ignored or external storage; the repository
does not contain or redistribute those dataset bytes.

## Dataset status and project direction

The [official LFM-1b page](https://www.cp.jku.at/datasets/LFM-1b/) says that the
dataset is no longer available for download due to license issues. LFM-2b is in
the same position. The old loader remains available for researchers who already
have lawful local access, but new experiments should not assume that readers can
obtain those files.

The selected independent-replication path uses timestamped
[ListenBrainz dumps](https://listenbrainz.readthedocs.io/en/latest/users/listenbrainz-dumps.html)
with MusicBrainz identities. MetaBrainz documents ListenBrainz and MusicBrainz
core/JSON dumps as CC0. This is a new population and data-generating process, not
an exact substitute for LFM-1b; new results must not be compared as if the
datasets were interchangeable.

- [Dataset migration, alternatives, licensing, and adapter contract](docs/dataset-migration.md)
- [Model-comparison and benchmark plan](docs/benchmark-plan.md)
- [Formal temporal/candidate protocol](docs/protocol.md)
- [Historical thesis evaluation audit](docs/thesis-evaluation-audit.md)

## Reproducible external-data acquisition

`research_data` dynamically reads the official HTTPS indexes for ListenBrainz
sample/full and MusicBrainz canonical/core snapshots. Discovery resolves
`latest` to an exact snapshot, obtains publisher checksums and byte sizes, and
writes a canonical lock. Fetching is a separate explicit step: it requires the
lock hash supplied through a trusted experiment record, enforces a byte budget,
supports verified resume, and rechecks every file before atomic promotion.

Locks and downloads must be written under an ignored generated root such as
`data/` or `downloads/`, or outside the checkout. Neither belongs in Git:

```bash
python -m research_data.cli list
python -m research_data.cli discover listenbrainz-sample \
  data/locks/listenbrainz-sample.json --snapshot latest

# Copy the lock_hash printed by discover into the experiment record, then pin it.
LOCK_HASH='paste-the-64-character-hash-from-discover'
python -m research_data.cli show data/locks/listenbrainz-sample.json \
  --expected-lock-hash "$LOCK_HASH"
python -m research_data.cli fetch data/locks/listenbrainz-sample.json downloads \
  --expected-lock-hash "$LOCK_HASH" --max-bytes 1073741824
python -m research_data.cli verify data/locks/listenbrainz-sample.json downloads \
  --expected-lock-hash "$LOCK_HASH"
python -m research_data.cli protocol-inputs \
  data/locks/listenbrainz-sample.json --expected-lock-hash "$LOCK_HASH"
python -m research_data.cli hygiene --root .
```

Do not use `latest` as the identity of a published experiment: retain the exact
resolved snapshot ID, lock JSON, and lock hash externally. At the time this
workflow was checked, the official ListenBrainz full index retained only two
recent full archives. Exact later recreation therefore requires copying all
locked files immediately to durable external storage. A mirror must expose the
locked filenames directly beneath one immutable HTTPS base URL:

```bash
export RESEARCH_DATA_MIRROR='https://archive.example.org/listenbrainz/snapshot/'
python -m research_data.cli fetch data/locks/listenbrainz-sample.json downloads \
  --expected-lock-hash "$LOCK_HASH" \
  --mirror-base-url-env RESEARCH_DATA_MIRROR
```

For a private mirror, an Authorization header may be injected from a separate
environment variable with `--mirror-authorization-env`; its value is not stored
in the lock or receipt. Use a secret manager and avoid putting credentials in a
URL or shell history. The mirror remains untrusted: downloaded bytes must match
the official hashes recorded in the lock. Institutional object storage is the
preferred durable source; DVC/DataLad/git-annex can manage such a remote, but
their generated pointer metadata is intentionally not part of this logic-only
repository policy.

The default 1-GiB guard accommodates the current sample but deliberately blocks
large full/core dumps until the operator chooses an explicit budget. Standalone
ListenBrainz incrementals are not offered: a reproducible incremental state must
pin a full base and every contiguous update, including deletion semantics.
Archive extraction and full-scale ETL remain caller-managed; inspect member
paths for traversal, extract only intended `.listens` members outside Git, and
record each extracted member hash before invoking `listenbrainz_protocol`.

## ListenBrainz subset workflow

`listenbrainz_protocol` reads one local, already-extracted `.listens` JSONL
member. It performs no network access or archive extraction and is currently an
in-memory implementation for synthetic data and bounded subsets, not an
unbounded full-dump ETL system.

The checked-in synthetic example can be prepared and evaluated with:

    python -m listenbrainz_protocol.cli prepare examples/listenbrainz_synthetic.listens /tmp/listenbrainz-protocol --dump-id synthetic --dump-type full --archive-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --member-path synthetic/member.listens --track-identity-policy server_mapped_musicbrainz --album-entity release_group --split-strategy global_time_cutoffs --validation-cutoff 20 --test-cutoff 30 --repeat-policy repeat_allowed --catalog-policy all_mapped --sampled-negatives 3 --seed 0
    python -m listenbrainz_protocol.cli verify /tmp/listenbrainz-protocol
    python -m listenbrainz_protocol.cli baseline /tmp/listenbrainz-protocol --model random --target track --candidate-policy full_catalog -k 10
    python -m listenbrainz_protocol.cli baseline /tmp/listenbrainz-protocol --model popularity --target track --candidate-policy full_catalog -k 10
    python -m listenbrainz_protocol.cli baseline /tmp/listenbrainz-protocol --model itemknn --target track --candidate-policy full_catalog -k 10
    python -m listenbrainz_protocol.cli graph-input /tmp/listenbrainz-protocol /tmp/listenbrainz-graph
    python -m listenbrainz_protocol.cli graph-verify /tmp/listenbrainz-protocol /tmp/listenbrainz-graph

The synthetic archive digest is intentionally a placeholder. For real local
data, use the acquisition lock to pin the publisher dump ID and archive hash,
then supply the exact extracted relative member path.
`server_mapped_musicbrainz` drops and counts rows without a returned recording
MBID; `recording_msid` keeps MSID-backed recordings instead. Raw usernames are
replaced with deterministic SHA256
pseudonyms, which are **not anonymous identifiers**. Duplicate event keys are
retained and reported.

The wrapper embeds protocol v2 and exposes deterministic random, train-only
popularity, and binary ItemKNN baselines. These are reference implementations,
not a claim that the historical thesis has been rerun. MusicBrainz metadata
edges, scalable dump assembly, modern model training, and uncertainty reporting
remain explicit planned stages.

## Bounded local comparisons

`protocol_comparison` runs the same three reference models over already-prepared
LFM or ListenBrainz protocol artifacts. It performs no discovery, download,
archive extraction, or remote execution. Plans bind expected artifact/config
hashes, masking configuration, targets, splits, candidate policies, model seeds,
and safety limits while treating relocatable local paths as nonidentity
provenance.

After running the two synthetic preparation commands documented above and below,
create and execute a local plan:

```bash
python -m protocol_comparison.cli init /tmp/comparison-plan.json \
  --artifact lfm1b lfm1b lfm-synthetic /tmp/lfm-protocol \
  --artifact listenbrainz listenbrainz-synthetic listenbrainz-synthetic \
    /tmp/listenbrainz-protocol \
  --target track --split test \
  --candidate-policy fixed_sampled --candidate-policy full_catalog \
  --model random --model popularity --model itemknn \
  -k 10 --random-seed 0

# Record the plan_hash printed by init outside the plan before execution.
PLAN_HASH='paste-the-64-character-hash-from-init'
python -m protocol_comparison.cli run /tmp/comparison-plan.json \
  /tmp/comparison-report.json --expected-plan-hash "$PLAN_HASH"

# Record the printed report_hash separately, then verify it when reused.
REPORT_HASH='paste-the-64-character-hash-from-run'
python -m protocol_comparison.cli verify-report /tmp/comparison-report.json \
  --expected-report-hash "$REPORT_HASH"

# List the result hashes that can be selected for an explicit paired contrast.
python -m protocol_comparison.cli list-results /tmp/comparison-report.json \
  --expected-report-hash "$REPORT_HASH"

# Use hashes from list-results. Deltas are COMPARISON_RESULT - BASELINE_RESULT.
BASELINE_RESULT='paste-the-baseline-result-hash'
COMPARISON_RESULT='paste-the-comparison-result-hash'
python -m protocol_comparison.cli contrast /tmp/comparison-report.json \
  /tmp/comparison-contrasts.json --expected-report-hash "$REPORT_HASH" \
  --pair baseline-v-comparison model "$BASELINE_RESULT" "$COMPARISON_RESULT"

# Record the printed analysis_hash separately before verification.
ANALYSIS_HASH='paste-the-64-character-hash-from-contrast'
python -m protocol_comparison.cli verify-contrast \
  /tmp/comparison-report.json /tmp/comparison-contrasts.json \
  --expected-report-hash "$REPORT_HASH" \
  --expected-analysis-hash "$ANALYSIS_HASH"
```

Each result records aggregate metrics, deterministic per-user contributions,
and hit ranks sufficient to verify recall, precision, hit rate, NDCG, MRR, and
MAP. The default plan fails closed above one million total candidate scores, one
million ItemKNN ordered co-occurrence pairs, five million candidate-ranking
items, one hundred thousand contribution rows, one thousand result records, or
a 64-MiB report. These guards make it suitable for synthetic data and bounded
local artifacts; they are not a scalable full-dump implementation.

Before scoring, the runner validates each case once to retain compact bounds
and preflight the entire report, discards that protocol payload, then reloads
and validates it once at the scoring boundary. This bounded two-validation-per-
case design preserves strict external-artifact checks without retaining every
case protocol for the full run.

Report schema v3 also binds a normalized `population_hash` and an explicit
`numerical_semantics` contract. It specifies `math.fsum` reductions with the
platform `libm` implementation of `log2`; exact canonical hashes have been
verified on CPython 3.11 and 3.14 on the same machine, not asserted as a
universal cross-platform floating-point guarantee. Per-recommendation novelty
values let report verification recompute recommendation-weighted novelty
self-consistently, along with recall, precision, hit rate, NDCG, MRR, MAP,
coverage, and paired contrast means. A report alone does not carry the complete
training-popularity table, so a coordinated rehash cannot prove those novelty
values originated from its artifact; the externally retained expected report
hash remains the trust anchor for that claim. Schema v2 reports are intentionally rejected rather than
silently interpreted under v3 arithmetic: regenerate a new report and contrast
analysis from the pinned plan and artifacts; leave historical v2/v5 files
unchanged. For LFM this always
combines the declared source identity with per-target known user--item
fingerprints, including when the source record is self-attested. For
ListenBrainz it binds the dump member identity, normalization/mapping policy,
and complete mapping hash. This allows the analyzer to reject mask comparisons
whose integer IDs do not denote the same normalized population. As elsewhere,
a self-attested source claim is not proof of the source bytes without an
externally retained hash record.

Paired analyses support three explicit kinds. `model` requires the exact same
artifact, candidates, and user cohort but different model families. `seed`
requires two random-baseline runs over that same task. `mask` requires the same
dataset, normalized population, target, split, K, and model under
`full_catalog`, but different mask configurations; it reports native user counts
and deltas only over the user intersection. Every per-user value is
`comparison - baseline`. Native report novelty remains recommendation-weighted,
while the paired novelty entry is the mean of per-user novelty contributions.
The analysis is descriptive: it does not provide confidence intervals or
multiple-comparison correction.

Contrast generation defaults to at most 100 requested pairs, 100,000 total
paired rows, and a 64-MiB artifact. Hard ceilings are 1,000 pairs, one million
paired rows, and 256 MiB. Contrast files bind the input report hash, selectors,
limits, native result metadata, common/excluded cohorts, per-user deltas, and
their own externally recordable analysis hash. Verification recomputes the
entire analysis from the pinned report rather than trusting rehashed summaries.

### Bounded local mask matrices

The matrix workflow prepares independently hashed protocol artifacts for a
bounded Cartesian product of repeat policy, positive-filter horizon, and catalog
policy. It then creates a normal comparison plan and resolves only one-axis
`full_catalog` mask contrasts after that plan has run. It performs no discovery,
download, extraction, remote storage, or remote execution.

This synthetic example varies only catalog policy so every stage remains small:

```bash
python -m protocol_comparison.cli matrix-init /tmp/matrix-spec.json \
  --name synthetic --dataset lfm1b --adapter lfm1b \
  --source examples/tiny_events.dat \
  --split-name global --split-strategy global_time_cutoffs \
  --validation-cutoff 2 --test-cutoff 3 \
  --repeat-policy repeat_allowed \
  --positive-filter-horizon as_of_split \
  --catalog-policy train_observed --catalog-policy all_mapped \
  --target track --split validation --candidate-policy full_catalog \
  --model popularity -k 1 --sampled-negatives 3 \
  --max-candidate-scores 1000 --max-itemknn-pairs 1000

# Record each printed hash outside its generated artifact before the next stage.
SPEC_HASH='paste-spec-hash'
python -m protocol_comparison.cli matrix-preview /tmp/matrix-spec.json \
  --expected-spec-hash "$SPEC_HASH"
python -m protocol_comparison.cli matrix-prepare /tmp/matrix-spec.json \
  /tmp/matrix-artifacts /tmp/matrix-index.json \
  --expected-spec-hash "$SPEC_HASH"

INDEX_HASH='paste-index-hash'
python -m protocol_comparison.cli matrix-plan /tmp/matrix-spec.json \
  /tmp/matrix-index.json /tmp/matrix-plan.json \
  --expected-spec-hash "$SPEC_HASH" --expected-index-hash "$INDEX_HASH"

PLAN_HASH='paste-plan-hash'
python -m protocol_comparison.cli run /tmp/matrix-plan.json \
  /tmp/matrix-report.json --expected-plan-hash "$PLAN_HASH"

REPORT_HASH='paste-report-hash'
python -m protocol_comparison.cli matrix-resolve-contrasts \
  /tmp/matrix-spec.json /tmp/matrix-index.json /tmp/matrix-plan.json \
  /tmp/matrix-report.json /tmp/matrix-contrast-plan.json \
  --expected-spec-hash "$SPEC_HASH" --expected-index-hash "$INDEX_HASH" \
  --expected-plan-hash "$PLAN_HASH" --expected-report-hash "$REPORT_HASH"

CONTRAST_PLAN_HASH='paste-contrast-plan-hash'
python -m protocol_comparison.cli matrix-contrast \
  /tmp/matrix-spec.json /tmp/matrix-index.json /tmp/matrix-plan.json \
  /tmp/matrix-report.json /tmp/matrix-contrast-plan.json \
  /tmp/matrix-analysis.json \
  --expected-spec-hash "$SPEC_HASH" --expected-index-hash "$INDEX_HASH" \
  --expected-plan-hash "$PLAN_HASH" --expected-report-hash "$REPORT_HASH" \
  --expected-contrast-plan-hash "$CONTRAST_PLAN_HASH"
```

Omitting the three axis flags selects both values of each axis, yielding eight
artifacts. `matrix-preview` expands names and checks result/contrast counts
without reading source data. The default matrix limits are 32 variants, 256 MiB
of local source bytes, one million parsed events, 100,000 users, one million
items per target, 100,000 materialized candidate rows, five million materialized
candidate items, 128 MiB per prepared artifact, 512 MiB across artifacts, 1,000
result records, 100 resolved contrasts, 100,000 paired rows, and the comparison
limits described above. Hard ceilings are 64 variants, 4 GiB of source bytes,
10 million events, one million users, five million items per target, one million
candidate rows, 50 million candidate items, 256 MiB per artifact, 4 GiB across
artifacts, and the existing report/contrast ceilings. Sampled-negative
preparation is additionally capped at one million negatives.

The spec and index hashes exclude relocatable source and artifact paths. The
index instead pins every generated artifact/config/protocol/population hash.
Plan creation reloads each artifact and rejects divergence from that index. The
artifact root must be empty: the workflow never deletes, cleans, or silently
reuses stale files, and an interrupted preparation remains visible for manual
inspection. Each variant reparses the same local source and preparation aborts
if its file identity, size, or modification time changes during the run. Source
byte and row limits are also enforced inside the parsers; artifact files are
created exclusively beneath an open, validated root directory. The index must
be written outside that artifact root.

Resolved plans orient `novel_only` to `repeat_allowed`, `as_of_split` to
`all_observed`, and `train_observed` to `all_mapped`. They compare only one axis
at a time while holding target, split, model, seed, and K fixed. Horizon
contrasts on the test split are omitted because test masking is identical by
design. Preparation can still produce a valid artifact whose selected
target/split has no evaluable positives; ordinary comparison execution then
fails closed rather than silently dropping that case.

Mask choices are identity-bearing properties of a prepared protocol artifact.
To compare `novel_only`/`repeat_allowed`, `as_of_split`/`all_observed`, or
`train_observed`/`all_mapped`, prepare one artifact per configuration and add
each as a separate case. Compare deltas within each dataset. Do not pool raw
metrics across LFM and ListenBrainz populations, and do not treat MusicBrainz
metadata snapshots as standalone interaction datasets.

## Protocol-v2 temporal evaluation preparation

`DGL_LFM1b.py`, `data_utils.py`, and `meta_paths.py` are the historical 2021-2022
DGL loader and retain their original behavior. In particular, that loader builds
the graph before downstream edge splitting and is not a temporally isolated
evaluation implementation. The dependency-free `lfm1b_protocol` package is a
separate, reproducible preparation and ranking-evaluation path; it does not
import DGL, PyTorch, pandas, or NumPy. It prepares interaction data and does not
by itself make a downstream model leakage-free.

The CLI and public preparation APIs require an explicit split design. Choose
either `per_user_last_timestamp_groups` (last two timestamp groups are
validation/test, with ties preserved) or `global_time_cutoffs` (train `<`
validation cutoff, validation before test cutoff, test `>=` test cutoff).
Global cutoffs partition every event without requiring a user to occur in every
window. The globally time-causal primary configuration is
`global_time_cutoffs` + `train_observed` + `as_of_split`. Per-user cutoffs are
user-relative rather than globally causal. `novel_only` estimates new-item
ranking; `repeat_allowed` includes future repeats. `all_observed` is an oracle,
future-aware sensitivity analysis. See [`docs/protocol.md`](docs/protocol.md).

The raw listening-event TSV order is:

    user_id, artist_id, album_id, track_id, timestamp

Fields are tab-separated, with no header by default. Empty item IDs map to
`None`; user ID and timestamp are required. A tiny synthetic input is available
at `examples/tiny_events.dat`.

Prepare, verify, and evaluate a training-popularity baseline:

    python -m lfm1b_protocol.cli prepare examples/tiny_events.dat /tmp/lfm-protocol --split-strategy global_time_cutoffs --validation-cutoff 2 --test-cutoff 3 --catalog-policy train_observed --sampled-negatives 1000 --seed 0
    python -m lfm1b_protocol.cli verify /tmp/lfm-protocol
    python -m lfm1b_protocol.cli baseline /tmp/lfm-protocol --item-type artist --policy fixed_sampled -k 10
    python -m lfm1b_protocol.cli baseline /tmp/lfm-protocol --item-type artist --policy full_catalog -k 10
    python -m lfm1b_protocol.cli graph-input /tmp/lfm-protocol /tmp/lfm-graph-input.json

`train_observed` is the default warm-start catalog policy. Each target catalog
contains only items observed in that target's training rows. A validation or
test pair is eligible only when its item is in that catalog and its user has
training history for the same target relation. Artist history therefore cannot
make an artist-cold user warm merely because the user has album or track
history. Validation and test eligibility are independent. Excluded pairs are
reported by target/window and by three mutually exclusive reasons: target-cold
user only, unseen training-catalog item only, or both.

`--catalog-policy all_mapped` is an explicit transductive option. It uses IDs
observed anywhere in the supplied stream, including validation/test, and may
include relation-cold users and items. Here, "mapped" means non-missing IDs in
the supplied event stream; this package does not infer IDs from external
metadata tables.

Candidate policies are `full_catalog` and `fixed_sampled`. Both always include
the current positives and use the same task-aware exclusions. For `novel_only`,
previously consumed items are not competitors. For `repeat_allowed`, historical
items remain eligible competitors; `as_of_split` does not inspect future
windows. During validation, oracle `all_observed` may mask only truly
future-only test items, not historical repeats. Full-catalog evaluation is the
primary estimate; sampled metrics are diagnostics and are not numerically
interchangeable with full-catalog metrics.

Fixed sampling is without replacement and selects the lowest deterministic
SHA256 random-oracle priorities (`sha256_priority_v1`) with bounded selection
rather than a full priority sort. Results must record the candidate policy.
The target-oriented `protocol.json` materializes fixed sampled candidates and
their hashes. Full-catalog rows are derived from the artifact's frozen catalog
and positive-snapshot horizon with `candidate_rows_for_policy`; they are not
materialized as an O(users x items) table. Public integration functions are
`prepare_protocol_artifact`, `save_protocol_artifact`, and
`load_protocol_artifact`.

For authenticity-sensitive verification, pass the 64-character
`protocol_hash` and `config_hash` printed by `prepare` through
`--expected-protocol-hash` and `--expected-config-hash`.
Bundle manifests hash canonical JSON content and contain no wall-clock timestamp;
loading verifies the manifest, file, config, protocol, candidate hashes, raw
window aggregates, retained rows, and declared temporal metadata.
Unpinned `verify` detects corruption and internal inconsistency, but hashes stored
inside the same artifact are not proof against an adversary who can rewrite the
artifact and all of its hashes. Record hashes externally and pass both
`--expected-*-hash` options when authenticity matters. CLI preparation consumes
a one-shot parser-bound stream and records the parser version, parsed row count,
exact byte count, lowercase SHA256, header flag, and LFM column order. Generic
library-supplied source records are marked `self_attested`; direct `Event` input
with no source is explicitly unprovenanced. Source paths are retained only as
non-identity provenance; source bytes/checksums remain identity-bearing.

`graph-input` writes graph-input schema v2: original-to-contiguous mappings and
explicit mapped **train-only** interaction edges with play counts and timestamp
ranges for user--artist, user--album, and user--track. It
uses only retained training users and train-observed target catalogs for the
default warm policy, excluding insufficient-only users and cold/future-only
items. The explicit `all_mapped` policy exports its broader transductive ID
universe. Consumers remain responsible for temporal and leakage controls;
metadata edges require a separate enrichment export. See
[`docs/protocol.md`](docs/protocol.md).

### Scalability scope

The pure standard-library protocol implementations group events and protocol
state in memory. They are intended for synthetic data and subset protocol
validation, not direct preparation of the billion-event LFM-1b corpus or an
unbounded ListenBrainz dump. Fixed candidate
preparation reuses per-target catalog and per-user exclusion indexes and uses
bounded priority selection. Derived full-catalog evaluation still has inherent
O(users x catalog) output and scoring cost.

Run the standard-library test suite with:

    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v

The 103-test combined suite was locally checked under Python 3.11 and 3.14.
Fifteen focused acquisition tests exercise dynamic discovery, strict HTTP
resume, mirrors, signed metadata, and filesystem hygiene. Seventeen comparison tests
exercise both adapters, workload limits, path safety, semantic tampering,
deterministic reports, and paired model/seed/mask contrasts. Ten tests directly
exercise the ListenBrainz wrapper, and five matrix tests exercise both adapters,
the complete staged CLI, relocation, one-axis resolution, and fail-closed
limits/tampering. The checked-in property test includes a
1,920-configuration randomized cross-product over split, repeat, catalog,
horizon, sample-size, and seed choices. All 41 Python files also parse with the
Python 3.8 grammar.
These checks establish the implemented invariants on synthetic data; they are
not a full-corpus model-performance result. See the
[thesis evaluation audit](docs/thesis-evaluation-audit.md) for the distinction
between the historical results and a required rerun.

### Evaluation references

- PyTorch's [Reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html) explain deterministic limitations, and [`inference_mode`](https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html) is the authoritative inference API.
- Current DGL documents [`remove_edges`](https://docs.dgl.ai/generated/dgl.remove_edges.html) and [`as_edge_prediction_sampler`](https://docs.dgl.ai/generated/dgl.dataloading.as_edge_prediction_sampler.html). This project historically used DGL 0.8.2; consult documentation or tagged source for that exact legacy release before translating current examples.
- Rendle, [Evaluation Metrics for Item Recommendation under Sampling](https://arxiv.org/abs/1912.02263), demonstrates why sampled ranking metrics require careful interpretation.
- Meng, McCreadie, Macdonald, and Ounis, [Exploring Data Splitting Strategies for the Evaluation of Recommendation Models](https://doi.org/10.1145/3383313.3418479), analyzes how split design changes conclusions.


## Historical LFM-1b context

LFM-1b contained more than one billion listening events and was intended for
music retrieval and recommendation research. The [dataset paper](https://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_icmr_2016.pdf)
by Schedl was published at ICMR 2016. Its
[official website](https://www.cp.jku.at/datasets/LFM-1b/) now retains the
description and citation but no longer offers the dataset because of license
issues.

In case you make use of the LFM-1b dataset in your own research, please cite the following paper:

    The LFM-1b Dataset for Music Retrieval and Recommendation
    Schedl, M.
    Proceedings of the ACM International Conference on Multimedia Retrieval (ICMR 2016), New York, USA, April 2016.

Additionally, the [paper](http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_ism_mam_2017.pdf) written by Schedl, M. and Ferwerda, B. discussing
the LFM1b User Genre Profile dataset was published in 2017 for ISM. It uses Last.fm artist tags indexed with two dictionaries of genre and style descriptors
(from Allmusic and Freebase) to create, for each user in LFM-1b, a preference profile as a vector over genres.


In case you make use of the LFM-1b UGP dataset in your own research, please cite the following paper:


    Large-scale Analysis of Group-specific Music Genre Taste From Collaborative Tags
    Schedl, M. and Ferwerda, B.
    Proceedings of the 19th IEEE International Symposium on Multimedia (ISM 2017), Taichung, Taiwan, December 2017.


## Historical loader requirements

The historical loader was built with Python 3.8.10 and requires these manually
installed framework versions:

- [torch](https://pytorch.org/) 1.11.0
- [dgl](https://www.dgl.ai/) 0.8.2

The referenced historical `requirements.txt` is not present in this checkout.
The new `lfm1b_protocol`, `listenbrainz_protocol`, `research_data`, and
`protocol_comparison` packages use only the Python standard library.

## Historical graph notes

The official LFM-1b site now says the core dataset is unavailable due to
licensing, and the historical download URL returns 404. Obtain files only
through a lawful independent route, keep them outside this repository, and run
the protocol `prepare` command against those local files. The acquisition
utility supports only the named open MetaBrainz products; it does not acquire or
redistribute LFM dataset files.

The node types of the graph:
- User (120K)
- Artist (3M)
- Album (15M)
- Track (32M)
- Genre (20)

The Edge types of the graph :
- User -> Artist (61411336)
- Artist -> User (61411336)
- User -> Album (na)
- Album -> User (na)
- User -> Track (na)
- Track -> User (na)
- Artist -> Genre (414379)
- Genre -> Artist (414379)
- Album -> Artist (14184326)
- Artist -> Album (14184326)
- Track -> Artist (27258365)
- Artist -> Track (27258365)


Additionally, for all the user edges:

- User -> Artist
- Artist -> User
- User -> Album
- Album -> User
- User -> Track
- Track -> User

There is `norm_connections` edge data indicating the normalized relative
interaction count a source node had with a specified destination artist, album,
or track node. The `norm_connections` edge data for all other edges is 1.

## Compile the historical dataset

The original `python LFM1b.py` command is stale: that filename does not exist,
and `DGL_LFM1b.py` uses package-relative imports. From this repository, the
historical equivalent is:

    PYTHONPATH=.. python -c "from DGL_LFM1b.DGL_LFM1b import LFM1b; LFM1b()"

### **Precursor warning**

I, the author of the repository, am using a Linux Machine with 30GB of RAM and 12GB of GPU.  To run the above script, it will take the machine ~2hrs, and I am unable to store the full knowledge graph in memory


### Compile a subset

To invoke the historical loader for a subset:

    PYTHONPATH=.. python -c "from DGL_LFM1b.DGL_LFM1b import LFM1b; LFM1b(n_users=50)"

This provides a subset of 50 users with their corresponding listen events and
the artists, albums, and tracks associated with their listening habits.


# The DGL Framework

The Deep Graph library ([DGL](https://www.dgl.ai/))  framework provides the ability to utilize the DGLDataset object
to generate a customizeable dataset for the purpose of node/link/graph down stream tasks.

Once the dataset is compiled you may import the class into any file and load the precompiled graph for DGL based analysis.

    from DGL_LFM1b.DGL_LFM1b import LFM1b

    dataset = LFM1b()
    glist, glabels = dataset.load()
    hg=glist[0]
    print(hg)
