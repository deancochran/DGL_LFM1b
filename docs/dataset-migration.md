# Dataset migration: LFM-1b to ListenBrainz

## Status and terminology

The historical LFM-1b loader remains in this repository for archival use. The
[official LFM-1b page](https://www.cp.jku.at/datasets/LFM-1b/) now says that the
dataset is **not available for download anymore due to license issues**. The
[LFM-2b page](https://www.cp.jku.at/datasets/LFM-2b/) says the same. This project
therefore calls those datasets *historical* or *unavailable from the official
source*, rather than claiming that their research questions or code have been
deprecated.

The open-data successor selected here is:

1. [ListenBrainz listens dumps](https://listenbrainz.readthedocs.io/en/latest/users/listenbrainz-dumps.html)
   for timestamped user--recording interactions; and
2. [MusicBrainz dumps](https://metabrainz.org/datasets/postgres-dumps#musicbrainz)
   for stable recording, artist, release, and release-group identities and,
   in a later enrichment stage, metadata relations.

This pairing supports an independent replication of the thesis research
question. It is not an exact replacement for LFM-1b and cannot reproduce the
published LFM-1b numbers.

## Why ListenBrainz plus MusicBrainz

As checked on 2026-09-05, MetaBrainz documents ListenBrainz dumps as CC0,
commercially usable, and containing hundreds of millions of listens. Full dumps
are published twice a month and incremental dumps are also available. The core
listens archive contains one JSON document per line in year/month `.listens`
members. MusicBrainz core and JSON metadata are also CC0; some supplementary
MusicBrainz dumps use CC BY-NC-SA, so each enrichment input must record its own
license rather than inherit the core-data assertion.

The combination is the strongest fit among the reviewed public research data
because it has:

- event timestamps rather than only aggregate counts or playlist membership;
- recording identifiers that can be connected to a maintained music catalog;
- recurring snapshots that can be pinned for provenance;
- an interaction signal close to the original listening-history task; and
- terms that permit broader reuse than noncommercial-only alternatives.

Important differences from LFM-1b remain. ListenBrainz users self-select into a
different service and time period, mapped-MBID coverage is incomplete, the
available metadata and demographics differ, and ListenBrainz's mapping may have
been produced after the listen. Report results as a new-dataset replication, not
as a continuation of the old result table.

## Dynamic acquisition and retention boundary

The standard-library `research-data` CLI separates metadata discovery from
payload acquisition. It supports four independently meaningful products:

- `listenbrainz-sample` and `listenbrainz-full`;
- `musicbrainz-canonical`; and
- `musicbrainz-core` (`mbdump.tar.bz2` plus signed checksum metadata).

`discover` parses the current official directory index, resolves an exact
snapshot directory, obtains each payload's byte size, and validates the
publisher SHA256 relationship before writing a canonical lock. `fetch` requires
an externally pinned lock hash, refuses downloads above an explicit byte budget,
supports strict HTTP range resume, and verifies bytes before atomically exposing
the final filename. `verify` rehashes the local cache. For MusicBrainz core,
`verify-pgp` additionally runs `gpgv` with a dedicated caller-supplied keyring
and requires the documented primary-key fingerprint
`D5E63B4BDCCE195642948684B8FC2375C777580F`.

This establishes content identity, not permanent availability. On 2026-09-05,
the official ListenBrainz full index exposed only two current full payloads. A
published experiment must therefore preserve, outside this repository:

1. the resolved lock JSON and its separately recorded `lock_hash`;
2. every file named by the lock, under immutable retention;
3. any PGP verification receipt and the trusted-key acquisition procedure;
4. selected archive-member paths and hashes; and
5. the resulting wrapper/protocol hashes.

The CLI can read an immutable HTTPS mirror selected through
`--mirror-base-url-env`; optional Authorization material is read from a second
environment variable and is never written to output. Mirror bytes are accepted
only when they match the publisher hash already frozen in the lock. Use
institutional object storage or a durable DVC/DataLad/git-annex remote for the
bytes. This repository intentionally keeps only reusable acquisition logic, not
locks, data, artifact pointers, or generated receipts.

Standalone incremental dumps are deliberately excluded. An incremental state
is not one self-contained dataset: it requires a pinned full base, every
contiguous incremental in order, and a tested policy for deleted listens. Until
that chain assembler exists, use an exact full snapshot.

`protocol-inputs` translates a verified ListenBrainz lock into the dump ID,
archive filename, and archive SHA256 expected by the local adapter. It does not
extract the archive. Extraction/full-scale assembly remains a separate ETL
boundary: reject absolute and parent-traversal members, extract only selected
`.listens` files into ignored/external storage, and hash each extracted member.

## Dataset comparison

The table compares suitability for this project's timestamped music-ranking
question. Size figures are publisher descriptions, not measurements made by
this repository.

| Dataset | Interaction and time signal | Publisher scale/freshness | Identity/metadata fit | Access or license status | Role here |
| --- | --- | --- | --- | --- | --- |
| **ListenBrainz + MusicBrainz** | Individual timestamped listens | ListenBrainz: hundreds of millions of listens, recurring full snapshots; MusicBrainz: recurring catalog snapshots | Server-provided MBID mappings connect recordings, artists, releases, and release groups | ListenBrainz and MusicBrainz core/JSON documented as CC0; commercial support is strongly encouraged | **Primary successor** |
| **LFM-1b** | More than one billion timestamped Last.fm listens | Static 2016 research release | Original user/artist/album/track graph and demographics | Official page says no longer downloadable due license issues | Historical reference only; lawful local copies can use `lfm1b_protocol` |
| **LFM-2b** | 2,014,164,872 timestamped listens; aggregate variants also existed | 120,322 users in the former full release | Track, album, artist, user, tags, lyrics, and demographic files formerly offered | Official page says no longer downloadable due license issues | Historical comparison only |
| **MLHD+** | More than 27 billion timestamped Last.fm logs, cleaned and canonicalized against MusicBrainz | Static derived release | Strong MusicBrainz mapping | Publisher says noncommercial use only and lists no open license | Large-scale sensitivity option only when its terms fit the study |
| **HetRec 2011 Last.fm** | Aggregate artist-listening records, not event timestamps | 92,800 records from 1,892 users | Artist, tag, and social information | Publisher directs users to the included README for terms | Small smoke/cross-study dataset, not a scale successor |
| **Spotify Million Playlist Dataset** | Ordered playlist membership, not longitudinal user listens | 1 million playlists, over 2 million tracks, nearly 300,000 artists | Spotify catalog IDs and playlist context | Ongoing noncommercial open-research access | Playlist-continuation comparison, not the headline estimand |
| **Amazon Reviews 2023: Digital Music** | Timestamped ratings/reviews, not listening events | 130,400 reviews, 101,000 users, 70,500 items | Product metadata and review text, not a recording graph | Check the publisher's current terms for a planned use | Cross-domain explicit-feedback control only |

Metrics from these datasets are not directly comparable unless interaction
definition, cohort, split, catalog, candidates, and metric implementation are
made equivalent. Literature numbers are context, not baselines for a new run.

## Implemented local adapter

`listenbrainz_protocol` is a Python-standard-library adapter for protocol tests
and bounded subsets. It performs no network request and does not extract an
archive. The caller provides one already-extracted `.listens` JSONL member and
records:

- the full/incremental dump identifier and type;
- a caller-computed SHA256 for the containing archive;
- the relative archive-member path;
- the exact member byte count and parser-computed SHA256; and
- the parser version and accepted/dropped-row statistics.

The archive digest is marked `self_attested`: storing a supplied digest binds it
to the artifact but does not prove that it came from MetaBrainz. Verify official
checksums or signatures separately when available, and preserve that evidence
with the experiment record.

### Explicit normalization choices

`--track-identity-policy` is required:

- `server_mapped_musicbrainz` keeps only listens with a server-returned
  `mbid_mapping.recording_mbid` and emits
  `musicbrainz:recording:<canonical-uuid>`; or
- `recording_msid` uses `recording_msid` and emits
  `listenbrainz:recording-msid:<canonical-uuid>`.

The phrase *server mapped* says only where the selected field was returned. It
does not prove whether the mapping originated from submitted metadata, a manual
mapping, or a fuzzy matcher.

`--album-entity` is also required and selects either release-group or release
MBIDs. When multiple artist MBIDs are returned, the interaction uses the first
server-returned credit and reports the row in `multi_artist_rows`. Missing
artist/album mappings remain nullable. Missing selected track identities are
dropped and counted.

Duplicate `(pseudonymous user, timestamp, selected recording)` event keys are
retained. The wrapper reports duplicate rows and keys whose artist/album values
conflict; it does not silently select one event.

### Privacy boundary

Raw usernames are never written to the wrapper. Each accepted username becomes:

```text
sha256(b"listenbrainz:user:v1\0" + username.encode("utf-8"))
```

This deterministic digest permits longitudinal joins and is therefore
**pseudonymization, not anonymization**. Listening histories can remain
sensitive. Minimize retained data, do not attempt re-identification, restrict
artifact access, and apply the research institution's ethics/privacy review.
The optional local path is excluded from artifact identity and emitted only
with `--include-local-path`.

## Synthetic workflow

The checked-in fixture exercises the documented JSONL shape without containing
real user data. Its archive digest below is deliberately a synthetic placeholder:

```bash
python -m listenbrainz_protocol.cli prepare \
  examples/listenbrainz_synthetic.listens /tmp/listenbrainz-protocol \
  --dump-id synthetic \
  --dump-type full \
  --archive-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --member-path synthetic/member.listens \
  --track-identity-policy server_mapped_musicbrainz \
  --album-entity release_group \
  --split-strategy global_time_cutoffs \
  --validation-cutoff 20 \
  --test-cutoff 30 \
  --repeat-policy repeat_allowed \
  --catalog-policy all_mapped \
  --sampled-negatives 3 \
  --seed 0

python -m listenbrainz_protocol.cli verify /tmp/listenbrainz-protocol
python -m listenbrainz_protocol.cli baseline /tmp/listenbrainz-protocol \
  --model popularity --target track --candidate-policy full_catalog -k 10
python -m listenbrainz_protocol.cli graph-input \
  /tmp/listenbrainz-protocol /tmp/listenbrainz-graph
python -m listenbrainz_protocol.cli graph-verify \
  /tmp/listenbrainz-protocol /tmp/listenbrainz-graph
```

`prepare` and `verify` report wrapper, wrapper-config, embedded-protocol, and
embedded-protocol-config hashes. Record those values outside the bundle and use
the `verify --expected-*-hash` options when integrity must be checked against a
trusted experiment record. `graph-verify` can additionally pin its graph hash.

## Delivery stages and limitations

| Stage | Status | Boundary |
| --- | --- | --- |
| Preserve historical LFM loader | Complete | Historical behavior is unchanged; it is not presented as leakage-controlled |
| Correct temporal/candidate protocol | Complete | Protocol v2 is shared after dataset-specific normalization |
| ListenBrainz JSONL normalization and wrapper | Complete for synthetic data and bounded subsets | One local extracted member, in-memory processing |
| Dynamic public-snapshot acquisition | Implemented and unit-tested for locked sample/full and MusicBrainz canonical/core downloads | Logic only; caller must externally retain locks and bytes because publisher retention is finite |
| Bounded cross-adapter comparison runner | Implemented for prepared LFM/ListenBrainz artifacts | Shared random, popularity, and ItemKNN baselines with pinned plans, per-user contributions, and fail-closed workload limits |
| Paired descriptive contrast analyzer | Implemented for bounded comparison reports | Explicit model/seed/mask result pairs, normalized-population checks, native and common-user cohorts, no inferential confidence intervals |
| Bounded mask-matrix orchestrator | Implemented for local LFM/ListenBrainz subset inputs | Canonical spec, separately prepared artifacts, pinned index, comparison plan, and one-axis resolved contrast plan; no network or full-dump scalability claim |
| Random, popularity, and binary ItemKNN baselines | Complete for protocol artifacts | Diagnostic/reference models, not thesis reproduction |
| Wrapper-bound train interaction graph | Complete | No MusicBrainz metadata edges yet |
| Full-dump ingestion | Planned | Requires streaming/distributed ETL, deletion-aware snapshot handling, and real-format coverage tests |
| MusicBrainz enrichment | Planned | Must pin a compatible snapshot and declare temporal/transductive semantics per relation |
| Modern model training and uncertainty pipeline | Planned | See [`benchmark-plan.md`](benchmark-plan.md) |

The current package should not be described as capable of loading an unbounded
full ListenBrainz dump. Full-catalog candidate derivation and scoring also have
inherent user-by-catalog cost.

The 104-test standard-library suite passes under Python 3.11 and 3.14, including
15 focused acquisition tests, 17 comparison tests, and 10 ListenBrainz tests
for discovery, integrity, filesystem safety, parsing, identity,
per-user and global splitting, rehashed semantic tampering, baseline scores,
bundle/graph binding, CLI flow, and nonidentity local-path provenance. Six
additional matrix tests cover staged local preparation and planning across both
adapters. The 41 Python files also parse with the Python 3.8 grammar. This is
synthetic protocol evidence, not real-dump compatibility or model-quality
evidence.

## Sources

- MetaBrainz, [database dumps and licenses](https://metabrainz.org/datasets/postgres-dumps)
- MusicBrainz, [core dump files, signatures, fingerprint, and licenses](https://musicbrainz.org/doc/MusicBrainz_Database/Download)
- ListenBrainz, [data dump structure](https://listenbrainz.readthedocs.io/en/latest/users/listenbrainz-dumps.html)
- MetaBrainz, [canonical MusicBrainz and MLHD+ derived dumps](https://metabrainz.org/datasets/derived-dumps)
- JKU, [LFM-1b status](https://www.cp.jku.at/datasets/LFM-1b/)
- JKU, [LFM-2b status and former schema](https://www.cp.jku.at/datasets/LFM-2b/)
- GroupLens, [HetRec 2011](https://grouplens.org/datasets/hetrec-2011/)
- Spotify Research, [Million Playlist Dataset](https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered/)
- McAuley Lab, [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)

License summaries are engineering notes, not legal advice. Recheck publisher
terms at acquisition and publication time.
