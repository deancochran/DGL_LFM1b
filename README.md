# DGL_LFM1b
This repository is a custom DGL datatset created from the LFM-1b database. 
The database downloads and processes the full database to create one singular DGL heterogeneous graph.

## Leakage-free evaluation protocol

`DGL_LFM1b.py`, `data_utils.py`, and `meta_paths.py` are the historical 2021-2022
DGL loader and retain their original behavior. In particular, that loader builds
the graph before downstream edge splitting and is not a leakage-free evaluation
implementation. The dependency-free `lfm1b_protocol` package is a separate,
reproducible preparation and ranking-evaluation path; it does not import DGL,
PyTorch, pandas, or NumPy.

The protocol splits each user on global listening-event timestamp groups. The
latest group is test, the previous group validation, and all earlier groups
train. Timestamp ties remain together for artist, album, and track projections.
Users need at least three timestamp groups. Validation pairs seen in train and
test pairs seen in train or validation are excluded, while all raw positives are
retained for negative filtering. Training repeats are aggregated as play counts
with first and last timestamps.

The raw listening-event TSV order is:

    user_id, artist_id, album_id, track_id, timestamp

Fields are tab-separated, with no header by default. Empty item IDs map to
`None`; user ID and timestamp are required. A tiny synthetic input is available
at `examples/tiny_events.dat`.

Prepare, verify, and evaluate a training-popularity baseline:

    python -m lfm1b_protocol.cli prepare examples/tiny_events.dat /tmp/lfm-protocol --catalog-policy train_observed --sampled-negatives 1000 --seed 0
    python -m lfm1b_protocol.cli verify /tmp/lfm-protocol
    python -m lfm1b_protocol.cli verify /tmp/lfm-protocol --expected-protocol-hash SHA256 --expected-config-hash SHA256
    python -m lfm1b_protocol.cli baseline /tmp/lfm-protocol --item-type artist --policy fixed_sampled -k 10
    python -m lfm1b_protocol.cli baseline /tmp/lfm-protocol --item-type artist --policy full_catalog -k 10
    python -m lfm1b_protocol.cli graph-input /tmp/lfm-protocol /tmp/lfm-graph-input.json

`train_observed` is the default warm-start catalog policy. Each target catalog
contains only items observed in eligible users' training rows. Cold validation
and test pairs are excluded from the primary benchmark and counted, with their
user IDs, in target statistics. All-horizon known positives are still retained
for negative filtering. `--catalog-policy all_mapped` is an explicit
transductive option: it uses item IDs observed in validation/test or otherwise
outside eligible training and therefore assumes future item availability. Here,
"mapped" means non-missing IDs represented in the supplied event stream; this
package does not infer unseen IDs from external metadata tables.

Candidate policies are `full_catalog` and `fixed_sampled`. Both include the
current split positives and remove every other known user positive. Sampling is
without replacement and uses a stable SHA256-derived seed. Results should record
the policy because sampled and full-catalog metrics are not interchangeable.
The target-oriented `protocol.json` materializes fixed sampled candidates and
their hashes. Full-catalog rows are derived from the artifact's frozen catalog
and known-positive snapshot with `candidate_rows_for_policy`; they are not
materialized as an O(users x items) table. Public integration functions are
`prepare_protocol_artifact`, `save_protocol_artifact`, and
`load_protocol_artifact`.
Bundle manifests hash canonical JSON content and contain no wall-clock timestamp;
loading verifies the manifest, file, config, protocol, and candidate hashes.
Unpinned `verify` detects corruption and internal inconsistency, but hashes stored
inside the same artifact are not proof against an adversary who can rewrite the
artifact and all of its hashes. Record hashes externally and pass both
`--expected-*-hash` options when authenticity matters. Source paths are retained
only as non-identity provenance; source bytes/checksums remain identity-bearing.

`graph-input` writes the interaction-only JSON accepted by the thesis
`train_temporal.py`: original-to-contiguous mappings and matching node counts for
user, artist, album, and track, with `static_edges` intentionally empty. It
uses only retained training users and train-observed target catalogs for the
default warm policy, excluding insufficient-only users and cold/future-only
items. The explicit `all_mapped` policy exports its broader transductive ID
universe. This is a supported synthetic/subset RHGNN input;
artist/album/track metadata edges require a separate enrichment export.

### Scalability scope

This pure standard-library implementation groups events and protocol state in
memory. It is intended for synthetic data and subset protocol validation, not
for direct preparation of the billion-event LFM-1b corpus. Fixed candidate
preparation reuses per-target catalog and per-user known-positive indexes and
uses rejection sampling where sparse. Derived full-catalog evaluation still has
inherent O(users x catalog) output and scoring cost.

Run the standard-library test suite with:

    python -m unittest discover -s tests -v

### Evaluation references

- PyTorch's [Reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html) explain deterministic limitations, and [`inference_mode`](https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html) is the authoritative inference API.
- Current DGL documents [`remove_edges`](https://docs.dgl.ai/generated/dgl.remove_edges.html) and [`as_edge_prediction_sampler`](https://docs.dgl.ai/generated/dgl.dataloading.as_edge_prediction_sampler.html). This project historically used DGL 0.8.2; consult the [0.8.x sampler documentation](https://www.dgl.ai/dgl_docs/en/0.8.x/generated/dgl.dataloading.as_edge_prediction_sampler.html) before translating current examples to that legacy environment.
- Rendle, [Evaluation Metrics for Item Recommendation under Sampling](https://arxiv.org/abs/1912.02263), demonstrates why sampled ranking metrics require careful interpretation.
- Meng, McCreadie, Macdonald, and Ounis, [Exploring Data Splitting Strategies for the Evaluation of Recommendation Models](https://doi.org/10.1145/3383313.3418479), analyzes how split design changes conclusions.


The LFM-1b dataset collection more than one billion listening events, intended to be used for various music retrieval and recommendation tasks. 
The [paper](http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_icmr_2016.pdf) written by Schedl, M. was published in 2016 
for ICMR and is directly available through the [website](http://www.cp.jku.at/datasets/LFM-1b/). 

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


# Requirements

The historical loader was built with Python 3.8.10 and requires these manually
installed framework versions:

- [torch](https://pytorch.org/) 1.11.0
- [dgl](https://www.dgl.ai/) 0.8.2

The referenced historical `requirements.txt` is not present in this checkout.
The new `lfm1b_protocol` package itself uses only the Python standard library.

# The Data

The node types of the graph:
- User (120K)
- Artsit (3M)
- Album (15M)
- Track (32M)
- Genre (20)

The Edge types of the graph :
- User -> Artsit (61411336)
- Artsit -> User (61411336)
- User -> Album (na)
- Album -> User (na)
- User -> Track (na)
- Track -> User (na)
- Artsit -> Genre (414379)
- Genre -> Artsit (414379)
- Album -> Artsit (14184326)
- Artsit -> Album (14184326)
- Track -> Artsit (27258365)
- Artsit -> Track (27258365)


Additionally, for all the user edges:

- User -> Artsit
- Artsit -> User
- User -> Album 
- Album -> User 
- User -> Track 
- Track -> User

There is 'norm_connections' edge data indicating the normalized realtive interaction count a src node had with a specified dst artist, album, track node. 
The 'norm_connections' edge data for all other edges is represented as a 1 

# Compile the dataset

The original `python LFM1b.py` command is stale: that filename does not exist,
and `DGL_LFM1b.py` uses package-relative imports. From this repository, the
historical equivalent is:

    PYTHONPATH=.. python -c "from DGL_LFM1b.DGL_LFM1b import LFM1b; LFM1b()"

### **Precurser warning**: 

I, the author of the repository, am using a Linux Machine with 30GB of RAM and 12GB of GPU.  To run the above script, it will take the machine ~2hrs, and I am unable to store the full knowledge graph in memory


## Compile a subset

To invoke the historical loader for a subset:

    PYTHONPATH=.. python -c "from DGL_LFM1b.DGL_LFM1b import LFM1b; LFM1b(n_users=50)"

`This provides a subset of 50 users with their correspoing listen events, and the artists/albums/tracks associated with their listening habits`


# The DGL Framework

The Deep Graph library ([DGL](https://www.dgl.ai/))  framework provides the ability to utilize the DGLDataset object
to generate a customizeable dataset for the purpose of node/link/graph down stream tasks.

Once the dataset is compiled you may import the class into any file and load the precompiled graph for DGL based analysis.

    from DGL_LFM1b.DGL_LFM1b import LFM1b

    dataset = LFM1b()
    glist, glabels = dataset.load()
    hg=glist[0]
    print(hg)
