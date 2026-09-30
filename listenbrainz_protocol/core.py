"""Local-only ListenBrainz JSONL normalization and protocol wrapper.

User identifiers are pseudonymized with a deterministic digest; this is not
anonymization.  No network, archive, or API operation is performed here.
"""
import hashlib
import json
import math
import os
import uuid
from collections import Counter, defaultdict

from lfm1b_protocol.artifacts import (ArtifactIntegrityError, candidate_rows_for_policy,
    graph_input_from_protocol, load_bundle, prepare_protocol_artifact, save_bundle,
    validate_protocol_artifact)
from lfm1b_protocol.canonical import sha256, to_primitive
from lfm1b_protocol.metrics import evaluate_rankings
from lfm1b_protocol.models import Event

SCHEMA_VERSION = 1
GRAPH_SCHEMA_VERSION = 1
PARSER_VERSION = "listenbrainz_listens_jsonl_v1"
_HEX = set("0123456789abcdef")

class ListenBrainzIntegrityError(ValueError):
    pass

def _fail(message):
    raise ListenBrainzIntegrityError(message)

def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        _fail(label + " must be a lowercase 64-character SHA256 digest")
    return value

def _integer(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        _fail(label + " must be %sa non-boolean integer" % ("a positive " if positive else ""))
    return value

def _nonnegative_integer(value, label):
    value = _integer(value, label)
    if value < 0:
        _fail(label + " must be non-negative")
    return value

def _member_path(value):
    if (not isinstance(value, str) or not value or "\\" in value or value.startswith("/") or
            any(ord(character) < 32 for character in value) or
            any(p in ("", ".", "..") for p in value.split("/"))):
        _fail("member_path must be a safe relative POSIX path")
    return value

def _namespaced_uuid(value, prefix):
    if not isinstance(value, str) or not value.startswith(prefix):
        return False
    suffix = value[len(prefix):]
    try:
        return suffix == str(uuid.UUID(suffix))
    except (ValueError, AttributeError, TypeError):
        return False

def _uuid(value, prefix, line, required=False):
    if value is None and not required: return None
    if not isinstance(value, str): _fail("line %d: selected UUID must be a string" % line)
    try: canonical = str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError): _fail("line %d: selected UUID is invalid" % line)
    return prefix + canonical

def _no_constants(value):
    raise ValueError("nonstandard JSON constant " + value)

def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError("duplicate JSON object key: " + key)
        result[key] = value
    return result

def _mapping(value, line, label):
    if not isinstance(value, dict): _fail("line %d: %s must be an object" % (line, label))
    return value

def _read(path, *, track_identity_policy, album_entity, max_bytes=None,
          max_events=None, max_users=None, max_items_per_target=None):
    """Read exact bytes once and return normalized identity-bearing rows."""
    if track_identity_policy not in ("server_mapped_musicbrainz", "recording_msid"):
        _fail("invalid track_identity_policy")
    if album_entity not in ("release_group", "release"): _fail("invalid album_entity")
    digest = hashlib.sha256(); size = 0; parsed = dropped = multi = missing_artist = missing_album = 0
    events = []; names = {}; duplicate = defaultdict(list)
    bounded_entities = {"artist": set(), "album": set(), "track": set()}
    try:
        with open(path, "rb") as handle:
            line = 0
            while True:
                raw = (handle.readline(max_bytes - size + 1)
                       if max_bytes is not None else handle.readline())
                if not raw:
                    break
                line += 1
                if max_bytes is not None and size + len(raw) > max_bytes:
                    _fail("local listens member exceeds its byte limit")
                digest.update(raw); size += len(raw)
                try: text = raw.decode("utf-8")
                except UnicodeDecodeError: _fail("line %d: invalid UTF-8" % line)
                if not text.strip(): _fail("line %d: blank JSONL line" % line)
                try: row = json.loads(text, object_pairs_hook=_pairs_no_duplicates, parse_constant=_no_constants)
                except (ValueError, TypeError, json.JSONDecodeError) as error: _fail("line %d: malformed JSON (%s)" % (line, error))
                row = _mapping(row, line, "record"); parsed += 1
                if max_events is not None and parsed > max_events:
                    _fail("local listens member exceeds its event limit")
                username = row.get("user_name")
                if not isinstance(username, str) or not username: _fail("line %d: user_name must be nonempty" % line)
                timestamp = _integer(row.get("listened_at"), "line %d: listened_at" % line)
                metadata = _mapping(row.get("track_metadata"), line, "track_metadata")
                mapping = _mapping(metadata.get("mbid_mapping"), line, "mbid_mapping") if metadata.get("mbid_mapping") is not None else {}
                if track_identity_policy == "server_mapped_musicbrainz":
                    selected = mapping.get("recording_mbid")
                    if selected is None: dropped += 1; continue
                    track = _uuid(selected, "musicbrainz:recording:", line, True)
                else:
                    selected = row.get("recording_msid")
                    if selected is None: dropped += 1; continue
                    track = _uuid(selected, "listenbrainz:recording-msid:", line, True)
                artist_values = mapping.get("artist_mbids")
                if artist_values is not None and (not isinstance(artist_values, list) or not all(isinstance(x, str) for x in artist_values)):
                    _fail("line %d: artist_mbids must be an array of strings" % line)
                artist = _uuid(artist_values[0], "musicbrainz:artist:", line) if artist_values else None
                if artist_values and len(artist_values) > 1: multi += 1
                if artist is None: missing_artist += 1
                album_key = "release_group_mbid" if album_entity == "release_group" else "release_mbid"
                album_prefix = "musicbrainz:release-group:" if album_entity == "release_group" else "musicbrainz:release:"
                album = _uuid(mapping.get(album_key), album_prefix, line) if mapping.get(album_key) is not None else None
                if album is None: missing_album += 1
                user = hashlib.sha256(b"listenbrainz:user:v1\0" + username.encode("utf-8")).hexdigest()
                if user in names and names[user] != username: _fail("pseudonymization digest collision between distinct user names")
                names[user] = username
                if max_users is not None and len(names) > max_users:
                    _fail("local listens member exceeds its user limit")
                for kind, value in (("artist", artist), ("album", album),
                                    ("track", track)):
                    if value is not None:
                        bounded_entities[kind].add(value)
                        if (max_items_per_target is not None and
                                len(bounded_entities[kind]) > max_items_per_target):
                            _fail("local listens member exceeds its per-target item limit")
                events.append((user, artist, album, track, timestamp, line))
                duplicate[(user, timestamp, track)].append((artist, album))
    except OSError as error: _fail("cannot read local listens member: %s" % error)
    duplicate_rows = sum(len(values) - 1 for values in duplicate.values())
    conflicts = sum(1 for values in duplicate.values() if len(set(values)) > 1)
    return events, {"bytes": size, "sha256": digest.hexdigest(), "parsed_rows": parsed,
        "dropped_missing_track_identity_rows": dropped, "emitted_events": len(events),
        "duplicate_event_key_rows": duplicate_rows, "conflicting_duplicate_event_keys": conflicts,
        "multi_artist_rows": multi, "missing_artist_rows": missing_artist, "missing_album_rows": missing_album}, names

def prepare(path, *, dump_id, dump_type, archive_sha256, member_path, track_identity_policy,
            album_entity, split_strategy, validation_cutoff=None, test_cutoff=None,
            repeat_policy="novel_only", positive_filter_horizon="as_of_split",
            catalog_policy="train_observed", sampled_negatives=1000, seed=0,
            include_local_path=False, max_source_bytes=None, max_events=None,
            max_users=None, max_items_per_target=None, max_candidate_rows=None,
            max_candidate_items=None):
    """Prepare a strict wrapper from one local extracted member; usernames are pseudonymized."""
    if not isinstance(dump_id, str) or not dump_id: _fail("dump_id must be a nonempty string")
    if dump_type not in ("full", "incremental"): _fail("dump_type must be full or incremental")
    _digest(archive_sha256, "archive_sha256"); _member_path(member_path)
    if not isinstance(include_local_path, bool): _fail("include_local_path must be boolean")
    for label, value in (("source bytes", max_source_bytes),
                         ("events", max_events), ("users", max_users),
                         ("items per target", max_items_per_target),
                         ("candidate rows", max_candidate_rows),
                         ("candidate items", max_candidate_items)):
        if (value is not None and
                (isinstance(value, bool) or not isinstance(value, int) or value <= 0)):
            _fail("maximum %s must be a positive integer or None" % label)
    rows, stats, names = _read(
        path, track_identity_policy=track_identity_policy, album_entity=album_entity,
        max_bytes=max_source_bytes, max_events=max_events, max_users=max_users,
        max_items_per_target=max_items_per_target)
    users = sorted(names)
    entities = {"artist": sorted({r[1] for r in rows if r[1]}), "album": sorted({r[2] for r in rows if r[2]}), "track": sorted({r[3] for r in rows})}
    uid = {x: n for n, x in enumerate(users)}
    ids = {kind: {x: n for n, x in enumerate(values)} for kind, values in entities.items()}
    events = tuple(Event(uid[u], ids["artist"].get(a), ids["album"].get(b), ids["track"][t], ts, line)
                   for u, a, b, t, ts, line in rows)
    protocol = prepare_protocol_artifact(events, sampled_negatives, seed, catalog_policy, sources=(),
        split_strategy=split_strategy, validation_cutoff=validation_cutoff, test_cutoff=test_cutoff,
        repeat_policy=repeat_policy, positive_filter_horizon=positive_filter_horizon,
        max_events=max_events, max_users=max_users,
        max_items_per_target=max_items_per_target,
        max_candidate_rows=max_candidate_rows,
        max_candidate_items=max_candidate_items)
    protocol = validate_protocol_artifact(protocol)
    source = {"source_kind": "listenbrainz_listens_jsonl_member", "license_assertion": "CC0-1.0",
        "parser_version": PARSER_VERSION, "dump_id": dump_id, "dump_type": dump_type,
        "archive_sha256": archive_sha256, "archive_digest_binding": "self_attested",
        "member_path": member_path, "member_bytes": stats.pop("bytes"), "member_sha256": stats.pop("sha256"),
        "parsed_row_count": stats["parsed_rows"]}
    if include_local_path: provenance = {"local_path": os.path.abspath(path)}
    else: provenance = {}
    stats.update({"user_count": len(users), "artist_count": len(entities["artist"]), "album_count": len(entities["album"]), "track_count": len(entities["track"])})
    config = {"track_identity_policy": track_identity_policy, "album_entity": album_entity,
        "mapping_policy": "sorted_namespaced_entity_ids_v1", "mapping_version": 1,
        "duplicate_policy": "retain_all_report_keys", "artist_interaction_policy": "first_server_mapped_credit_only",
        "embedded_protocol_config_hash": protocol["config_hash"]}
    mappings = {"users": users, "artists": entities["artist"], "albums": entities["album"], "tracks": entities["track"]}
    wrapper = {"schema_version": SCHEMA_VERSION, "config": config, "config_hash": sha256(config), "source": source,
        "statistics": stats, "mappings": mappings, "protocol": protocol, "provenance": provenance}
    wrapper["artifact_hash"] = sha256({k:v for k,v in wrapper.items() if k not in ("artifact_hash", "provenance")})
    return validate(wrapper)

def _identity(wrapper): return {k:v for k,v in wrapper.items() if k not in ("artifact_hash", "provenance")}

def validate(wrapper, expected_artifact_hash=None, expected_protocol_hash=None, expected_config_hash=None):
    try:
        required = {"schema_version", "config", "config_hash", "source", "statistics", "mappings", "protocol", "artifact_hash", "provenance"}
        if (not isinstance(wrapper, dict) or set(wrapper) != required or
                isinstance(wrapper.get("schema_version"), bool) or
                not isinstance(wrapper.get("schema_version"), int) or
                wrapper.get("schema_version") != SCHEMA_VERSION):
            _fail("unsupported ListenBrainz wrapper schema")
        for key in ("config_hash", "artifact_hash"): _digest(wrapper.get(key), key)
        if not isinstance(wrapper["config"], dict) or wrapper["config_hash"] != sha256(wrapper["config"]): _fail("wrapper config hash mismatch")
        if wrapper["artifact_hash"] != sha256(_identity(wrapper)): _fail("wrapper artifact hash mismatch")
        provenance = wrapper["provenance"]
        if (not isinstance(provenance, dict) or set(provenance) not in (set(), {"local_path"}) or
                ("local_path" in provenance and
                 (not isinstance(provenance["local_path"], str) or
                  not os.path.isabs(provenance["local_path"])))):
            _fail("invalid nonidentity provenance")
        c = wrapper["config"]
        if (set(c) != {"track_identity_policy", "album_entity", "mapping_policy", "mapping_version", "duplicate_policy", "artist_interaction_policy", "embedded_protocol_config_hash"} or
                c["track_identity_policy"] not in ("server_mapped_musicbrainz", "recording_msid") or
                c["album_entity"] not in ("release_group", "release") or
                c["mapping_policy"] != "sorted_namespaced_entity_ids_v1" or
                isinstance(c["mapping_version"], bool) or
                not isinstance(c["mapping_version"], int) or c["mapping_version"] != 1 or
                c["duplicate_policy"] != "retain_all_report_keys" or
                c["artist_interaction_policy"] != "first_server_mapped_credit_only"):
            _fail("invalid wrapper config")
        _digest(c["embedded_protocol_config_hash"], "embedded protocol config hash")
        s = wrapper["source"]; source_keys = {"source_kind", "license_assertion", "parser_version", "dump_id", "dump_type", "archive_sha256", "archive_digest_binding", "member_path", "member_bytes", "member_sha256", "parsed_row_count"}
        if (not isinstance(s, dict) or set(s) != source_keys or
                s["source_kind"] != "listenbrainz_listens_jsonl_member" or
                s["license_assertion"] != "CC0-1.0" or s["parser_version"] != PARSER_VERSION or
                not isinstance(s["dump_id"], str) or not s["dump_id"] or
                any(ord(character) < 32 for character in s["dump_id"]) or
                s["dump_type"] not in ("full", "incremental") or
                s["archive_digest_binding"] != "self_attested"):
            _fail("invalid source identity")
        _digest(s["archive_sha256"], "archive sha256")
        _digest(s["member_sha256"], "member sha256")
        _member_path(s["member_path"])
        _nonnegative_integer(s["member_bytes"], "member bytes")
        _nonnegative_integer(s["parsed_row_count"], "parsed row count")
        p = validate_protocol_artifact(wrapper["protocol"])
        if p["sources"] or p["config_hash"] != c["embedded_protocol_config_hash"]: _fail("embedded protocol source/config mismatch")
        m = wrapper["mappings"]
        if not isinstance(m, dict) or set(m) != {"users", "artists", "albums", "tracks"}: _fail("invalid mappings")
        prefixes = {"users": None, "artists": "musicbrainz:artist:", "albums": "musicbrainz:" + ("release-group:" if c["album_entity"] == "release_group" else "release:"), "tracks": "musicbrainz:recording:" if c["track_identity_policy"] == "server_mapped_musicbrainz" else "listenbrainz:recording-msid:"}
        for key, prefix in prefixes.items():
            values = m[key]
            if (not isinstance(values, list) or values != sorted(set(values)) or
                    any(not isinstance(x, str) or
                        (prefix is not None and not _namespaced_uuid(x, prefix)) for x in values)):
                _fail("mappings must be sorted, unique, and canonically namespaced")
            if prefix is None and any(len(x) != 64 or set(x) - _HEX for x in values): _fail("user mappings must contain pseudonym digests only")
        st = wrapper["statistics"]
        stat_keys = {"parsed_rows", "dropped_missing_track_identity_rows", "emitted_events", "duplicate_event_key_rows", "conflicting_duplicate_event_keys", "multi_artist_rows", "missing_artist_rows", "missing_album_rows", "user_count", "artist_count", "album_count", "track_count"}
        if not isinstance(st, dict) or set(st) != stat_keys or any(isinstance(v,bool) or not isinstance(v,int) or v < 0 for v in st.values()) or st["parsed_rows"] != s["parsed_row_count"] or st["parsed_rows"] != st["emitted_events"] + st["dropped_missing_track_identity_rows"]: _fail("invalid normalization statistics")
        if any(st[n] != len(m[k]) for n,k in (("user_count","users"),("artist_count","artists"),("album_count","albums"),("track_count","tracks"))): _fail("mapping counts mismatch")
        emitted = st["emitted_events"]
        if (st["duplicate_event_key_rows"] > max(0, emitted - 1) or
                st["conflicting_duplicate_event_keys"] > st["duplicate_event_key_rows"] or
                any(st[name] > emitted for name in
                    ("multi_artist_rows", "missing_artist_rows", "missing_album_rows",
                     "user_count", "artist_count", "album_count", "track_count"))):
            _fail("normalization statistics exceed feasible bounds")
        # Every mapped entity must appear in accepted, embedded positives; IDs must be in range.
        seen = {"users":set(), "artists":set(), "albums":set(), "tracks":set()}
        for kind, key in (("artist","artists"),("album","albums"),("track","tracks")):
            for row in p["targets"][kind]["known_positives"]:
                if (not (0 <= row["user_id"] < len(m["users"])) or
                        not (0 <= row["item_id"] < len(m[key]))):
                    _fail("embedded ID outside mapping range")
                seen["users"].add(row["user_id"]); seen[key].add(row["item_id"])
        if any(seen[k] != set(range(len(m[k]))) for k in seen): _fail("mappings include entities not represented by accepted positives")
        if p["statistics"]["total_users"] != len(m["users"]):
            _fail("embedded user count differs from wrapper mappings")
        raw_track_plays = sum(r["play_count"] for name in ("train", "validation", "test")
                              for r in p["targets"]["track"]["raw_splits"][name])
        insufficient = p["statistics"]["insufficient_users"]
        raw_track_users = {r["user_id"] for name in ("train", "validation", "test")
                           for r in p["targets"]["track"]["raw_splits"][name]}
        expected_insufficient = tuple(sorted(set(range(len(m["users"]))) - raw_track_users))
        if tuple(insufficient) != expected_insufficient:
            _fail("embedded insufficient-user identities do not match track rows")
        if ((p["config"]["split_strategy"] == "global_time_cutoffs" or not insufficient) and
                raw_track_plays != emitted):
            _fail("embedded track play total differs from emitted events")
        if insufficient and not raw_track_plays < emitted:
            _fail("per-user split does not omit the declared insufficient users")
        if expected_artifact_hash is not None and wrapper["artifact_hash"] != expected_artifact_hash: _fail("expected wrapper hash does not match")
        if expected_protocol_hash is not None and p["protocol_hash"] != expected_protocol_hash: _fail("expected protocol hash does not match")
        if expected_config_hash is not None and wrapper["config_hash"] != expected_config_hash: _fail("expected config hash does not match")
        return wrapper
    except (ArtifactIntegrityError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ListenBrainzIntegrityError): raise
        _fail("malformed ListenBrainz wrapper: %s" % error)

def save(directory, wrapper): return save_bundle(directory, {"listenbrainz": validate(wrapper)})
def load(directory, expected_artifact_hash=None, expected_protocol_hash=None, expected_config_hash=None):
    loaded = load_bundle(directory, max_artifact_count=1)
    if set(loaded) != {"listenbrainz"}: _fail("bundle must contain only listenbrainz.json")
    return validate(loaded["listenbrainz"], expected_artifact_hash, expected_protocol_hash, expected_config_hash)

def evaluate_baseline(wrapper, *, model, target="track", split="test", candidate_policy="full_catalog", k=10, random_seed=0, neighbor_limit=50):
    wrapper = validate(wrapper); p = wrapper["protocol"]
    if model not in ("random", "popularity", "itemknn") or target not in ("artist","album","track") or split not in ("validation","test") or candidate_policy not in ("fixed_sampled","full_catalog"): _fail("invalid baseline request")
    _integer(k, "k", True); _integer(random_seed, "random seed"); _integer(neighbor_limit, "neighbor limit", True)
    rows = candidate_rows_for_policy(p, target, split, candidate_policy)
    if not rows: _fail("no candidate rows for requested baseline")
    train = p["targets"][target]["splits"]["train"]; popularity = Counter({})
    histories = defaultdict(set)
    for row in train: popularity[row["item_id"]] += row["play_count"]; histories[row["user_id"]].add(row["item_id"])
    similarities = defaultdict(dict)
    if model == "itemknn":
        counts = Counter(); co = Counter()
        for items in histories.values():
            for i in items: counts[i] += 1
            for i in items:
                for j in items:
                    if i != j: co[(i,j)] += 1
        for (i,j), value in co.items(): similarities[i][j] = value / math.sqrt(counts[i] * counts[j])
    scores = {}
    for row in rows:
        user = row["user_id"]; candidates = row["candidate_item_ids"]
        if model == "random": score = lambda item: int(hashlib.sha256(("listenbrainz:random:v1:%s:%s:%s:%s:%s" % (random_seed, target, split, user, item)).encode("ascii")).hexdigest(), 16)
        elif model == "popularity": score = lambda item: popularity[item]
        else:
            neighbors = {}
            for prior in histories[user]:
                for item, sim in similarities[prior].items(): neighbors.setdefault(item, []).append((sim, prior))
            def score(item): return sum(sim for sim, prior in sorted(neighbors.get(item, ()), key=lambda x:(-x[0],x[1]))[:neighbor_limit])
        scores[user] = [(item, score(item)) for item in candidates]
    result = evaluate_rankings(scores, {r["user_id"]:r["positive_item_ids"] for r in rows}, {r["user_id"]:r["candidate_item_ids"] for r in rows}, k, p["targets"][target]["catalog"], popularity)
    model_config = ({"algorithm": "sha256_priority_v1", "random_seed": random_seed}
                    if model == "random" else
                    {"algorithm": "train_play_count_popularity_v1"}
                    if model == "popularity" else
                    {"algorithm": "binary_cosine_itemknn_v1", "neighbor_limit": neighbor_limit})
    output = dict(result.__dict__); output.update({"model":model,"model_config":model_config,"item_type":target,"split":split,"candidate_policy":candidate_policy,"wrapper_hash":wrapper["artifact_hash"],"protocol_hash":p["protocol_hash"],"config_hash":wrapper["config_hash"],"protocol_config_hash":p["config_hash"]})
    return output

def _expected_graph(wrapper):
    core = to_primitive(graph_input_from_protocol(wrapper["protocol"]))
    mapping_hash = sha256(wrapper["mappings"])
    graph = {"schema_version":GRAPH_SCHEMA_VERSION,"wrapper_artifact_hash":wrapper["artifact_hash"],"protocol_hash":wrapper["protocol"]["protocol_hash"],"mapping_hash":mapping_hash,"core_graph":core}
    graph["graph_hash"] = sha256(graph); return graph
def graph_input(wrapper):
    wrapper = validate(wrapper)
    return _expected_graph(wrapper)
def validate_graph(wrapper, graph, expected_graph_hash=None):
    wrapper=validate(wrapper)
    if (not isinstance(graph,dict) or set(graph) != {"schema_version","wrapper_artifact_hash","protocol_hash","mapping_hash","core_graph","graph_hash"} or
            isinstance(graph.get("schema_version"), bool) or
            not isinstance(graph.get("schema_version"), int) or
            graph.get("schema_version") != GRAPH_SCHEMA_VERSION):
        _fail("invalid ListenBrainz graph schema")
    for key in ("wrapper_artifact_hash", "protocol_hash", "mapping_hash", "graph_hash"):
        _digest(graph.get(key), "graph " + key)
    if graph.get("graph_hash") != sha256({k:v for k,v in graph.items() if k != "graph_hash"}): _fail("graph hash mismatch")
    expected=_expected_graph(wrapper)
    if sha256({k:v for k,v in graph.items() if k != "graph_hash"}) != sha256({k:v for k,v in expected.items() if k != "graph_hash"}): _fail("graph is not bound to this ListenBrainz wrapper")
    if expected_graph_hash is not None and graph["graph_hash"] != expected_graph_hash:
        _fail("expected graph hash does not match")
    return graph
def save_graph(directory, wrapper): return save_bundle(directory, {"listenbrainz_graph": graph_input(wrapper)})
def load_graph(directory, wrapper, expected_graph_hash=None):
    loaded=load_bundle(directory,max_artifact_count=1)
    if set(loaded)!={"listenbrainz_graph"}: _fail("bundle must contain only listenbrainz_graph.json")
    return validate_graph(wrapper, loaded["listenbrainz_graph"], expected_graph_hash)
