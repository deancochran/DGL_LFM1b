import random
import unittest
from collections import defaultdict

from lfm1b_protocol.artifacts import (prepare_protocol_artifact,
                                      validate_protocol_artifact)
from lfm1b_protocol.models import Event, ITEM_TYPES


WINDOWS = ("train", "validation", "test")


def _pair_set(rows):
    return {(row["user_id"], row["item_id"]) for row in rows}


def _expected_raw(events, strategy, validation_cutoff=None, test_cutoff=None):
    by_user = defaultdict(list)
    for event in events:
        by_user[event.user_id].append(event)
    result = {
        kind: {window: [] for window in WINDOWS}
        for kind in ITEM_TYPES
    }
    for user_id in sorted(by_user):
        user_events = by_user[user_id]
        timestamps = sorted({event.timestamp for event in user_events})
        if strategy == "per_user_last_timestamp_groups":
            if len(timestamps) < 3:
                continue
            validation_time, test_time = timestamps[-2:]
            assignments = {
                "train": [event for event in user_events
                          if event.timestamp < validation_time],
                "validation": [event for event in user_events
                               if event.timestamp == validation_time],
                "test": [event for event in user_events
                         if event.timestamp == test_time],
            }
        else:
            assignments = {
                "train": [event for event in user_events
                          if event.timestamp < validation_cutoff],
                "validation": [event for event in user_events
                               if validation_cutoff <= event.timestamp < test_cutoff],
                "test": [event for event in user_events
                         if event.timestamp >= test_cutoff],
            }
        for kind in ITEM_TYPES:
            for window in WINDOWS:
                grouped = defaultdict(list)
                for event in assignments[window]:
                    item_id = getattr(event, kind + "_id")
                    if item_id is not None:
                        grouped[item_id].append(event.timestamp)
                for item_id in sorted(grouped):
                    values = grouped[item_id]
                    result[kind][window].append({
                        "user_id": user_id,
                        "item_id": item_id,
                        "play_count": len(values),
                        "first_timestamp": min(values),
                        "last_timestamp": max(values),
                    })
    return result


def _candidate_exclusions(snapshots, split, repeat_policy, horizon):
    if repeat_policy == "novel_only":
        names = (WINDOWS if horizon == "all_observed" or split == "test"
                 else ("train", "validation"))
        return set().union(*(snapshots[name] for name in names))
    if horizon == "all_observed" and split == "validation":
        return (snapshots["test"] - snapshots["train"] -
                snapshots["validation"])
    return set()


class ProtocolPropertyTests(unittest.TestCase):
    def test_randomized_1920_configuration_matrix(self):
        random_source = random.Random(20260904)
        checked = 0
        for unused_trial in range(30):
            events = []
            for user_id in range(1, random_source.randint(3, 9)):
                for unused_event in range(random_source.randint(1, 13)):
                    timestamp = random_source.randint(0, 12)
                    item_ids = []
                    for maximum in (8, 10, 14):
                        item_ids.append(
                            None if random_source.random() < 0.2
                            else random_source.randint(1, maximum))
                    events.append(Event(user_id, item_ids[0], item_ids[1],
                                        item_ids[2], timestamp))
            random_source.shuffle(events)

            for strategy in ("per_user_last_timestamp_groups",
                             "global_time_cutoffs"):
                cutoffs = ({"validation_cutoff": 4, "test_cutoff": 8}
                           if strategy == "global_time_cutoffs" else {})
                expected_raw = _expected_raw(events, strategy, **cutoffs)
                for repeat_policy in ("novel_only", "repeat_allowed"):
                    for catalog_policy in ("train_observed", "all_mapped"):
                        for horizon in ("as_of_split", "all_observed"):
                            for negatives in (0, 4):
                                for seed in (0, 17):
                                    options = dict(
                                        sampled_negatives=negatives,
                                        seed=seed,
                                        catalog_policy=catalog_policy,
                                        split_strategy=strategy,
                                        repeat_policy=repeat_policy,
                                        positive_filter_horizon=horizon,
                                        **cutoffs
                                    )
                                    artifact = prepare_protocol_artifact(
                                        events, **options)
                                    validate_protocol_artifact(artifact)
                                    self.assertEqual(
                                        artifact,
                                        prepare_protocol_artifact(
                                            reversed(events), **options))
                                    expected_causal = (
                                        strategy == "global_time_cutoffs" and
                                        catalog_policy == "train_observed" and
                                        horizon == "as_of_split")
                                    self.assertIs(
                                        artifact["config"]["globally_time_causal"],
                                        expected_causal)
                                    if strategy == "global_time_cutoffs":
                                        statistics = artifact["statistics"]
                                        self.assertEqual(
                                            statistics["eligible_users"],
                                            statistics["total_users"])
                                        self.assertEqual(
                                            tuple(statistics["insufficient_users"]),
                                            ())

                                    for kind in ITEM_TYPES:
                                        target = artifact["targets"][kind]
                                        for window in WINDOWS:
                                            self.assertEqual(
                                                list(target["raw_splits"][window]),
                                                expected_raw[kind][window])
                                        snapshots = {
                                            window: _pair_set(
                                                target["positive_snapshots"][window])
                                            for window in WINDOWS
                                        }
                                        task_pairs = {
                                            "train": set(snapshots["train"])
                                        }
                                        if repeat_policy == "novel_only":
                                            task_pairs["validation"] = (
                                                snapshots["validation"] -
                                                snapshots["train"])
                                            task_pairs["test"] = (
                                                snapshots["test"] -
                                                snapshots["train"] -
                                                snapshots["validation"])
                                        else:
                                            task_pairs["validation"] = set(
                                                snapshots["validation"])
                                            task_pairs["test"] = set(
                                                snapshots["test"])
                                        catalog = set(target["catalog"])
                                        training_users = {
                                            user_id for user_id, unused_item
                                            in snapshots["train"]
                                        }
                                        for window in WINDOWS:
                                            expected_pairs = task_pairs[window]
                                            if (window != "train" and
                                                    catalog_policy ==
                                                    "train_observed"):
                                                expected_pairs = {
                                                    pair for pair in expected_pairs
                                                    if pair[0] in training_users and
                                                    pair[1] in catalog
                                                }
                                            else:
                                                expected_pairs = {
                                                    pair for pair in expected_pairs
                                                    if pair[1] in catalog
                                                }
                                            self.assertEqual(
                                                _pair_set(
                                                    target["splits"][window]),
                                                expected_pairs)

                                        for split in ("validation", "test"):
                                            positives = defaultdict(set)
                                            for row in target["splits"][split]:
                                                positives[row["user_id"]].add(
                                                    row["item_id"])
                                            blocked = _candidate_exclusions(
                                                snapshots, split, repeat_policy,
                                                horizon)
                                            blocked_by_user = defaultdict(set)
                                            for user_id, item_id in blocked:
                                                blocked_by_user[user_id].add(item_id)
                                            candidate_rows = {
                                                row["user_id"]: row for row in
                                                target["candidates"][split]
                                            }
                                            self.assertEqual(
                                                set(candidate_rows), set(positives))
                                            for user_id, user_positives in positives.items():
                                                row = candidate_rows[user_id]
                                                items = set(
                                                    row["candidate_item_ids"])
                                                eligible_negatives = (
                                                    catalog -
                                                    blocked_by_user[user_id] -
                                                    user_positives)
                                                self.assertTrue(
                                                    user_positives.issubset(items))
                                                self.assertTrue(
                                                    (items - user_positives).
                                                    issubset(eligible_negatives))
                                                self.assertEqual(
                                                    len(items - user_positives),
                                                    min(negatives,
                                                        len(eligible_negatives)))
                                    checked += 1
        self.assertEqual(checked, 1920)


if __name__ == "__main__":
    unittest.main()
