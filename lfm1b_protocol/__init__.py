"""Dependency-free LFM-1b protocol preparation and evaluation utilities."""

from .artifacts import (ArtifactIntegrityError, candidate_rows_for_policy,
                        graph_input_from_protocol, load_bundle,
                        load_protocol_artifact,
                        prepare_protocol_artifact, save_bundle,
                         save_graph_input, save_protocol_artifact,
                        validate_graph_input, validate_protocol_artifact)
from .candidates import build_candidates, candidate_digest
from .io import read_listening_events
from .leakage import LeakageFinding, LeakageReport, verify_endpoint_mappings
from .metrics import (MetricResult, RankingContribution,
                      aggregate_ranking_contributions,
                      evaluate_ranking_contributions, evaluate_rankings)
from .models import CandidateSet, Event, PairInteraction, TemporalSplit
from .split import ProtocolData, SplitStatistics, temporal_split

__all__ = [
    "ArtifactIntegrityError", "CandidateSet", "Event", "LeakageFinding",
    "LeakageReport", "MetricResult", "PairInteraction", "ProtocolData",
    "RankingContribution",
    "SplitStatistics", "TemporalSplit", "aggregate_ranking_contributions",
    "build_candidates", "candidate_digest",
    "candidate_rows_for_policy", "evaluate_ranking_contributions",
    "evaluate_rankings", "load_bundle",
    "graph_input_from_protocol", "load_protocol_artifact",
    "prepare_protocol_artifact", "read_listening_events", "save_bundle",
    "save_graph_input", "save_protocol_artifact",
    "temporal_split", "validate_graph_input", "validate_protocol_artifact", "verify_endpoint_mappings",
]
