"""Deterministic bounded comparisons over local protocol artifacts."""

from .adapters import (LoadedArtifact, artifact_summary, case_from_artifact,
                       inspect_artifact, load_case_artifact, mask_configuration)
from .contrasts import (contrast_pair, create_contrast_analysis,
                        load_contrast_analysis, result_summaries,
                        save_contrast_analysis, validate_contrast_analysis)
from .matrix import (create_contrast_analysis_from_matrix_plan,
                     create_mask_matrix_spec, create_matrix_comparison_plan,
                     load_mask_matrix_spec, load_matrix_contrast_plan,
                     load_matrix_index, matrix_preview, matrix_variants,
                     prepare_mask_matrix, resolve_matrix_contrast_plan,
                     save_mask_matrix_spec, save_matrix_contrast_plan,
                     save_matrix_index, validate_mask_matrix_spec,
                     validate_matrix_contrast_plan, validate_matrix_index,
                     validate_matrix_output_paths)
from .plan import (ComparisonError, create_plan, load_plan, save_plan,
                    validate_case, validate_plan)
from .runner import (load_report, run_plan, save_report, validate_report)
from .scoring import evaluate_protocol_baseline

__all__ = [
    "ComparisonError", "LoadedArtifact", "artifact_summary",
    "case_from_artifact", "contrast_pair", "create_contrast_analysis",
    "create_contrast_analysis_from_matrix_plan", "create_mask_matrix_spec",
    "create_matrix_comparison_plan", "create_plan",
    "evaluate_protocol_baseline", "inspect_artifact", "load_case_artifact",
    "load_contrast_analysis", "load_mask_matrix_spec",
    "load_matrix_contrast_plan", "load_matrix_index", "load_plan",
    "load_report", "mask_configuration", "matrix_preview", "matrix_variants",
    "prepare_mask_matrix", "resolve_matrix_contrast_plan", "result_summaries",
    "run_plan", "save_contrast_analysis", "save_mask_matrix_spec",
    "save_matrix_contrast_plan", "save_matrix_index", "save_plan",
    "save_report", "validate_case", "validate_contrast_analysis",
    "validate_mask_matrix_spec", "validate_matrix_contrast_plan",
    "validate_matrix_index", "validate_matrix_output_paths", "validate_plan",
    "validate_report",
]
