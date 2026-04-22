"""Post-processing and analysis utilities for NeuroPose predictions.

This subpackage operates on :class:`~neuropose.io.VideoPredictions`
objects (and the numpy arrays derived from them) rather than on raw
dicts or JSON files. The intent is a set of composable pure functions —
feature extraction, normalization, joint-angle computation, and Dynamic
Time Warping — that researchers can assemble into their own pipelines.

.. note::
    The analyzer's heavy dependencies (:mod:`fastdtw`, :mod:`scipy`) are
    declared under the ``analysis`` optional-dependencies extra in
    :file:`pyproject.toml`. Install them with::

        pip install neuropose[analysis]

    The imports inside :mod:`neuropose.analyzer.dtw` and the peak-finding
    helper in :mod:`neuropose.analyzer.features` are lazy, so importing
    this subpackage does not require those dependencies. You will only
    hit a clear :class:`ImportError` at call time if they are missing.

Public API
----------
See :mod:`neuropose.analyzer.dtw` and :mod:`neuropose.analyzer.features`
for per-module details; the most commonly used names are re-exported
here for ergonomic access.
"""

from __future__ import annotations

from neuropose.analyzer.dtw import (
    AlignMode,
    DTWResult,
    NanPolicy,
    Representation,
    dtw_all,
    dtw_per_joint,
    dtw_relation,
)
from neuropose.analyzer.features import (
    AlignmentDiagnostics,
    FeatureStatistics,
    ProcrustesMode,
    extract_feature_statistics,
    extract_joint_angles,
    find_peaks,
    normalize_pose_sequence,
    pad_sequences,
    predictions_to_numpy,
    procrustes_align,
)
from neuropose.analyzer.pipeline import (
    AnalysisConfig,
    AnalysisReport,
    AnalysisResults,
    AnalysisStage,
    DtwAnalysis,
    DtwResults,
    ExtractorSegmentation,
    FeatureSummary,
    GaitCyclesBilateralSegmentation,
    GaitCyclesSegmentation,
    InputsConfig,
    InputSummary,
    NoAnalysis,
    NoResults,
    OutputConfig,
    PreprocessingConfig,
    SegmentationStage,
    StatsAnalysis,
    StatsResults,
    analysis_config_to_dict,
    load_config,
    load_report,
    run_analysis,
    save_report,
)
from neuropose.analyzer.segment import (
    JOINT_INDEX,
    JOINT_NAMES,
    AxisLetter,
    extract_signal,
    joint_angle,
    joint_axis,
    joint_index,
    joint_pair_distance,
    joint_speed,
    segment_by_peaks,
    segment_gait_cycles,
    segment_gait_cycles_bilateral,
    segment_predictions,
    slice_predictions,
)

__all__ = [
    "JOINT_INDEX",
    "JOINT_NAMES",
    "AlignMode",
    "AlignmentDiagnostics",
    "AnalysisConfig",
    "AnalysisReport",
    "AnalysisResults",
    "AnalysisStage",
    "AxisLetter",
    "DTWResult",
    "DtwAnalysis",
    "DtwResults",
    "ExtractorSegmentation",
    "FeatureStatistics",
    "FeatureSummary",
    "GaitCyclesBilateralSegmentation",
    "GaitCyclesSegmentation",
    "InputSummary",
    "InputsConfig",
    "NanPolicy",
    "NoAnalysis",
    "NoResults",
    "OutputConfig",
    "PreprocessingConfig",
    "ProcrustesMode",
    "Representation",
    "SegmentationStage",
    "StatsAnalysis",
    "StatsResults",
    "analysis_config_to_dict",
    "dtw_all",
    "dtw_per_joint",
    "dtw_relation",
    "extract_feature_statistics",
    "extract_joint_angles",
    "extract_signal",
    "find_peaks",
    "joint_angle",
    "joint_axis",
    "joint_index",
    "joint_pair_distance",
    "joint_speed",
    "load_config",
    "load_report",
    "normalize_pose_sequence",
    "pad_sequences",
    "predictions_to_numpy",
    "procrustes_align",
    "run_analysis",
    "save_report",
    "segment_by_peaks",
    "segment_gait_cycles",
    "segment_gait_cycles_bilateral",
    "segment_predictions",
    "slice_predictions",
]
