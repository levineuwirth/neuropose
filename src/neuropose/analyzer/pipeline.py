"""YAML-configurable analysis pipeline.

This module unifies the analyzer's individual primitives (Procrustes
alignment, gait-cycle segmentation, DTW on coords or angles, feature
statistics) behind a single declarative configuration object so an
experiment can be reproduced from one file that lives in a git
repository, carries through to the :class:`~neuropose.io.Provenance`
envelope on the output artifact, and can be cited unambiguously in
accompanying papers.

Two top-level schemas live here:

- :class:`AnalysisConfig` — what the user writes in YAML. Describes
  the full pipeline: inputs, preprocessing, optional segmentation,
  required analysis stage, and output path.
- :class:`AnalysisReport` — what :func:`run_analysis` emits. Carries
  the config, a :class:`~neuropose.io.Provenance` envelope (with the
  config serialised into :attr:`~neuropose.io.Provenance.analysis_config`),
  per-input summaries, segmentation results, and the analysis results
  themselves.

Both schemas parse from (and serialise to) JSON via pydantic; the
config additionally parses from YAML via :func:`load_config`. Cross-field
invariants (for example, ``method="dtw_relation"`` requires ``joint_i``
and ``joint_j``) are enforced at parse time so typo-laden configs fail
fast rather than after an expensive multi-minute load.

Execution
---------
:func:`run_analysis` is the top-level executor: it loads the
predictions files named in the config, applies any configured
segmentation stage, dispatches to the configured analysis stage, and
returns a fully populated :class:`AnalysisReport`. The executor is
intended to be called from the ``neuropose analyze`` CLI but is
equally valid as a Python-level entry point for notebook-driven
exploration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    extract_feature_statistics,
    predictions_to_numpy,
)
from neuropose.analyzer.segment import (
    AxisLetter,
    extract_signal,
    segment_gait_cycles,
    segment_gait_cycles_bilateral,
    segment_predictions,
)
from neuropose.io import (
    ExtractorSpec,
    Provenance,
    Segmentation,
    VideoPredictions,
    load_video_predictions,
)
from neuropose.migrations import CURRENT_VERSION, migrate_analysis_report

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class InputsConfig(BaseModel):
    """Predictions files consumed by the pipeline.

    Attributes
    ----------
    primary
        Path to a :class:`~neuropose.io.VideoPredictions` JSON file.
        Always required.
    reference
        Optional second predictions file. When provided,
        :class:`DtwAnalysis` runs comparative DTW between primary and
        reference; when absent, analysis stages that require a
        reference (i.e. DTW) raise a validation error at parse time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary: Path
    reference: Path | None = None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class PreprocessingConfig(BaseModel):
    """Per-input preprocessing applied before segmentation and analysis.

    Minimal today — just picks which detected person to extract from
    each frame. Left as a named stage so future extensions (coordinate
    normalisation, smoothing) can land here without reshuffling the
    config shape.

    Attributes
    ----------
    person_index
        Which detected person to extract per frame. Defaults to ``0``
        (the first detected person), matching the single-subject
        clinical case.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    person_index: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Segmentation stage
# ---------------------------------------------------------------------------


class GaitCyclesSegmentation(BaseModel):
    """Single-heel gait-cycle segmentation via peak detection.

    Produces one :class:`~neuropose.io.Segmentation` keyed under the
    joint name (e.g. ``"rhee_cycles"``). See
    :func:`~neuropose.analyzer.segment.segment_gait_cycles` for the
    underlying implementation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["gait_cycles"]
    joint: str = "rhee"
    axis: AxisLetter = "y"
    invert: bool = False
    min_cycle_seconds: float = Field(default=0.4, gt=0.0)
    min_prominence: float | None = None


class GaitCyclesBilateralSegmentation(BaseModel):
    """Bilateral (both heels) gait-cycle segmentation.

    Produces two :class:`~neuropose.io.Segmentation` objects keyed as
    ``"left_heel_strikes"`` and ``"right_heel_strikes"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["gait_cycles_bilateral"]
    axis: AxisLetter = "y"
    invert: bool = False
    min_cycle_seconds: float = Field(default=0.4, gt=0.0)
    min_prominence: float | None = None


class ExtractorSegmentation(BaseModel):
    """Generic extractor-driven segmentation.

    Wraps :func:`~neuropose.analyzer.segment.segment_predictions` with
    a caller-supplied :class:`~neuropose.io.ExtractorSpec`. Use this
    when the signal of interest is not the vertical heel trace — e.g.
    wrist-hip distance for a reach-and-grasp task, or elbow flexion
    angle for a range-of-motion trial.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["extractor"]
    extractor: ExtractorSpec
    label: str = Field(
        default="segmentation",
        description="Key under which the resulting Segmentation is stored.",
    )
    person_index: int | None = Field(
        default=None,
        description=(
            "Overrides preprocessing.person_index for this stage. "
            "None defers to the global preprocessing setting."
        ),
    )
    min_distance_seconds: float | None = Field(default=None, ge=0.0)
    min_prominence: float | None = None
    min_height: float | None = None
    pad_seconds: float = Field(default=0.0, ge=0.0)


SegmentationStage = Annotated[
    GaitCyclesSegmentation | GaitCyclesBilateralSegmentation | ExtractorSegmentation,
    Field(discriminator="kind"),
]
"""Discriminated-union alias for the three segmentation variants.

Pydantic dispatches on the ``kind`` field. A config without a
``segmentation`` key at all skips this stage entirely
(see :class:`AnalysisConfig.segmentation`).
"""


# ---------------------------------------------------------------------------
# Analysis stage
# ---------------------------------------------------------------------------


class DtwAnalysis(BaseModel):
    """Dynamic Time Warping between the primary and reference inputs.

    Dispatches to one of :func:`~neuropose.analyzer.dtw.dtw_all`,
    :func:`~neuropose.analyzer.dtw.dtw_per_joint`, or
    :func:`~neuropose.analyzer.dtw.dtw_relation` per the ``method``
    field. Cross-field invariants — ``method="dtw_relation"`` requires
    ``joint_i`` and ``joint_j``, ``representation="angles"`` requires
    a non-empty ``angle_triplets`` — are enforced at parse time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["dtw"]
    method: Literal["dtw_all", "dtw_per_joint", "dtw_relation"] = "dtw_all"
    align: AlignMode = "none"
    representation: Representation = "coords"
    angle_triplets: list[tuple[int, int, int]] | None = None
    joint_i: int | None = None
    joint_j: int | None = None
    nan_policy: NanPolicy = "propagate"

    @model_validator(mode="after")
    def _check_method_fields(self) -> DtwAnalysis:
        if self.method == "dtw_relation":
            if self.joint_i is None or self.joint_j is None:
                raise ValueError("method='dtw_relation' requires joint_i and joint_j")
            if self.representation != "coords":
                raise ValueError(
                    "method='dtw_relation' only supports representation='coords' "
                    "(a two-joint displacement is not a joint-angle signal)"
                )
        if self.representation == "angles":
            if not self.angle_triplets:
                raise ValueError("representation='angles' requires a non-empty angle_triplets list")
            if self.method == "dtw_relation":
                # Guarded by the earlier branch, but make the invariant explicit.
                raise ValueError(
                    "representation='angles' is incompatible with method='dtw_relation'"
                )
        return self


class StatsAnalysis(BaseModel):
    """Summary statistics over a scalar signal extracted from the primary input.

    Runs :func:`~neuropose.analyzer.segment.extract_signal` with the
    caller-supplied :class:`~neuropose.io.ExtractorSpec`, then
    computes :func:`~neuropose.analyzer.features.extract_feature_statistics`
    on each segment (or on the full trial if no segmentation stage
    runs).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["stats"]
    extractor: ExtractorSpec


class NoAnalysis(BaseModel):
    """Terminal stage placeholder; produces no per-segment results.

    Useful when the pipeline's goal is just to segment the input and
    persist the :class:`~neuropose.io.Segmentation` plus an
    :class:`AnalysisReport` with provenance — the ``none`` analysis
    kind makes that explicit rather than requiring the absence of the
    stage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["none"]


AnalysisStage = Annotated[
    DtwAnalysis | StatsAnalysis | NoAnalysis,
    Field(discriminator="kind"),
]
"""Discriminated-union alias for the three analysis variants.

Pydantic dispatches on ``kind``. One of the three must always be
present in a valid config.
"""


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    """Where :func:`run_analysis` should write its :class:`AnalysisReport`.

    Kept as a sub-object rather than a bare path so downstream
    extensions (figure paths, supplementary distance-matrix files)
    can land here without changing the config's top-level shape.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    report: Path


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class AnalysisConfig(BaseModel):
    """Declarative description of a full analysis run.

    Parsed from YAML (via :func:`load_config`) or JSON (via
    :meth:`pydantic.BaseModel.model_validate`). Every field is
    cross-validated at parse time so a typo in a nested sub-field
    fails in milliseconds rather than after a multi-minute
    predictions load.

    Attributes
    ----------
    config_version
        Schema version for the config itself. Only ``1`` is valid at
        this release. Future config-format breaks bump this and a
        sibling migration registry handles legacy YAML in place.
    inputs
        Predictions-file paths.
    preprocessing
        Per-input preprocessing (person-index selection today).
    segmentation
        Optional segmentation stage. ``None`` skips segmentation
        entirely and analysis runs over each full trial as a single
        "segment".
    analysis
        Required analysis stage. Exactly one of
        :class:`DtwAnalysis` / :class:`StatsAnalysis` / :class:`NoAnalysis`.
    output
        Output paths.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: Literal[1] = 1
    inputs: InputsConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    segmentation: SegmentationStage | None = None
    analysis: AnalysisStage
    output: OutputConfig

    @model_validator(mode="after")
    def _check_cross_stage_invariants(self) -> AnalysisConfig:
        # DTW is comparative — it needs a reference input.
        if isinstance(self.analysis, DtwAnalysis) and self.inputs.reference is None:
            raise ValueError("analysis.kind='dtw' requires inputs.reference to be set")
        # Stats is non-comparative — a reference without a use is
        # almost certainly an operator error.
        if isinstance(self.analysis, StatsAnalysis) and self.inputs.reference is not None:
            raise ValueError(
                "analysis.kind='stats' operates on inputs.primary only; "
                "remove inputs.reference or switch analysis.kind to 'dtw'"
            )
        return self


# ---------------------------------------------------------------------------
# Report pieces
# ---------------------------------------------------------------------------


class InputSummary(BaseModel):
    """Capsule of an input predictions file's headline metadata.

    Stored in the :class:`AnalysisReport` so a reader of the report
    can tell at a glance what was analysed without having to load the
    underlying predictions JSONs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    frame_count: int = Field(ge=0)
    fps: float = Field(ge=0.0)
    provenance: Provenance | None = None


class FeatureSummary(BaseModel):
    """Pydantic twin of :class:`~neuropose.analyzer.features.FeatureStatistics`.

    The dataclass is used throughout the analyzer for ad-hoc Python
    consumption; the report path needs a pydantic model for
    round-tripping through JSON.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    std: float
    min: float
    max: float
    range: float


class DtwResults(BaseModel):
    """DTW results attached to an :class:`AnalysisReport`.

    ``distances`` is parallel to ``segment_labels``. For an
    unsegmented run the lists have length 1 and the label is
    ``"full_trial"``. For a segmented run each label takes the form
    ``"<segmentation_key>[<index>]"`` (e.g. ``"left_heel_strikes[3]"``).
    ``per_joint_distances`` carries a per-unit breakdown for
    ``method="dtw_per_joint"`` only; its outer length matches
    ``distances``, inner length matches either ``num_joints`` (coords)
    or ``len(angle_triplets)`` (angles).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["dtw"]
    method: Literal["dtw_all", "dtw_per_joint", "dtw_relation"]
    distances: list[float]
    paths: list[list[tuple[int, int]]]
    per_joint_distances: list[list[float]] | None = None
    segment_labels: list[str]
    summary: dict[str, float]


class StatsResults(BaseModel):
    """Feature-statistics results attached to an :class:`AnalysisReport`.

    ``statistics`` is parallel to ``segment_labels``; see
    :class:`DtwResults` for the labelling convention.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["stats"]
    statistics: list[FeatureSummary]
    segment_labels: list[str]


class NoResults(BaseModel):
    """Empty results payload for ``analysis.kind='none'`` runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["none"]


AnalysisResults = Annotated[
    DtwResults | StatsResults | NoResults,
    Field(discriminator="kind"),
]
"""Discriminated-union alias for the three analysis-result shapes.

Mirrors :data:`AnalysisStage` one-for-one: ``DtwAnalysis`` produces
:class:`DtwResults`, etc.
"""


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


class AnalysisReport(BaseModel):
    """Self-describing output artifact of :func:`run_analysis`.

    Serialised to JSON on disk. Carries the originating config, the
    :class:`~neuropose.io.Provenance` envelope (with the config
    serialised into :attr:`~neuropose.io.Provenance.analysis_config`
    so the report is self-describing even if the YAML is lost), each
    input's headline metadata plus its own provenance if available,
    any segmentations produced, and the analysis results themselves.

    Lives in the schema-migration registry under ``"AnalysisReport"``
    at ``CURRENT_VERSION``; see :mod:`neuropose.migrations`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=CURRENT_VERSION, ge=1)
    config: AnalysisConfig
    provenance: Provenance | None = None
    primary: InputSummary
    reference: InputSummary | None = None
    segmentations: dict[str, Segmentation] = Field(default_factory=dict)
    results: AnalysisResults


def analysis_config_to_dict(config: AnalysisConfig) -> dict[str, Any]:
    """Serialise an :class:`AnalysisConfig` to a JSON-safe dict.

    Returned shape is identical to what pydantic would produce via
    :meth:`~pydantic.BaseModel.model_dump` in ``mode="json"`` — paths
    become strings, tuples become lists, enums become their values.
    Useful for stamping
    :attr:`~neuropose.io.Provenance.analysis_config` on the
    :class:`AnalysisReport`'s provenance envelope.
    """
    return config.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_config(path: Path) -> AnalysisConfig:
    """Load and validate an :class:`AnalysisConfig` from a YAML file.

    Parameters
    ----------
    path
        Filesystem path to a YAML file conforming to the
        :class:`AnalysisConfig` schema.

    Returns
    -------
    AnalysisConfig
        The fully validated config. Cross-field invariants have
        already been checked.

    Raises
    ------
    pydantic.ValidationError
        On any schema violation — unknown keys, wrong types, or
        failed cross-field invariants.
    yaml.YAMLError
        On malformed YAML.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    return AnalysisConfig.model_validate(raw)


def save_report(path: Path, report: AnalysisReport) -> None:
    """Serialise an :class:`AnalysisReport` to ``path`` atomically.

    Writes to a sibling ``<path>.tmp`` first, then renames over
    ``path`` so a crash mid-write cannot leave behind a truncated
    file. The parent directory is created if it does not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = report.model_dump(mode="json")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def load_report(path: Path) -> AnalysisReport:
    """Load and validate an :class:`AnalysisReport` JSON file.

    Runs the payload through :func:`~neuropose.migrations.migrate_analysis_report`
    before pydantic validation so future schema bumps can upgrade
    legacy reports transparently.
    """
    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)
    if isinstance(data, dict):
        data = migrate_analysis_report(data)
    return AnalysisReport.model_validate(data)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def run_analysis(config: AnalysisConfig) -> AnalysisReport:
    """Execute the pipeline described by ``config`` end-to-end.

    Loads the predictions files named in ``config.inputs``, applies
    the configured preprocessing + segmentation + analysis stages,
    and returns an :class:`AnalysisReport` whose
    :attr:`~AnalysisReport.provenance` inherits the inference-time
    provenance of the primary input with
    :attr:`~neuropose.io.Provenance.analysis_config` populated so the
    report is self-describing even if the YAML config is later lost.

    Parameters
    ----------
    config
        The pre-validated pipeline configuration.

    Returns
    -------
    AnalysisReport
        Fully populated report. Not yet written to disk — the caller
        passes it to :func:`save_report` (or inspects it directly).

    Notes
    -----
    For DTW runs with a segmentation stage, segments are paired
    one-to-one by index across primary and reference, truncating to
    ``min(len_primary, len_reference)``. Bilateral segmentations
    produce distances for each side independently, labelled under
    their segmentation key (e.g. ``"left_heel_strikes[3]"``).
    """
    primary_preds = load_video_predictions(config.inputs.primary)
    reference_preds: VideoPredictions | None = None
    if config.inputs.reference is not None:
        reference_preds = load_video_predictions(config.inputs.reference)

    person_index = config.preprocessing.person_index

    primary_seq = predictions_to_numpy(primary_preds, person_index=person_index)
    reference_seq: np.ndarray | None = None
    if reference_preds is not None:
        reference_seq = predictions_to_numpy(reference_preds, person_index=person_index)

    primary_segmentations: dict[str, Segmentation] = {}
    reference_segmentations: dict[str, Segmentation] = {}
    if config.segmentation is not None:
        primary_segmentations = _run_segmentation(primary_preds, config.segmentation, person_index)
        if reference_preds is not None:
            reference_segmentations = _run_segmentation(
                reference_preds, config.segmentation, person_index
            )

    results = _run_analysis_stage(
        config.analysis,
        primary_seq=primary_seq,
        reference_seq=reference_seq,
        primary_segmentations=primary_segmentations,
        reference_segmentations=reference_segmentations,
    )

    analysis_config_dump = analysis_config_to_dict(config)
    report_provenance: Provenance | None = None
    if primary_preds.provenance is not None:
        report_provenance = primary_preds.provenance.model_copy(
            update={"analysis_config": analysis_config_dump}
        )

    primary_summary = InputSummary(
        path=config.inputs.primary,
        frame_count=primary_preds.metadata.frame_count,
        fps=primary_preds.metadata.fps,
        provenance=primary_preds.provenance,
    )
    reference_summary: InputSummary | None = None
    if reference_preds is not None and config.inputs.reference is not None:
        reference_summary = InputSummary(
            path=config.inputs.reference,
            frame_count=reference_preds.metadata.frame_count,
            fps=reference_preds.metadata.fps,
            provenance=reference_preds.provenance,
        )

    return AnalysisReport(
        config=config,
        provenance=report_provenance,
        primary=primary_summary,
        reference=reference_summary,
        segmentations=primary_segmentations,
        results=results,
    )


# ---------------------------------------------------------------------------
# Internal dispatch helpers
# ---------------------------------------------------------------------------


def _run_segmentation(
    predictions: VideoPredictions,
    stage: SegmentationStage,  # type: ignore[valid-type]
    person_index: int,
) -> dict[str, Segmentation]:
    """Apply a segmentation stage to a :class:`VideoPredictions`.

    Returns a dict keyed by a stage-appropriate label: single-side
    gait cycles use ``"<joint>_cycles"``, bilateral gait cycles use
    ``"left_heel_strikes"`` / ``"right_heel_strikes"``, and extractor
    segmentation uses the caller-supplied
    :attr:`~ExtractorSegmentation.label`.
    """
    if isinstance(stage, GaitCyclesSegmentation):
        seg = segment_gait_cycles(
            predictions,
            joint=stage.joint,
            axis=stage.axis,
            invert=stage.invert,
            min_cycle_seconds=stage.min_cycle_seconds,
            min_prominence=stage.min_prominence,
        )
        return {f"{stage.joint}_cycles": seg}
    if isinstance(stage, GaitCyclesBilateralSegmentation):
        return segment_gait_cycles_bilateral(
            predictions,
            axis=stage.axis,
            invert=stage.invert,
            min_cycle_seconds=stage.min_cycle_seconds,
            min_prominence=stage.min_prominence,
        )
    if isinstance(stage, ExtractorSegmentation):
        effective_person_index = (
            stage.person_index if stage.person_index is not None else person_index
        )
        seg = segment_predictions(
            predictions,
            stage.extractor,
            person_index=effective_person_index,
            min_distance_seconds=stage.min_distance_seconds,
            min_prominence=stage.min_prominence,
            min_height=stage.min_height,
            pad_seconds=stage.pad_seconds,
        )
        return {stage.label: seg}
    raise TypeError(f"unknown segmentation stage: {type(stage).__name__}")


def _run_analysis_stage(
    stage: AnalysisStage,  # type: ignore[valid-type]
    *,
    primary_seq: np.ndarray,
    reference_seq: np.ndarray | None,
    primary_segmentations: dict[str, Segmentation],
    reference_segmentations: dict[str, Segmentation],
) -> AnalysisResults:  # type: ignore[valid-type]
    """Dispatch to the appropriate analysis executor per ``stage.kind``."""
    if isinstance(stage, DtwAnalysis):
        if reference_seq is None:
            # AnalysisConfig's cross-stage validator should prevent
            # this; duplicate the check here so a direct programmatic
            # call can't slip through.
            raise ValueError("DtwAnalysis requires a reference sequence")
        return _run_dtw(
            stage,
            primary_seq=primary_seq,
            reference_seq=reference_seq,
            primary_segmentations=primary_segmentations,
            reference_segmentations=reference_segmentations,
        )
    if isinstance(stage, StatsAnalysis):
        return _run_stats(
            stage,
            primary_seq=primary_seq,
            primary_segmentations=primary_segmentations,
        )
    if isinstance(stage, NoAnalysis):
        return NoResults(kind="none")
    raise TypeError(f"unknown analysis stage: {type(stage).__name__}")


def _run_dtw(
    stage: DtwAnalysis,
    *,
    primary_seq: np.ndarray,
    reference_seq: np.ndarray,
    primary_segmentations: dict[str, Segmentation],
    reference_segmentations: dict[str, Segmentation],
) -> DtwResults:
    """Execute a DTW analysis stage, returning :class:`DtwResults`."""
    labels: list[str] = []
    distances: list[float] = []
    paths: list[list[tuple[int, int]]] = []
    per_joint_distances: list[list[float]] | None = [] if stage.method == "dtw_per_joint" else None

    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    if primary_segmentations:
        for key, primary_seg in primary_segmentations.items():
            reference_seg = reference_segmentations.get(key)
            if reference_seg is None:
                # Same config was applied to both, so this should not
                # happen unless the segmentation depends on the input
                # length in some unexpected way. Skip with a warning
                # rather than crash the whole run.
                continue
            pair_count = min(len(primary_seg.segments), len(reference_seg.segments))
            for i in range(pair_count):
                p_seg = primary_seg.segments[i]
                r_seg = reference_seg.segments[i]
                pairs.append(
                    (
                        f"{key}[{i}]",
                        primary_seq[p_seg.start : p_seg.end],
                        reference_seq[r_seg.start : r_seg.end],
                    )
                )
    else:
        pairs.append(("full_trial", primary_seq, reference_seq))

    for label, primary_slice, reference_slice in pairs:
        labels.append(label)
        if stage.method == "dtw_all":
            result = dtw_all(
                primary_slice,
                reference_slice,
                align=stage.align,
                representation=stage.representation,
                angle_triplets=stage.angle_triplets,
                nan_policy=stage.nan_policy,
            )
            distances.append(result.distance)
            paths.append(result.path)
        elif stage.method == "dtw_per_joint":
            assert per_joint_distances is not None
            per_joint_results = dtw_per_joint(
                primary_slice,
                reference_slice,
                align=stage.align,
                representation=stage.representation,
                angle_triplets=stage.angle_triplets,
                nan_policy=stage.nan_policy,
            )
            # "distance" for a per-joint run is the sum across units;
            # "per_joint_distances" carries the full breakdown.
            per_unit = [r.distance for r in per_joint_results]
            distances.append(float(sum(per_unit)))
            per_joint_distances.append(per_unit)
            # Store just the first joint's path as a representative —
            # per-joint paths are a list of equal length, but
            # reporting all of them on disk is almost always overkill.
            paths.append(per_joint_results[0].path if per_joint_results else [])
        else:  # "dtw_relation"
            assert stage.joint_i is not None
            assert stage.joint_j is not None
            result = _invoke_dtw_relation(
                primary_slice,
                reference_slice,
                joint_i=stage.joint_i,
                joint_j=stage.joint_j,
                align=stage.align,
                nan_policy=stage.nan_policy,
            )
            distances.append(result.distance)
            paths.append(result.path)

    return DtwResults(
        kind="dtw",
        method=stage.method,
        distances=distances,
        paths=paths,
        per_joint_distances=per_joint_distances,
        segment_labels=labels,
        summary=_summarize_distances(distances),
    )


def _invoke_dtw_relation(
    primary_slice: np.ndarray,
    reference_slice: np.ndarray,
    *,
    joint_i: int,
    joint_j: int,
    align: AlignMode,
    nan_policy: NanPolicy,
) -> DTWResult:
    """Isolating thin wrapper so test fakes can replace the call site cleanly."""
    return dtw_relation(
        primary_slice,
        reference_slice,
        joint_i,
        joint_j,
        align=align,
        nan_policy=nan_policy,
    )


def _run_stats(
    stage: StatsAnalysis,
    *,
    primary_seq: np.ndarray,
    primary_segmentations: dict[str, Segmentation],
) -> StatsResults:
    """Execute a stats analysis stage, returning :class:`StatsResults`."""
    labels: list[str] = []
    stats: list[FeatureSummary] = []

    if primary_segmentations:
        for key, seg in primary_segmentations.items():
            for i, segment in enumerate(seg.segments):
                labels.append(f"{key}[{i}]")
                signal = extract_signal(
                    primary_seq[segment.start : segment.end],
                    stage.extractor,
                )
                stats.append(_feature_summary(signal))
    else:
        labels.append("full_trial")
        signal = extract_signal(primary_seq, stage.extractor)
        stats.append(_feature_summary(signal))

    return StatsResults(kind="stats", statistics=stats, segment_labels=labels)


def _feature_summary(signal: np.ndarray) -> FeatureSummary:
    """Wrap :func:`extract_feature_statistics` output in a pydantic model."""
    raw = extract_feature_statistics(signal)
    return FeatureSummary(
        mean=raw.mean,
        std=raw.std,
        min=raw.min,
        max=raw.max,
        range=raw.range,
    )


def _summarize_distances(distances: list[float]) -> dict[str, float]:
    """Compute mean / p50 / p95 / p99 of a distance list.

    Empty inputs return an empty dict so the report's ``summary``
    field still round-trips through JSON without special cases.
    """
    if not distances:
        return {}
    arr = np.asarray(distances, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
