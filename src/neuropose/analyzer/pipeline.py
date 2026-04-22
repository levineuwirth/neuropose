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
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuropose.analyzer.dtw import AlignMode, NanPolicy, Representation
from neuropose.analyzer.segment import AxisLetter
from neuropose.io import ExtractorSpec, Provenance, Segmentation
from neuropose.migrations import CURRENT_VERSION

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
