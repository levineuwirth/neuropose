"""Tests for :mod:`neuropose.analyzer.pipeline`.

This file covers the schema half of the pipeline:

- :class:`AnalysisConfig` parsing, including the discriminated unions
  for the segmentation and analysis stages, and the cross-field
  invariants enforced at parse time.
- :class:`AnalysisReport` construction + JSON round-trip, including
  the migration hook (schema_version defaults to CURRENT_VERSION).
- :func:`analysis_config_to_dict` JSON-safety.

The executor (``run_analysis``) gets its own test module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from neuropose.analyzer.pipeline import (
    AnalysisConfig,
    AnalysisReport,
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
    StatsAnalysis,
    StatsResults,
    analysis_config_to_dict,
)
from neuropose.migrations import CURRENT_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_dtw_config(tmp_path: Path) -> dict[str, Any]:
    """A minimal AnalysisConfig dict with dtw_all + reference."""
    return {
        "inputs": {
            "primary": str(tmp_path / "primary.json"),
            "reference": str(tmp_path / "reference.json"),
        },
        "analysis": {"kind": "dtw", "method": "dtw_all"},
        "output": {"report": str(tmp_path / "report.json")},
    }


def _minimal_stats_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "inputs": {"primary": str(tmp_path / "primary.json")},
        "analysis": {
            "kind": "stats",
            "extractor": {"kind": "joint_axis", "joint": 32, "axis": 1, "invert": False},
        },
        "output": {"report": str(tmp_path / "report.json")},
    }


# ---------------------------------------------------------------------------
# InputsConfig / PreprocessingConfig / OutputConfig
# ---------------------------------------------------------------------------


class TestInputsConfig:
    def test_primary_only(self, tmp_path: Path) -> None:
        cfg = InputsConfig(primary=tmp_path / "a.json")
        assert cfg.reference is None

    def test_primary_and_reference(self, tmp_path: Path) -> None:
        cfg = InputsConfig(
            primary=tmp_path / "a.json",
            reference=tmp_path / "b.json",
        )
        assert cfg.reference == tmp_path / "b.json"

    def test_extra_field_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            InputsConfig.model_validate(
                {
                    "primary": str(tmp_path / "a.json"),
                    "extra": "field",
                }
            )

    def test_frozen(self, tmp_path: Path) -> None:
        cfg = InputsConfig(primary=tmp_path / "a.json")
        with pytest.raises(ValidationError, match="frozen"):
            cfg.primary = tmp_path / "b.json"  # type: ignore[misc]


class TestPreprocessingConfig:
    def test_default_person_index_zero(self) -> None:
        cfg = PreprocessingConfig()
        assert cfg.person_index == 0

    def test_negative_person_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PreprocessingConfig(person_index=-1)


class TestOutputConfig:
    def test_report_path(self, tmp_path: Path) -> None:
        cfg = OutputConfig(report=tmp_path / "out.json")
        assert cfg.report == tmp_path / "out.json"


# ---------------------------------------------------------------------------
# Segmentation stage discriminated union
# ---------------------------------------------------------------------------


class TestSegmentationStage:
    def test_gait_cycles_parses_from_dict(self, tmp_path: Path) -> None:
        config_dict = _minimal_dtw_config(tmp_path)
        config_dict["segmentation"] = {
            "kind": "gait_cycles",
            "joint": "lhee",
            "axis": "y",
            "min_cycle_seconds": 0.5,
        }
        cfg = AnalysisConfig.model_validate(config_dict)
        assert isinstance(cfg.segmentation, GaitCyclesSegmentation)
        assert cfg.segmentation.joint == "lhee"
        assert cfg.segmentation.min_cycle_seconds == 0.5

    def test_bilateral_parses_from_dict(self, tmp_path: Path) -> None:
        config_dict = _minimal_dtw_config(tmp_path)
        config_dict["segmentation"] = {"kind": "gait_cycles_bilateral"}
        cfg = AnalysisConfig.model_validate(config_dict)
        assert isinstance(cfg.segmentation, GaitCyclesBilateralSegmentation)

    def test_extractor_parses_from_dict(self, tmp_path: Path) -> None:
        config_dict = _minimal_dtw_config(tmp_path)
        config_dict["segmentation"] = {
            "kind": "extractor",
            "extractor": {
                "kind": "joint_axis",
                "joint": 15,
                "axis": 1,
                "invert": False,
            },
            "label": "wrist_cycles",
            "min_distance_seconds": 0.5,
        }
        cfg = AnalysisConfig.model_validate(config_dict)
        assert isinstance(cfg.segmentation, ExtractorSegmentation)
        assert cfg.segmentation.label == "wrist_cycles"

    def test_unknown_kind_rejected(self, tmp_path: Path) -> None:
        config_dict = _minimal_dtw_config(tmp_path)
        config_dict["segmentation"] = {"kind": "unknown_method"}
        with pytest.raises(ValidationError):
            AnalysisConfig.model_validate(config_dict)

    def test_segmentation_omitted_is_none(self, tmp_path: Path) -> None:
        cfg = AnalysisConfig.model_validate(_minimal_dtw_config(tmp_path))
        assert cfg.segmentation is None

    def test_invalid_min_cycle_seconds_rejected(self, tmp_path: Path) -> None:
        config_dict = _minimal_dtw_config(tmp_path)
        config_dict["segmentation"] = {
            "kind": "gait_cycles",
            "min_cycle_seconds": 0.0,  # must be > 0
        }
        with pytest.raises(ValidationError):
            AnalysisConfig.model_validate(config_dict)


# ---------------------------------------------------------------------------
# Analysis stage discriminated union + cross-field invariants
# ---------------------------------------------------------------------------


class TestDtwAnalysisValidation:
    def test_dtw_relation_requires_joints(self) -> None:
        with pytest.raises(ValidationError, match="joint_i and joint_j"):
            DtwAnalysis(kind="dtw", method="dtw_relation")

    def test_dtw_relation_rejects_angles_representation(self) -> None:
        with pytest.raises(ValidationError, match="only supports representation='coords'"):
            DtwAnalysis(
                kind="dtw",
                method="dtw_relation",
                joint_i=0,
                joint_j=1,
                representation="angles",
                angle_triplets=[(0, 1, 2)],
            )

    def test_angles_requires_triplets(self) -> None:
        with pytest.raises(ValidationError, match="angle_triplets"):
            DtwAnalysis(
                kind="dtw",
                method="dtw_all",
                representation="angles",
            )

    def test_angles_with_empty_triplets_rejected(self) -> None:
        with pytest.raises(ValidationError, match="angle_triplets"):
            DtwAnalysis(
                kind="dtw",
                method="dtw_all",
                representation="angles",
                angle_triplets=[],
            )

    def test_happy_path_dtw_all_coords(self) -> None:
        analysis = DtwAnalysis(
            kind="dtw",
            method="dtw_all",
            align="procrustes_per_sequence",
            nan_policy="interpolate",
        )
        assert analysis.align == "procrustes_per_sequence"

    def test_happy_path_dtw_all_angles(self) -> None:
        analysis = DtwAnalysis(
            kind="dtw",
            method="dtw_all",
            representation="angles",
            angle_triplets=[(0, 1, 2), (3, 4, 5)],
        )
        assert len(analysis.angle_triplets or []) == 2

    def test_happy_path_dtw_relation(self) -> None:
        analysis = DtwAnalysis(
            kind="dtw",
            method="dtw_relation",
            joint_i=15,
            joint_j=23,
        )
        assert analysis.joint_i == 15


class TestAnalysisCrossStage:
    def test_dtw_without_reference_rejected(self, tmp_path: Path) -> None:
        config_dict = {
            "inputs": {"primary": str(tmp_path / "a.json")},
            "analysis": {"kind": "dtw", "method": "dtw_all"},
            "output": {"report": str(tmp_path / "out.json")},
        }
        with pytest.raises(ValidationError, match=r"inputs\.reference"):
            AnalysisConfig.model_validate(config_dict)

    def test_stats_with_reference_rejected(self, tmp_path: Path) -> None:
        config_dict = _minimal_stats_config(tmp_path)
        config_dict["inputs"]["reference"] = str(tmp_path / "b.json")
        with pytest.raises(ValidationError, match="primary only"):
            AnalysisConfig.model_validate(config_dict)

    def test_none_analysis_requires_no_reference(self, tmp_path: Path) -> None:
        # NoAnalysis is fine with either reference present or absent.
        config_dict = {
            "inputs": {"primary": str(tmp_path / "a.json")},
            "analysis": {"kind": "none"},
            "output": {"report": str(tmp_path / "out.json")},
        }
        cfg = AnalysisConfig.model_validate(config_dict)
        assert isinstance(cfg.analysis, NoAnalysis)


# ---------------------------------------------------------------------------
# Top-level AnalysisConfig
# ---------------------------------------------------------------------------


class TestAnalysisConfig:
    def test_minimal_dtw_config_parses(self, tmp_path: Path) -> None:
        cfg = AnalysisConfig.model_validate(_minimal_dtw_config(tmp_path))
        assert cfg.config_version == 1
        assert isinstance(cfg.analysis, DtwAnalysis)
        assert cfg.preprocessing.person_index == 0  # default

    def test_minimal_stats_config_parses(self, tmp_path: Path) -> None:
        cfg = AnalysisConfig.model_validate(_minimal_stats_config(tmp_path))
        assert isinstance(cfg.analysis, StatsAnalysis)

    def test_config_version_must_be_1(self, tmp_path: Path) -> None:
        config_dict = _minimal_dtw_config(tmp_path)
        config_dict["config_version"] = 99
        with pytest.raises(ValidationError):
            AnalysisConfig.model_validate(config_dict)

    def test_round_trip_json(self, tmp_path: Path) -> None:
        original = AnalysisConfig.model_validate(_minimal_dtw_config(tmp_path))
        serialised = original.model_dump_json()
        restored = AnalysisConfig.model_validate_json(serialised)
        assert restored == original

    def test_extra_top_level_field_rejected(self, tmp_path: Path) -> None:
        config_dict = _minimal_dtw_config(tmp_path)
        config_dict["unknown_key"] = "typo"
        with pytest.raises(ValidationError):
            AnalysisConfig.model_validate(config_dict)


# ---------------------------------------------------------------------------
# analysis_config_to_dict
# ---------------------------------------------------------------------------


class TestAnalysisConfigToDict:
    def test_returns_json_safe_dict(self, tmp_path: Path) -> None:
        cfg = AnalysisConfig.model_validate(_minimal_dtw_config(tmp_path))
        dumped = analysis_config_to_dict(cfg)
        # Paths must have become strings.
        assert isinstance(dumped["inputs"]["primary"], str)
        assert isinstance(dumped["output"]["report"], str)

    def test_round_trips_through_dict(self, tmp_path: Path) -> None:
        original = AnalysisConfig.model_validate(_minimal_dtw_config(tmp_path))
        dumped = analysis_config_to_dict(original)
        restored = AnalysisConfig.model_validate(dumped)
        assert restored == original


# ---------------------------------------------------------------------------
# Result sub-schemas
# ---------------------------------------------------------------------------


class TestDtwResults:
    def test_minimal_construction(self) -> None:
        res = DtwResults(
            kind="dtw",
            method="dtw_all",
            distances=[0.5],
            paths=[[(0, 0), (1, 1)]],
            segment_labels=["full_trial"],
            summary={"mean": 0.5},
        )
        assert res.kind == "dtw"

    def test_per_joint_distances_shape_is_free(self) -> None:
        # No validator enforces that per_joint_distances outer length
        # matches distances — that's run-time semantics of the
        # executor. Still, verify the field round-trips.
        res = DtwResults(
            kind="dtw",
            method="dtw_per_joint",
            distances=[0.1, 0.2],
            paths=[[(0, 0)], [(0, 0)]],
            per_joint_distances=[[0.05, 0.05], [0.1, 0.1]],
            segment_labels=["rhee_cycles[0]", "rhee_cycles[1]"],
            summary={"mean": 0.15},
        )
        assert res.per_joint_distances is not None


class TestStatsResults:
    def test_round_trip(self) -> None:
        res = StatsResults(
            kind="stats",
            statistics=[
                FeatureSummary(mean=1.0, std=0.1, min=0.8, max=1.2, range=0.4),
                FeatureSummary(mean=1.1, std=0.2, min=0.7, max=1.5, range=0.8),
            ],
            segment_labels=["rhee_cycles[0]", "rhee_cycles[1]"],
        )
        dumped = res.model_dump_json()
        restored = StatsResults.model_validate_json(dumped)
        assert restored == res


class TestNoResults:
    def test_construction(self) -> None:
        res = NoResults(kind="none")
        assert res.kind == "none"


# ---------------------------------------------------------------------------
# AnalysisReport
# ---------------------------------------------------------------------------


def _make_report(tmp_path: Path) -> AnalysisReport:
    config = AnalysisConfig.model_validate(_minimal_dtw_config(tmp_path))
    return AnalysisReport(
        config=config,
        primary=InputSummary(
            path=tmp_path / "primary.json",
            frame_count=300,
            fps=30.0,
        ),
        reference=InputSummary(
            path=tmp_path / "reference.json",
            frame_count=300,
            fps=30.0,
        ),
        results=DtwResults(
            kind="dtw",
            method="dtw_all",
            distances=[0.42],
            paths=[[(0, 0), (1, 1)]],
            segment_labels=["full_trial"],
            summary={"mean": 0.42, "p50": 0.42},
        ),
    )


class TestAnalysisReport:
    def test_schema_version_defaults_to_current(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        assert report.schema_version == CURRENT_VERSION

    def test_round_trip_json(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        serialised = report.model_dump_json()
        restored = AnalysisReport.model_validate_json(serialised)
        assert restored == report

    def test_empty_segmentations_default(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        assert report.segmentations == {}

    def test_extra_field_rejected(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        dumped = report.model_dump(mode="json")
        dumped["mystery_field"] = 1
        with pytest.raises(ValidationError):
            AnalysisReport.model_validate(dumped)
