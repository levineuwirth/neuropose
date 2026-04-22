"""Tests for :mod:`neuropose.analyzer.pipeline`.

Covers both halves of the pipeline:

- **Schemas** — :class:`AnalysisConfig` parsing (discriminated unions
  for segmentation and analysis stages, cross-field invariants),
  :class:`AnalysisReport` construction + JSON round-trip (including
  the migration hook on ``schema_version``), and
  :func:`analysis_config_to_dict` JSON-safety.
- **Executor** — :func:`run_analysis` dispatches to each analysis kind
  (dtw / stats / none) with and without segmentation; provenance is
  inherited from the primary input with ``analysis_config``
  populated; :func:`load_config`, :func:`save_report`, and
  :func:`load_report` round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
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
    load_config,
    load_report,
    run_analysis,
    save_report,
)
from neuropose.io import Provenance, VideoPredictions, save_video_predictions
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


# ---------------------------------------------------------------------------
# Executor helpers
# ---------------------------------------------------------------------------

NUM_JOINTS = 43


def _heel_signal(num_cycles: int, frames_per_cycle: int, amplitude: float = 100.0) -> np.ndarray:
    """Clean sinusoid stand-in for a heel's vertical trace."""
    import math

    total = num_cycles * frames_per_cycle
    t = np.linspace(0.0, num_cycles * 2.0 * math.pi, total, endpoint=False)
    return (np.sin(t) * amplitude + amplitude).astype(float)


def _build_predictions(
    signal: np.ndarray,
    joint: int,
    *,
    axis: int = 1,
    fps: float = 30.0,
    provenance: Provenance | None = None,
) -> VideoPredictions:
    """Build a VideoPredictions whose ``joint``'s ``axis`` follows ``signal``."""
    frames = {}
    for i, value in enumerate(signal):
        poses = [[[0.0, 0.0, 0.0] for _ in range(NUM_JOINTS)]]
        poses[0][joint][axis] = float(value)
        frames[f"frame_{i:06d}"] = {
            "boxes": [[0.0, 0.0, 1.0, 1.0, 0.9]],
            "poses3d": poses,
            "poses2d": [[[0.0, 0.0]] * NUM_JOINTS],
        }
    return VideoPredictions.model_validate(
        {
            "metadata": {
                "frame_count": len(signal),
                "fps": fps,
                "width": 640,
                "height": 480,
            },
            "frames": frames,
            "provenance": provenance.model_dump() if provenance is not None else None,
        }
    )


def _write_heel_trial(
    tmp_path: Path,
    filename: str,
    *,
    joint: int,
    num_cycles: int = 4,
    frames_per_cycle: int = 30,
    amplitude: float = 100.0,
    provenance: Provenance | None = None,
) -> Path:
    """Write a heel-trace VideoPredictions JSON and return its path."""
    signal = _heel_signal(num_cycles, frames_per_cycle, amplitude=amplitude)
    preds = _build_predictions(signal, joint=joint, provenance=provenance)
    path = tmp_path / filename
    save_video_predictions(path, preds)
    return path


def _fake_provenance(sha: str = "a" * 64) -> Provenance:
    return Provenance(
        model_sha256=sha,
        model_filename="fake_model.tar.gz",
        tensorflow_version="2.18.0",
        numpy_version="1.26.0",
        neuropose_version="0.0.0",
        python_version="3.11.0",
    )


# ---------------------------------------------------------------------------
# Executor: run_analysis
# ---------------------------------------------------------------------------


class TestRunAnalysisDtwFullTrial:
    def test_dtw_all_unsegmented_yields_one_distance(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"])
        reference = _write_heel_trial(tmp_path, "b.json", joint=JOINT_INDEX["rhee"])
        report_path = tmp_path / "report.json"
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary), "reference": str(reference)},
                "analysis": {"kind": "dtw", "method": "dtw_all"},
                "output": {"report": str(report_path)},
            }
        )
        report = run_analysis(config)
        assert isinstance(report.results, DtwResults)
        assert report.results.segment_labels == ["full_trial"]
        assert len(report.results.distances) == 1
        # Identical inputs → distance 0.
        assert report.results.distances[0] == pytest.approx(0.0, abs=1e-9)

    def test_dtw_all_different_trials_positive_distance(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"], amplitude=100.0)
        reference = _write_heel_trial(
            tmp_path, "b.json", joint=JOINT_INDEX["rhee"], amplitude=200.0
        )
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary), "reference": str(reference)},
                "analysis": {"kind": "dtw", "method": "dtw_all"},
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert isinstance(report.results, DtwResults)
        assert report.results.distances[0] > 0.0
        assert "mean" in report.results.summary


class TestRunAnalysisDtwSegmented:
    def test_dtw_with_gait_cycles_produces_per_segment_distances(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"], num_cycles=4)
        reference = _write_heel_trial(tmp_path, "b.json", joint=JOINT_INDEX["rhee"], num_cycles=4)
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary), "reference": str(reference)},
                "segmentation": {"kind": "gait_cycles", "joint": "rhee"},
                "analysis": {"kind": "dtw", "method": "dtw_all"},
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert isinstance(report.results, DtwResults)
        # 4 cycles detected on both → 4 paired distances.
        assert len(report.results.distances) == 4
        assert all(label.startswith("rhee_cycles[") for label in report.results.segment_labels)

    def test_dtw_bilateral_produces_distances_per_side(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        # Put the heel trace on both lhee and rhee.
        rng_signal = _heel_signal(num_cycles=3, frames_per_cycle=30)
        frames = {}
        for i, value in enumerate(rng_signal):
            poses = [[[0.0, 0.0, 0.0] for _ in range(NUM_JOINTS)]]
            poses[0][JOINT_INDEX["lhee"]][1] = float(value)
            poses[0][JOINT_INDEX["rhee"]][1] = float(value)
            frames[f"frame_{i:06d}"] = {
                "boxes": [[0.0, 0.0, 1.0, 1.0, 0.9]],
                "poses3d": poses,
                "poses2d": [[[0.0, 0.0]] * NUM_JOINTS],
            }
        preds = VideoPredictions.model_validate(
            {
                "metadata": {
                    "frame_count": len(rng_signal),
                    "fps": 30.0,
                    "width": 640,
                    "height": 480,
                },
                "frames": frames,
            }
        )
        primary = tmp_path / "a.json"
        reference = tmp_path / "b.json"
        save_video_predictions(primary, preds)
        save_video_predictions(reference, preds)
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary), "reference": str(reference)},
                "segmentation": {"kind": "gait_cycles_bilateral"},
                "analysis": {"kind": "dtw", "method": "dtw_all"},
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert isinstance(report.results, DtwResults)
        # 3 cycles * 2 sides.
        assert len(report.results.distances) == 6
        left = [lbl for lbl in report.results.segment_labels if lbl.startswith("left_heel_strikes")]
        right = [
            lbl for lbl in report.results.segment_labels if lbl.startswith("right_heel_strikes")
        ]
        assert len(left) == 3
        assert len(right) == 3
        # Identical primary and reference → all distances zero.
        for d in report.results.distances:
            assert d == pytest.approx(0.0, abs=1e-9)

    def test_dtw_per_joint_populates_per_joint_distances(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"])
        reference = _write_heel_trial(tmp_path, "b.json", joint=JOINT_INDEX["rhee"])
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary), "reference": str(reference)},
                "analysis": {"kind": "dtw", "method": "dtw_per_joint"},
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert isinstance(report.results, DtwResults)
        assert report.results.per_joint_distances is not None
        assert len(report.results.per_joint_distances) == 1  # unsegmented → one pair
        assert len(report.results.per_joint_distances[0]) == NUM_JOINTS


class TestRunAnalysisStats:
    def test_stats_unsegmented_single_block(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"])
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary)},
                "analysis": {
                    "kind": "stats",
                    "extractor": {
                        "kind": "joint_axis",
                        "joint": JOINT_INDEX["rhee"],
                        "axis": 1,
                        "invert": False,
                    },
                },
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert isinstance(report.results, StatsResults)
        assert report.results.segment_labels == ["full_trial"]
        assert len(report.results.statistics) == 1
        stat = report.results.statistics[0]
        assert isinstance(stat, FeatureSummary)
        assert stat.max > stat.min  # Signal oscillates.

    def test_stats_with_segmentation_emits_per_segment(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"], num_cycles=3)
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary)},
                "segmentation": {"kind": "gait_cycles", "joint": "rhee"},
                "analysis": {
                    "kind": "stats",
                    "extractor": {
                        "kind": "joint_axis",
                        "joint": JOINT_INDEX["rhee"],
                        "axis": 1,
                        "invert": False,
                    },
                },
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert isinstance(report.results, StatsResults)
        assert len(report.results.statistics) == 3


class TestRunAnalysisNone:
    def test_none_analysis_returns_no_results(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"])
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary)},
                "analysis": {"kind": "none"},
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert isinstance(report.results, NoResults)

    def test_none_with_segmentation_still_emits_segmentations(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"])
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary)},
                "segmentation": {"kind": "gait_cycles", "joint": "rhee"},
                "analysis": {"kind": "none"},
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert isinstance(report.results, NoResults)
        assert "rhee_cycles" in report.segmentations
        assert len(report.segmentations["rhee_cycles"].segments) > 0


# ---------------------------------------------------------------------------
# Provenance inheritance
# ---------------------------------------------------------------------------


class TestRunAnalysisProvenance:
    def test_inherits_primary_provenance_and_stamps_config(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        provenance = _fake_provenance()
        primary = _write_heel_trial(
            tmp_path, "a.json", joint=JOINT_INDEX["rhee"], provenance=provenance
        )
        reference = _write_heel_trial(tmp_path, "b.json", joint=JOINT_INDEX["rhee"])
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary), "reference": str(reference)},
                "analysis": {"kind": "dtw", "method": "dtw_all"},
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert report.provenance is not None
        # Model SHA inherited from primary.
        assert report.provenance.model_sha256 == provenance.model_sha256
        # analysis_config populated with the serialised config.
        assert report.provenance.analysis_config is not None
        assert report.provenance.analysis_config["config_version"] == 1

    def test_no_primary_provenance_yields_none_report_provenance(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"])
        reference = _write_heel_trial(tmp_path, "b.json", joint=JOINT_INDEX["rhee"])
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary), "reference": str(reference)},
                "analysis": {"kind": "dtw", "method": "dtw_all"},
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert report.provenance is None

    def test_input_summaries_track_paths_and_metadata(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"], num_cycles=5)
        reference = _write_heel_trial(tmp_path, "b.json", joint=JOINT_INDEX["rhee"], num_cycles=3)
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary), "reference": str(reference)},
                "analysis": {"kind": "dtw", "method": "dtw_all"},
                "output": {"report": str(tmp_path / "r.json")},
            }
        )
        report = run_analysis(config)
        assert report.primary.path == primary
        assert report.primary.frame_count == 5 * 30
        assert report.reference is not None
        assert report.reference.frame_count == 3 * 30


# ---------------------------------------------------------------------------
# load_config / save_report / load_report
# ---------------------------------------------------------------------------


class TestLoadSave:
    def test_load_config_parses_yaml(self, tmp_path: Path) -> None:
        config_dict = {
            "inputs": {
                "primary": str(tmp_path / "a.json"),
                "reference": str(tmp_path / "b.json"),
            },
            "analysis": {"kind": "dtw", "method": "dtw_all"},
            "output": {"report": str(tmp_path / "r.json")},
        }
        yaml_path = tmp_path / "exp.yaml"
        yaml_path.write_text(yaml.safe_dump(config_dict))
        loaded = load_config(yaml_path)
        assert isinstance(loaded.analysis, DtwAnalysis)

    def test_load_config_empty_file_fails_cleanly(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ValidationError):
            load_config(empty)

    def test_load_config_rejects_malformed_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        # Unclosed flow-style mapping — yaml.safe_load raises here.
        bad.write_text("inputs: {primary: foo\n")
        with pytest.raises(yaml.YAMLError):
            load_config(bad)

    def test_save_report_round_trip(self, tmp_path: Path) -> None:
        from neuropose.analyzer.segment import JOINT_INDEX

        primary = _write_heel_trial(tmp_path, "a.json", joint=JOINT_INDEX["rhee"])
        reference = _write_heel_trial(tmp_path, "b.json", joint=JOINT_INDEX["rhee"])
        config = AnalysisConfig.model_validate(
            {
                "inputs": {"primary": str(primary), "reference": str(reference)},
                "analysis": {"kind": "dtw", "method": "dtw_all"},
                "output": {"report": str(tmp_path / "report.json")},
            }
        )
        report = run_analysis(config)
        report_path = tmp_path / "report.json"
        save_report(report_path, report)
        assert report_path.exists()

        restored = load_report(report_path)
        assert restored == report

    def test_save_report_is_atomic(self, tmp_path: Path) -> None:
        """The saver writes via a sibling .tmp path and renames."""
        report = _make_report(tmp_path)
        report_path = tmp_path / "subdir" / "report.json"
        save_report(report_path, report)
        # Parent directory was created.
        assert report_path.exists()
        # No .tmp sibling left behind.
        assert not (report_path.with_suffix(report_path.suffix + ".tmp")).exists()

    def test_load_report_rejects_future_schema(self, tmp_path: Path) -> None:
        """Future schema_version surfaces as a migration error."""
        from neuropose.migrations import FutureSchemaError

        future = {"schema_version": CURRENT_VERSION + 1}
        path = tmp_path / "future.json"
        path.write_text(json.dumps(future))
        with pytest.raises(FutureSchemaError):
            load_report(path)
