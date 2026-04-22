"""Example-config sanity integration tests.

Every YAML config under ``examples/analysis/`` is exercised here in
two ways:

1. ``test_<name>_parses`` — :func:`load_config` accepts the YAML,
   i.e. the file matches the current :class:`AnalysisConfig` schema
   including cross-field invariants. Catches silent drift between the
   example configs and the schema they claim to exercise.
2. ``test_<name>_runs`` — the example's pipeline runs end-to-end
   against synthetic predictions. Paths in the YAML are overwritten
   with test fixtures before :func:`run_analysis` is invoked;
   everything else (stages, thresholds, triplets) is used verbatim.

These tests are deliberately not marked ``slow`` — they use synthetic
fixtures and do not touch the MeTRAbs SavedModel, so they run in the
default unit-test suite.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from neuropose.analyzer.pipeline import (
    AnalysisConfig,
    AnalysisReport,
    DtwResults,
    load_config,
    run_analysis,
)
from neuropose.analyzer.segment import JOINT_INDEX
from neuropose.io import VideoPredictions, save_video_predictions

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "analysis"

NUM_JOINTS = 43


def _sinusoid(num_cycles: int, frames_per_cycle: int, amplitude: float = 100.0) -> np.ndarray:
    total = num_cycles * frames_per_cycle
    t = np.linspace(0.0, num_cycles * 2.0 * math.pi, total, endpoint=False)
    return (np.sin(t) * amplitude + amplitude).astype(float)


def _write_trial(
    path: Path,
    *,
    num_cycles: int = 4,
    frames_per_cycle: int = 30,
    seed: int = 0,
) -> Path:
    """Write a synthetic VideoPredictions with every joint oscillating.

    Joint ``0`` gets a reproducible RNG-driven trace so Procrustes has
    something non-degenerate to align. All other joints get their own
    phase-shifted sinusoid so joint-angle triplets and per-joint DTW
    have signal to act on.
    """
    rng = np.random.default_rng(seed)
    base = _sinusoid(num_cycles, frames_per_cycle)
    total = base.shape[0]
    frames: dict[str, dict] = {}
    for frame_idx in range(total):
        poses = [[[0.0, 0.0, 0.0] for _ in range(NUM_JOINTS)]]
        for j in range(NUM_JOINTS):
            # Unique per-joint position so no triplet is degenerate.
            phase = rng.uniform(0.0, 2.0 * math.pi)
            amplitude = 30.0 + 10.0 * (j % 5)
            offset = float(j) * 15.0
            poses[0][j][0] = offset + amplitude * math.cos(
                2.0 * math.pi * frame_idx / frames_per_cycle + phase
            )
            poses[0][j][1] = offset * 0.5 + base[frame_idx] + 5.0 * j
            poses[0][j][2] = 3.0 * j
        frames[f"frame_{frame_idx:06d}"] = {
            "boxes": [[0.0, 0.0, 1.0, 1.0, 0.9]],
            "poses3d": poses,
            "poses2d": [[[0.0, 0.0]] * NUM_JOINTS],
        }
    preds = VideoPredictions.model_validate(
        {
            "metadata": {
                "frame_count": total,
                "fps": float(frames_per_cycle),
                "width": 640,
                "height": 480,
            },
            "frames": frames,
        }
    )
    save_video_predictions(path, preds)
    return path


@pytest.fixture
def example_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (primary_path, reference_path, report_path) under ``tmp_path``."""
    primary = _write_trial(tmp_path / "primary.json", seed=1)
    reference = _write_trial(tmp_path / "reference.json", seed=2)
    report = tmp_path / "report.json"
    return primary, reference, report


def _rewrite_paths(
    example_path: Path, primary: Path, reference: Path, report: Path
) -> AnalysisConfig:
    """Load an example YAML and rewrite inputs/output paths to fixtures.

    Tests run against synthetic predictions in ``tmp_path``; the
    example YAML's hardcoded ``data/*.json`` paths would never resolve
    otherwise.
    """
    config = load_config(example_path)
    update: dict = {
        "inputs": config.inputs.model_copy(update={"primary": primary, "reference": reference}),
        "output": config.output.model_copy(update={"report": report}),
    }
    return config.model_copy(update=update)


class TestMinimalExample:
    def test_minimal_parses(self) -> None:
        config = load_config(EXAMPLES_DIR / "minimal.yaml")
        assert isinstance(config, AnalysisConfig)
        assert config.analysis.kind == "dtw"
        assert config.segmentation is None

    def test_minimal_runs(self, example_fixtures: tuple[Path, Path, Path]) -> None:
        primary, reference, report = example_fixtures
        config = _rewrite_paths(EXAMPLES_DIR / "minimal.yaml", primary, reference, report)
        result = run_analysis(config)
        assert isinstance(result, AnalysisReport)
        assert isinstance(result.results, DtwResults)
        # Unsegmented → one distance.
        assert result.results.segment_labels == ["full_trial"]
        assert len(result.results.distances) == 1


class TestPaperCExample:
    def test_paper_c_parses(self) -> None:
        config = load_config(EXAMPLES_DIR / "paper_c_headline.yaml")
        assert config.analysis.kind == "dtw"
        assert config.segmentation is not None
        assert config.segmentation.kind == "gait_cycles_bilateral"
        # Joint triplets must be in range for berkeley_mhad_43.
        assert config.analysis.kind == "dtw"
        angle_triplets = config.analysis.angle_triplets  # type: ignore[union-attr]
        assert angle_triplets is not None
        for a, b, c in angle_triplets:
            for idx in (a, b, c):
                assert 0 <= idx < NUM_JOINTS

    def test_paper_c_runs(self, example_fixtures: tuple[Path, Path, Path]) -> None:
        primary, reference, report = example_fixtures
        config = _rewrite_paths(EXAMPLES_DIR / "paper_c_headline.yaml", primary, reference, report)
        result = run_analysis(config)
        assert isinstance(result.results, DtwResults)
        # Bilateral segmentation → distances labelled per side.
        assert any(lbl.startswith("left_heel_strikes") for lbl in result.results.segment_labels)
        assert any(lbl.startswith("right_heel_strikes") for lbl in result.results.segment_labels)

    def test_paper_c_uses_documented_knee_triplets(self) -> None:
        """The Paper C config must target knee-flexion joint triplets.

        Safety net: if someone edits the YAML and breaks the joint
        references, this test catches it before the example silently
        starts computing the wrong angles.
        """
        config = load_config(EXAMPLES_DIR / "paper_c_headline.yaml")
        assert config.analysis.kind == "dtw"
        triplets = config.analysis.angle_triplets  # type: ignore[union-attr]
        assert triplets is not None
        # Left knee flex = hip → knee → ankle.
        assert (
            JOINT_INDEX["lhipb"],
            JOINT_INDEX["lkne"],
            JOINT_INDEX["lank"],
        ) in triplets or (
            JOINT_INDEX["lhipf"],
            JOINT_INDEX["lkne"],
            JOINT_INDEX["lank"],
        ) in triplets


class TestPerJointDebugExample:
    def test_per_joint_debug_parses(self) -> None:
        config = load_config(EXAMPLES_DIR / "per_joint_debug.yaml")
        assert config.analysis.kind == "dtw"
        assert config.analysis.method == "dtw_per_joint"  # type: ignore[union-attr]

    def test_per_joint_debug_runs(self, example_fixtures: tuple[Path, Path, Path]) -> None:
        primary, reference, report = example_fixtures
        config = _rewrite_paths(EXAMPLES_DIR / "per_joint_debug.yaml", primary, reference, report)
        result = run_analysis(config)
        assert isinstance(result.results, DtwResults)
        # dtw_per_joint → per_joint_distances populated.
        assert result.results.per_joint_distances is not None
        # Inner length must match num_joints for the coords
        # representation.
        for per_seg in result.results.per_joint_distances:
            assert len(per_seg) == NUM_JOINTS
