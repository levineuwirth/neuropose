"""Tests for :mod:`neuropose.analyzer.features`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuropose.analyzer.features import (
    AlignmentDiagnostics,
    FeatureStatistics,
    extract_feature_statistics,
    extract_joint_angles,
    find_peaks,
    normalize_pose_sequence,
    pad_sequences,
    predictions_to_numpy,
    procrustes_align,
)
from neuropose.io import VideoPredictions

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_predictions(num_frames: int, num_persons: int = 1) -> VideoPredictions:
    """Build a minimal VideoPredictions object for tests."""
    frames = {}
    for i in range(num_frames):
        frames[f"frame_{i:06d}"] = {
            "boxes": [[0.0, 0.0, 1.0, 1.0, 0.9]] * num_persons,
            "poses3d": [[[float(i), float(i) * 2, float(i) * 3], [0.0, 0.0, 0.0]]] * num_persons,
            "poses2d": [[[0.0, 0.0], [1.0, 1.0]]] * num_persons,
        }
    return VideoPredictions.model_validate(
        {
            "metadata": {
                "frame_count": num_frames,
                "fps": 30.0,
                "width": 640,
                "height": 480,
            },
            "frames": frames,
        }
    )


# ---------------------------------------------------------------------------
# predictions_to_numpy
# ---------------------------------------------------------------------------


class TestPredictionsToNumpy:
    def test_single_person_shape(self) -> None:
        predictions = _make_predictions(num_frames=4)
        arr = predictions_to_numpy(predictions)
        assert arr.shape == (4, 2, 3)
        assert arr.dtype == np.float64

    def test_values_preserved(self) -> None:
        predictions = _make_predictions(num_frames=3)
        arr = predictions_to_numpy(predictions)
        # Frame i has joint 0 at (i, 2i, 3i) per _make_predictions.
        for i in range(3):
            np.testing.assert_allclose(arr[i, 0], [i, 2 * i, 3 * i])
            np.testing.assert_allclose(arr[i, 1], [0, 0, 0])

    def test_person_index_out_of_range(self) -> None:
        predictions = _make_predictions(num_frames=2, num_persons=1)
        with pytest.raises(ValueError, match="out of range"):
            predictions_to_numpy(predictions, person_index=1)

    def test_multi_person_with_explicit_index(self) -> None:
        predictions = _make_predictions(num_frames=2, num_persons=2)
        arr = predictions_to_numpy(predictions, person_index=1)
        assert arr.shape == (2, 2, 3)

    def test_empty_predictions_raises(self) -> None:
        predictions = _make_predictions(num_frames=0)
        with pytest.raises(ValueError, match="zero frames"):
            predictions_to_numpy(predictions)


# ---------------------------------------------------------------------------
# normalize_pose_sequence
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_uniform_preserves_ratio(self) -> None:
        # (frames, joints, 3) — one joint per frame, two frames.
        seq = np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[3.0, 6.0, 9.0]],
            ]
        )
        # Ranges: x=3, y=6, z=9. Uniform scale = 9. All values / 9.
        result = normalize_pose_sequence(seq, axis_wise=False)
        np.testing.assert_allclose(result, seq / 9.0)

    def test_axis_wise_each_axis_to_unit_range(self) -> None:
        seq = np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[3.0, 6.0, 9.0]],
            ]
        )
        result = normalize_pose_sequence(seq, axis_wise=True)
        # Per-axis normalization → each axis's max becomes 1.
        np.testing.assert_allclose(result[0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result[1, 0], [1.0, 1.0, 1.0])

    def test_does_not_mutate_input(self) -> None:
        seq = np.array([[[0.0, 0.0, 0.0]], [[1.0, 2.0, 3.0]]])
        before = seq.copy()
        normalize_pose_sequence(seq)
        np.testing.assert_array_equal(seq, before)

    def test_degenerate_sequence_rejected(self) -> None:
        seq = np.zeros((3, 2, 3))
        with pytest.raises(ValueError, match="degenerate"):
            normalize_pose_sequence(seq)

    def test_bad_shape_rejected(self) -> None:
        seq = np.zeros((3, 2))  # Missing the xyz axis.
        with pytest.raises(ValueError, match="expected"):
            normalize_pose_sequence(seq)

    def test_axis_wise_with_zero_axis_keeps_it_zero(self) -> None:
        # Sequence where the Z axis never moves — axis_wise should not
        # divide by zero; the Z column should remain at 0.
        seq = np.array(
            [
                [[0.0, 0.0, 5.0]],
                [[4.0, 8.0, 5.0]],
            ]
        )
        result = normalize_pose_sequence(seq, axis_wise=True)
        np.testing.assert_allclose(result[:, 0, 2], [0.0, 0.0])


# ---------------------------------------------------------------------------
# pad_sequences
# ---------------------------------------------------------------------------


class TestPadSequences:
    def test_pads_to_max_when_target_length_none(self) -> None:
        a = np.zeros((3, 2, 3))
        b = np.zeros((5, 2, 3))
        padded = pad_sequences([a, b])
        assert all(seq.shape[0] == 5 for seq in padded)

    def test_pads_to_explicit_target_length(self) -> None:
        a = np.zeros((3, 2, 3))
        padded = pad_sequences([a], target_length=10)
        assert padded[0].shape == (10, 2, 3)

    def test_edge_padding_repeats_last_frame(self) -> None:
        a = np.array([[[1.0, 2.0, 3.0]]])  # shape (1, 1, 3)
        padded = pad_sequences([a], target_length=4)
        # All 4 frames should equal the original single frame.
        for i in range(4):
            np.testing.assert_allclose(padded[0][i, 0], [1.0, 2.0, 3.0])

    def test_truncates_longer_than_target(self) -> None:
        a = np.zeros((10, 2, 3))
        padded = pad_sequences([a], target_length=4)
        assert padded[0].shape == (4, 2, 3)

    def test_does_not_mutate_input(self) -> None:
        a = np.zeros((3, 2, 3))
        pad_sequences([a], target_length=5)
        assert a.shape == (3, 2, 3)

    def test_mismatched_trailing_shape_rejected(self) -> None:
        a = np.zeros((3, 2, 3))
        b = np.zeros((3, 4, 3))  # Different joint count.
        with pytest.raises(ValueError, match="trailing shape"):
            pad_sequences([a, b])

    def test_empty_input_with_target(self) -> None:
        assert pad_sequences([], target_length=5) == []

    def test_empty_input_without_target_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pad_sequences([])


# ---------------------------------------------------------------------------
# extract_joint_angles
# ---------------------------------------------------------------------------


class TestExtractJointAngles:
    def test_right_angle(self) -> None:
        # Three joints forming a right angle at joint 1.
        # joint 0 at (1, 0, 0), joint 1 at origin, joint 2 at (0, 1, 0).
        sequence = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        angles = extract_joint_angles(sequence, triplets=[(0, 1, 2)])
        assert angles.shape == (1, 1)
        assert angles[0, 0] == pytest.approx(math.pi / 2)

    def test_collinear_gives_pi(self) -> None:
        sequence = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            ]
        )
        angles = extract_joint_angles(sequence, triplets=[(0, 1, 2)])
        assert angles[0, 0] == pytest.approx(math.pi)

    def test_multiple_triplets(self) -> None:
        sequence = np.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ]
        )
        # Right angle at 1 (first triplet) and right angle at 1 again
        # using joint 3 as the other arm — still 90°.
        angles = extract_joint_angles(sequence, triplets=[(0, 1, 2), (0, 1, 3)])
        assert angles.shape == (1, 2)
        assert angles[0, 0] == pytest.approx(math.pi / 2)
        assert angles[0, 1] == pytest.approx(math.pi / 2)

    def test_zero_length_vector_yields_nan(self) -> None:
        # Joints 0 and 1 coincide → v1 is the zero vector → NaN angle.
        sequence = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ]
        )
        angles = extract_joint_angles(sequence, triplets=[(0, 1, 2)])
        assert math.isnan(angles[0, 0])

    def test_out_of_range_index_rejected(self) -> None:
        sequence = np.zeros((1, 3, 3))
        with pytest.raises(ValueError, match="out of range"):
            extract_joint_angles(sequence, triplets=[(0, 1, 10)])


# ---------------------------------------------------------------------------
# extract_feature_statistics
# ---------------------------------------------------------------------------


class TestExtractFeatureStatistics:
    def test_basic_stats(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = extract_feature_statistics(values)
        assert isinstance(stats, FeatureStatistics)
        assert stats.mean == pytest.approx(3.0)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(5.0)
        assert stats.range == pytest.approx(4.0)
        assert stats.std == pytest.approx(np.std(values))

    def test_rejects_2d(self) -> None:
        values = np.zeros((3, 3))
        with pytest.raises(ValueError, match="1D"):
            extract_feature_statistics(values)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            extract_feature_statistics(np.array([]))


# ---------------------------------------------------------------------------
# find_peaks
# ---------------------------------------------------------------------------


class TestFindPeaks:
    def test_sine_wave_peaks(self) -> None:
        # A sine wave over two full cycles has two peaks at quarter
        # cycles — roughly at t=pi/2 and t=5pi/2 given 4pi duration.
        t = np.linspace(0, 4 * np.pi, 401)
        values = np.sin(t)
        indices = find_peaks(values)
        assert indices.ndim == 1
        assert len(indices) == 2

    def test_flat_signal_has_no_peaks(self) -> None:
        indices = find_peaks(np.zeros(100))
        assert indices.size == 0

    def test_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            find_peaks(np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# procrustes_align
# ---------------------------------------------------------------------------


def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Rotation matrix about the Z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _skeleton(num_joints: int = 8, seed: int = 0) -> np.ndarray:
    """A deterministic, non-degenerate single-frame skeleton."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_joints, 3))


class TestProcrustesAlignPerSequence:
    def test_identical_sequences_yield_identity_transform(self) -> None:
        sequence = _skeleton()[np.newaxis, :, :].repeat(3, axis=0)  # (3, 8, 3)
        aligned, target, diag = procrustes_align(sequence, sequence, mode="per_sequence")
        np.testing.assert_allclose(aligned, sequence, atol=1e-10)
        np.testing.assert_array_equal(target, sequence)
        assert diag.mode == "per_sequence"
        assert diag.rotation_deg == pytest.approx(0.0, abs=1e-6)
        assert diag.translation == pytest.approx(0.0, abs=1e-9)
        assert diag.scale == pytest.approx(1.0)

    def test_recovers_known_rotation(self) -> None:
        # Build a reference sequence; construct the source by rotating it
        # about Z, then verify alignment returns the reference up to
        # floating-point error.
        rotation = _rotation_matrix_z(np.deg2rad(37.0))
        reference = _skeleton(num_joints=10)[np.newaxis, :, :].repeat(4, axis=0)
        source = reference @ rotation.T
        aligned, _, diag = procrustes_align(source, reference, mode="per_sequence")
        np.testing.assert_allclose(aligned, reference, atol=1e-8)
        # The recovered rotation's magnitude should be the original 37°.
        assert diag.rotation_deg == pytest.approx(37.0, abs=1e-4)

    def test_recovers_known_translation(self) -> None:
        reference = _skeleton()[np.newaxis, :, :].repeat(5, axis=0)
        translation = np.array([10.0, -4.5, 2.25])
        source = reference + translation
        aligned, _, diag = procrustes_align(source, reference, mode="per_sequence")
        np.testing.assert_allclose(aligned, reference, atol=1e-9)
        # rotation_deg may be numerically tiny but not exactly 0.
        assert diag.rotation_deg == pytest.approx(0.0, abs=1e-4)
        assert diag.translation == pytest.approx(np.linalg.norm(translation), rel=1e-6)

    def test_recovers_combined_rotation_and_translation(self) -> None:
        rotation = _rotation_matrix_z(np.deg2rad(-12.0))
        translation = np.array([1.0, 2.0, 3.0])
        reference = _skeleton(num_joints=6)[np.newaxis, :, :].repeat(3, axis=0)
        source = reference @ rotation.T + translation
        aligned, _, diag = procrustes_align(source, reference, mode="per_sequence")
        np.testing.assert_allclose(aligned, reference, atol=1e-8)
        assert diag.rotation_deg == pytest.approx(12.0, abs=1e-4)
        assert diag.translation == pytest.approx(np.linalg.norm(translation), rel=1e-4)

    def test_scale_flag_recovers_known_scale(self) -> None:
        reference = _skeleton()[np.newaxis, :, :].repeat(2, axis=0)
        source = reference * 0.5
        aligned, _, diag = procrustes_align(source, reference, mode="per_sequence", scale=True)
        np.testing.assert_allclose(aligned, reference, atol=1e-8)
        assert diag.scale == pytest.approx(2.0, rel=1e-6)

    def test_scale_flag_off_leaves_scale_at_one(self) -> None:
        reference = _skeleton()[np.newaxis, :, :].repeat(2, axis=0)
        source = reference * 0.5
        _, _, diag = procrustes_align(source, reference, mode="per_sequence", scale=False)
        assert diag.scale == pytest.approx(1.0)

    def test_rejects_mismatched_shapes(self) -> None:
        a = np.zeros((4, 8, 3))
        b = np.zeros((4, 7, 3))
        with pytest.raises(ValueError, match="same shape"):
            procrustes_align(a, b)

    def test_rejects_wrong_trailing_axis(self) -> None:
        a = np.zeros((4, 8, 2))
        b = np.zeros((4, 8, 2))
        with pytest.raises(ValueError, match="joints, 3"):
            procrustes_align(a, b)

    def test_rejects_unknown_mode(self) -> None:
        a = np.zeros((2, 4, 3))
        with pytest.raises(ValueError, match="unknown mode"):
            procrustes_align(a, a, mode="nope")  # type: ignore[arg-type]

    def test_does_not_mutate_inputs(self) -> None:
        source = _skeleton()[np.newaxis, :, :].repeat(3, axis=0).copy()
        target = (source @ _rotation_matrix_z(np.deg2rad(10.0)).T).copy()
        source_before = source.copy()
        target_before = target.copy()
        procrustes_align(source, target, mode="per_sequence")
        np.testing.assert_array_equal(source, source_before)
        np.testing.assert_array_equal(target, target_before)

    def test_returns_alignment_diagnostics_dataclass(self) -> None:
        a = _skeleton()[np.newaxis, :, :].repeat(2, axis=0)
        _, _, diag = procrustes_align(a, a)
        assert isinstance(diag, AlignmentDiagnostics)


class TestProcrustesAlignPerFrame:
    def test_per_frame_recovers_varying_rotations(self) -> None:
        # Each frame is rotated by a different angle; per_frame alignment
        # should recover each frame independently.
        num_frames = 4
        reference_frame = _skeleton(num_joints=6)
        angles = np.deg2rad([5.0, -10.0, 20.0, 45.0])
        reference = np.stack([reference_frame for _ in range(num_frames)], axis=0)
        source = np.stack([reference_frame @ _rotation_matrix_z(a).T for a in angles], axis=0)
        aligned, _, diag = procrustes_align(source, reference, mode="per_frame")
        np.testing.assert_allclose(aligned, reference, atol=1e-8)
        assert diag.mode == "per_frame"
        # The max rotation across frames should be 45°.
        assert diag.rotation_deg_max == pytest.approx(45.0, abs=1e-4)
        # The mean rotation across frames should be 20°.
        assert diag.rotation_deg == pytest.approx(20.0, abs=1e-4)

    def test_per_frame_with_identical_sequences_yields_zero(self) -> None:
        sequence = _skeleton(num_joints=5)[np.newaxis, :, :].repeat(3, axis=0)
        aligned, _, diag = procrustes_align(sequence, sequence, mode="per_frame")
        np.testing.assert_allclose(aligned, sequence, atol=1e-10)
        # Per-frame SVD on a symmetric covariance is numerically ambiguous
        # in axis selection, so the fitted rotation can be a few micro-
        # degrees off zero; the residual positions are still exact.
        assert diag.rotation_deg == pytest.approx(0.0, abs=1e-3)
        assert diag.rotation_deg_max == pytest.approx(0.0, abs=1e-3)
        assert diag.translation == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# DTW with align= (integration)
# ---------------------------------------------------------------------------


class TestDtwAlignIntegration:
    """Smoke tests: align= routes through procrustes_align correctly.

    Depth tests of the DTW path itself live in test_analyzer_dtw.
    """

    def test_dtw_all_with_alignment_cancels_rigid_offset(self) -> None:
        pytest.importorskip("fastdtw")
        from neuropose.analyzer.dtw import dtw_all

        rotation = _rotation_matrix_z(np.deg2rad(30.0))
        translation = np.array([5.0, -2.0, 1.0])
        reference = _skeleton(num_joints=6)[np.newaxis, :, :].repeat(4, axis=0)
        source = reference @ rotation.T + translation
        baseline = dtw_all(source, reference, align="none")
        aligned_result = dtw_all(source, reference, align="procrustes_per_sequence")
        assert baseline.distance > 0.0
        assert aligned_result.distance == pytest.approx(0.0, abs=1e-6)

    def test_dtw_align_rejects_mismatched_frame_counts(self) -> None:
        pytest.importorskip("fastdtw")
        from neuropose.analyzer.dtw import dtw_all

        a = np.zeros((5, 3, 3))
        b = np.zeros((6, 3, 3))
        with pytest.raises(ValueError, match="matching frame counts"):
            dtw_all(a, b, align="procrustes_per_sequence")
