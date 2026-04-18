"""Tests for :mod:`neuropose.analyzer.dtw`."""

from __future__ import annotations

import numpy as np
import pytest

from neuropose.analyzer.dtw import (
    DTWResult,
    dtw_all,
    dtw_per_joint,
    dtw_relation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_sequence() -> np.ndarray:
    """A 5-frame, 3-joint sequence of linearly-moving joints."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((5, 3, 3))


# ---------------------------------------------------------------------------
# dtw_all
# ---------------------------------------------------------------------------


class TestDtwAll:
    def test_identical_sequences_distance_zero(self, simple_sequence: np.ndarray) -> None:
        result = dtw_all(simple_sequence, simple_sequence)
        assert isinstance(result, DTWResult)
        assert result.distance == pytest.approx(0.0, abs=1e-9)
        # Identical sequences produce a diagonal warping path.
        assert all(i == j for i, j in result.path)

    def test_shifted_sequences_distance_zero(self, simple_sequence: np.ndarray) -> None:
        """DTW should absorb a pure time shift without penalty."""
        # Duplicate the first frame to create a one-frame shift.
        shifted = np.concatenate([simple_sequence[:1], simple_sequence], axis=0)
        result = dtw_all(simple_sequence, shifted)
        assert result.distance == pytest.approx(0.0, abs=1e-9)

    def test_different_sequences_positive_distance(self) -> None:
        a = np.zeros((5, 3, 3))
        b = np.ones((5, 3, 3))
        result = dtw_all(a, b)
        assert result.distance > 0.0

    def test_mismatched_joint_count_rejected(self) -> None:
        a = np.zeros((5, 3, 3))
        b = np.zeros((5, 4, 3))
        with pytest.raises(ValueError, match="joint count"):
            dtw_all(a, b)

    def test_non_3d_input_rejected(self) -> None:
        a = np.zeros((5, 3))  # missing trailing axis
        b = np.zeros((5, 3))
        with pytest.raises(ValueError, match="expected 3D"):
            dtw_all(a, b)


# ---------------------------------------------------------------------------
# dtw_per_joint
# ---------------------------------------------------------------------------


class TestDtwPerJoint:
    def test_returns_one_result_per_joint(self, simple_sequence: np.ndarray) -> None:
        results = dtw_per_joint(simple_sequence, simple_sequence)
        assert len(results) == simple_sequence.shape[1]
        for result in results:
            assert isinstance(result, DTWResult)
            assert result.distance == pytest.approx(0.0, abs=1e-9)

    def test_independent_joint_distances(self) -> None:
        # Construct two sequences where joint 0 matches exactly but
        # joint 1 is offset by a constant. Per-joint DTW should give
        # distance 0 for joint 0 and distance > 0 for joint 1.
        a = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            ]
        )
        b = a.copy()
        b[:, 1, :] += 10.0
        results = dtw_per_joint(a, b)
        assert results[0].distance == pytest.approx(0.0, abs=1e-9)
        assert results[1].distance > 0.0

    def test_mismatched_joint_count_rejected(self) -> None:
        a = np.zeros((5, 3, 3))
        b = np.zeros((5, 2, 3))
        with pytest.raises(ValueError, match="joint count"):
            dtw_per_joint(a, b)


# ---------------------------------------------------------------------------
# dtw_relation
# ---------------------------------------------------------------------------


class TestDtwRelation:
    def test_identical_sequences_distance_zero(self, simple_sequence: np.ndarray) -> None:
        result = dtw_relation(simple_sequence, simple_sequence, joint_i=0, joint_j=1)
        assert result.distance == pytest.approx(0.0, abs=1e-9)

    def test_same_relative_position_is_zero_even_under_translation(self) -> None:
        """Translating the whole body does not change the
        joint-to-joint displacement, so dtw_relation should be 0."""
        a = np.zeros((4, 3, 3))
        a[:, 0, :] = [0.0, 0.0, 0.0]
        a[:, 1, :] = [1.0, 0.0, 0.0]
        a[:, 2, :] = [0.0, 1.0, 0.0]
        b = a + 50.0  # translate the whole body
        result = dtw_relation(a, b, joint_i=0, joint_j=1)
        assert result.distance == pytest.approx(0.0, abs=1e-9)

    def test_joint_index_out_of_range_rejected(self) -> None:
        a = np.zeros((3, 2, 3))
        b = np.zeros((3, 2, 3))
        with pytest.raises(ValueError, match="joint indices"):
            dtw_relation(a, b, joint_i=0, joint_j=5)

    def test_mismatched_joint_count_rejected(self) -> None:
        a = np.zeros((3, 3, 3))
        b = np.zeros((3, 2, 3))
        with pytest.raises(ValueError, match="joint count"):
            dtw_relation(a, b, joint_i=0, joint_j=1)


# ---------------------------------------------------------------------------
# representation="angles"
# ---------------------------------------------------------------------------


def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _three_joint_arm(num_frames: int = 6) -> np.ndarray:
    """A three-joint arm opening from a right angle to straight.

    Joints laid out as [shoulder, elbow, wrist], forming an angle at
    the elbow that linearly opens from pi/2 to pi across ``num_frames``.
    """
    sequence = np.zeros((num_frames, 3, 3))
    angles = np.linspace(np.pi / 2, np.pi, num_frames)
    for i, theta in enumerate(angles):
        sequence[i, 0] = [-1.0, 0.0, 0.0]  # shoulder
        sequence[i, 1] = [0.0, 0.0, 0.0]  # elbow
        sequence[i, 2] = [np.cos(theta - np.pi), np.sin(theta - np.pi), 0.0]  # wrist
    return sequence


class TestDtwAllAngles:
    def test_angles_identical_sequences_distance_zero(self) -> None:
        seq = _three_joint_arm()
        result = dtw_all(
            seq,
            seq,
            representation="angles",
            angle_triplets=[(0, 1, 2)],
        )
        assert result.distance == pytest.approx(0.0, abs=1e-9)

    def test_angles_invariant_to_global_rotation(self) -> None:
        """Angle-space DTW must not change under a global rotation."""
        seq = _three_joint_arm()
        rotated = seq @ _rotation_matrix_z(np.deg2rad(40.0)).T
        baseline = dtw_all(seq, seq, representation="angles", angle_triplets=[(0, 1, 2)])
        under_rotation = dtw_all(
            seq,
            rotated,
            representation="angles",
            angle_triplets=[(0, 1, 2)],
        )
        assert baseline.distance == pytest.approx(under_rotation.distance, abs=1e-6)

    def test_angles_translation_invariant(self) -> None:
        seq = _three_joint_arm()
        translated = seq + np.array([10.0, -5.0, 2.0])
        result = dtw_all(
            seq,
            translated,
            representation="angles",
            angle_triplets=[(0, 1, 2)],
        )
        assert result.distance == pytest.approx(0.0, abs=1e-9)

    def test_angles_detects_different_motion(self) -> None:
        # A sequence whose angle is constant vs. one that opens.
        constant = np.zeros((6, 3, 3))
        constant[:, 0] = [-1.0, 0.0, 0.0]
        constant[:, 1] = [0.0, 0.0, 0.0]
        constant[:, 2] = [0.0, 1.0, 0.0]  # right angle throughout
        opening = _three_joint_arm()
        result = dtw_all(
            constant,
            opening,
            representation="angles",
            angle_triplets=[(0, 1, 2)],
        )
        assert result.distance > 0.0

    def test_angles_without_triplets_rejected(self) -> None:
        seq = _three_joint_arm()
        with pytest.raises(ValueError, match="angle_triplets"):
            dtw_all(seq, seq, representation="angles")


class TestDtwPerJointAngles:
    def test_returns_one_result_per_triplet(self) -> None:
        seq = _three_joint_arm()
        triplets = [(0, 1, 2), (0, 1, 2)]  # duplicate triplet on purpose
        results = dtw_per_joint(
            seq,
            seq,
            representation="angles",
            angle_triplets=triplets,
        )
        assert len(results) == 2
        for result in results:
            assert result.distance == pytest.approx(0.0, abs=1e-9)

    def test_per_triplet_distinct_paths(self) -> None:
        # Two triplets covering different angles; with different motion
        # per triplet, the per-unit results should differ.
        seq_a = np.zeros((5, 4, 3))
        seq_b = np.zeros((5, 4, 3))
        # joint 0: pivot, joint 1/2/3: arm endpoints
        for i in range(5):
            seq_a[i, 0] = [0.0, 0.0, 0.0]
            seq_a[i, 1] = [1.0, 0.0, 0.0]
            seq_a[i, 2] = [0.0, 1.0, 0.0]
            seq_a[i, 3] = [0.0, 0.0, 1.0]
            seq_b[i, 0] = [0.0, 0.0, 0.0]
            seq_b[i, 1] = [1.0, 0.0, 0.0]
            seq_b[i, 2] = [np.cos(i * 0.3), np.sin(i * 0.3), 0.0]  # rotating
            seq_b[i, 3] = [0.0, 0.0, 1.0]
        results = dtw_per_joint(
            seq_a,
            seq_b,
            representation="angles",
            angle_triplets=[(1, 0, 2), (1, 0, 3)],
        )
        assert len(results) == 2
        # First triplet tracks the rotation, second is stationary.
        assert results[0].distance > 0.0
        assert results[1].distance == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# nan_policy
# ---------------------------------------------------------------------------


def _collinear_sequence(num_frames: int = 4) -> np.ndarray:
    """Three collinear joints — the angle at the middle joint is degenerate."""
    seq = np.zeros((num_frames, 3, 3))
    seq[:, 0] = [-1.0, 0.0, 0.0]
    # Middle joint at (0,0,0); but because the outer joints are collinear
    # through the origin, we need one joint overlapping with the middle
    # to force a zero-length vector. Place joint 2 AT joint 1 to trigger
    # the degenerate case in extract_joint_angles.
    seq[:, 1] = [0.0, 0.0, 0.0]
    seq[:, 2] = [0.0, 0.0, 0.0]
    return seq


class TestNanPolicy:
    def test_propagate_surfaces_error(self) -> None:
        # Degenerate triplet produces NaN angles for every frame.
        # With nan_policy="propagate" the NaN reaches fastdtw, which
        # validates via numpy.asarray_chkfinite and raises ValueError —
        # the intended behaviour ("make the problem visible").
        seq = _collinear_sequence(num_frames=4)
        other = _three_joint_arm(num_frames=4)
        with pytest.raises(ValueError, match="infs or NaNs"):
            dtw_all(
                seq,
                other,
                representation="angles",
                angle_triplets=[(0, 1, 2)],
                nan_policy="propagate",
            )

    def test_interpolate_fills_isolated_nan(self) -> None:
        # One bad frame in a 5-frame sequence — the other four are
        # finite anchors to interpolate between.
        good = _three_joint_arm(num_frames=5)
        # Inject a degenerate middle frame.
        good[2, 2] = good[2, 1]  # force zero-length vector → NaN angle
        # Reference is the same arm without injection.
        reference = _three_joint_arm(num_frames=5)
        result = dtw_all(
            good,
            reference,
            representation="angles",
            angle_triplets=[(0, 1, 2)],
            nan_policy="interpolate",
        )
        assert not np.isnan(result.distance)

    def test_interpolate_all_nan_column_rejected(self) -> None:
        seq = _collinear_sequence(num_frames=5)
        other = _three_joint_arm(num_frames=5)
        with pytest.raises(ValueError, match="all values are NaN"):
            dtw_all(
                seq,
                other,
                representation="angles",
                angle_triplets=[(0, 1, 2)],
                nan_policy="interpolate",
            )

    def test_drop_removes_nan_frames(self) -> None:
        good = _three_joint_arm(num_frames=6)
        good[2, 2] = good[2, 1]  # inject NaN at frame 2
        good[4, 2] = good[4, 1]  # inject NaN at frame 4
        reference = _three_joint_arm(num_frames=6)
        result = dtw_all(
            good,
            reference,
            representation="angles",
            angle_triplets=[(0, 1, 2)],
            nan_policy="drop",
        )
        # The 4 remaining finite frames should align cleanly with
        # their counterparts in the reference.
        assert not np.isnan(result.distance)

    def test_drop_empties_sequence_rejected(self) -> None:
        seq = _collinear_sequence(num_frames=5)
        other = _three_joint_arm(num_frames=5)
        with pytest.raises(ValueError, match="every frame"):
            dtw_all(
                seq,
                other,
                representation="angles",
                angle_triplets=[(0, 1, 2)],
                nan_policy="drop",
            )


# ---------------------------------------------------------------------------
# align + representation composition
# ---------------------------------------------------------------------------


class TestAlignWithAngles:
    def test_procrustes_before_angles_is_no_op_on_invariant_representation(self) -> None:
        """Procrustes on angle-space DTW should be redundant but safe."""
        seq = _three_joint_arm()
        rotated = seq @ _rotation_matrix_z(np.deg2rad(20.0)).T
        with_align = dtw_all(
            seq,
            rotated,
            align="procrustes_per_sequence",
            representation="angles",
            angle_triplets=[(0, 1, 2)],
        )
        without_align = dtw_all(
            seq,
            rotated,
            representation="angles",
            angle_triplets=[(0, 1, 2)],
        )
        assert with_align.distance == pytest.approx(without_align.distance, abs=1e-6)
