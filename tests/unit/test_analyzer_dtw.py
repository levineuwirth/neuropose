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
