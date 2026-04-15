"""Tests for :mod:`neuropose.analyzer.segment`.

Three layers of coverage:

- **Layer 1** (:func:`segment_by_peaks`) against synthetic 1D signals
  with known peaks and valleys.
- **Layer 2** (:func:`segment_predictions`) against synthetic
  :class:`VideoPredictions` fixtures exercising every extractor variant.
- **Slicing** (:func:`slice_predictions`) — per-rep round-trip and the
  metadata rewrite.

The extractor factories and the discriminated :class:`ExtractorSpec`
union are covered incidentally through Layer 2 (which is the only
layer that cares about the spec shape) plus a handful of targeted
schema tests for the validators (distinct joint indices etc.).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from neuropose.analyzer.segment import (
    JOINT_INDEX,
    JOINT_NAMES,
    extract_signal,
    joint_angle,
    joint_axis,
    joint_index,
    joint_pair_distance,
    joint_speed,
    segment_by_peaks,
    segment_predictions,
    slice_predictions,
)
from neuropose.io import (
    JointAngleExtractor,
    JointAxisExtractor,
    JointPairDistanceExtractor,
    JointSpeedExtractor,
    Segment,
    Segmentation,
    VideoPredictions,
)

NUM_JOINTS = 43


def _triple_hump_signal(num_frames: int = 300) -> np.ndarray:
    """Three non-negative sine humps separated by clear zero-valleys."""
    t = np.linspace(0.0, 6.0 * math.pi, num_frames)
    return np.maximum(0.0, np.sin(t)) ** 2


def _make_predictions(
    signal: np.ndarray,
    joint: int,
    *,
    axis: int = 1,
    fps: float = 30.0,
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
        }
    )


# ---------------------------------------------------------------------------
# JOINT_NAMES / JOINT_INDEX / joint_index()
# ---------------------------------------------------------------------------


class TestJointNames:
    def test_tuple_length_is_43(self) -> None:
        assert len(JOINT_NAMES) == 43

    def test_index_matches_position(self) -> None:
        for idx, name in enumerate(JOINT_NAMES):
            assert JOINT_INDEX[name] == idx

    def test_joint_index_by_name(self) -> None:
        assert joint_index("lwri") == JOINT_NAMES.index("lwri")
        assert joint_index("rwri") == JOINT_NAMES.index("rwri")

    def test_joint_index_unknown_name(self) -> None:
        with pytest.raises(KeyError, match="unknown joint name"):
            joint_index("elbow")  # deliberately wrong spelling


# ---------------------------------------------------------------------------
# Factory shortcuts
# ---------------------------------------------------------------------------


class TestFactories:
    def test_joint_axis_factory(self) -> None:
        spec = joint_axis(JOINT_INDEX["lwri"], 1, invert=True)
        assert isinstance(spec, JointAxisExtractor)
        assert spec.kind == "joint_axis"
        assert spec.joint == JOINT_INDEX["lwri"]
        assert spec.axis == 1
        assert spec.invert is True

    def test_joint_pair_distance_factory(self) -> None:
        spec = joint_pair_distance(JOINT_INDEX["lwri"], JOINT_INDEX["rwri"])
        assert isinstance(spec, JointPairDistanceExtractor)
        assert spec.joints == (JOINT_INDEX["lwri"], JOINT_INDEX["rwri"])

    def test_joint_pair_distance_rejects_same_joint(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="distinct"):
            joint_pair_distance(5, 5)

    def test_joint_speed_factory(self) -> None:
        spec = joint_speed(JOINT_INDEX["rwri"])
        assert isinstance(spec, JointSpeedExtractor)
        assert spec.joint == JOINT_INDEX["rwri"]

    def test_joint_angle_factory(self) -> None:
        spec = joint_angle(
            JOINT_INDEX["larm"],
            JOINT_INDEX["lelb"],
            JOINT_INDEX["lwri"],
        )
        assert isinstance(spec, JointAngleExtractor)
        assert spec.triplet == (
            JOINT_INDEX["larm"],
            JOINT_INDEX["lelb"],
            JOINT_INDEX["lwri"],
        )


# ---------------------------------------------------------------------------
# extract_signal: one test per extractor variant
# ---------------------------------------------------------------------------


class TestExtractSignal:
    def test_joint_axis_selects_axis(self) -> None:
        seq = np.zeros((4, NUM_JOINTS, 3))
        seq[:, 10, 1] = [1.0, 2.0, 3.0, 4.0]
        signal = extract_signal(seq, joint_axis(10, 1))
        np.testing.assert_array_equal(signal, [1.0, 2.0, 3.0, 4.0])

    def test_joint_axis_invert(self) -> None:
        seq = np.zeros((3, NUM_JOINTS, 3))
        seq[:, 0, 0] = [1.0, 2.0, 3.0]
        signal = extract_signal(seq, joint_axis(0, 0, invert=True))
        np.testing.assert_array_equal(signal, [-1.0, -2.0, -3.0])

    def test_joint_pair_distance(self) -> None:
        seq = np.zeros((3, NUM_JOINTS, 3))
        seq[:, 0, 0] = [0.0, 0.0, 0.0]
        seq[:, 1, 0] = [3.0, 6.0, 9.0]  # distances 3, 6, 9 along x
        signal = extract_signal(seq, joint_pair_distance(0, 1))
        np.testing.assert_allclose(signal, [3.0, 6.0, 9.0])

    def test_joint_speed_pads_first_frame_with_zero(self) -> None:
        seq = np.zeros((4, NUM_JOINTS, 3))
        seq[:, 5, 0] = [0.0, 1.0, 3.0, 6.0]  # speeds: 1, 2, 3
        signal = extract_signal(seq, joint_speed(5))
        np.testing.assert_allclose(signal, [0.0, 1.0, 2.0, 3.0])

    def test_joint_speed_single_frame_rejected(self) -> None:
        seq = np.zeros((1, NUM_JOINTS, 3))
        with pytest.raises(ValueError, match="at least two frames"):
            extract_signal(seq, joint_speed(0))

    def test_joint_angle_straight(self) -> None:
        seq = np.zeros((2, NUM_JOINTS, 3))
        # Straight line: a=(-1,0,0), b=(0,0,0), c=(1,0,0). Angle = pi.
        seq[:, 0, 0] = [-1.0, -1.0]
        seq[:, 1, 0] = [0.0, 0.0]
        seq[:, 2, 0] = [1.0, 1.0]
        signal = extract_signal(seq, joint_angle(0, 1, 2))
        np.testing.assert_allclose(signal, [math.pi, math.pi])

    def test_joint_angle_right(self) -> None:
        seq = np.zeros((1, NUM_JOINTS, 3))
        seq[0, 0] = [1.0, 0.0, 0.0]
        seq[0, 1] = [0.0, 0.0, 0.0]
        seq[0, 2] = [0.0, 1.0, 0.0]
        signal = extract_signal(seq, joint_angle(0, 1, 2))
        np.testing.assert_allclose(signal, [math.pi / 2])

    def test_out_of_range_joint_index(self) -> None:
        seq = np.zeros((3, NUM_JOINTS, 3))
        with pytest.raises(ValueError, match="out of range"):
            extract_signal(seq, joint_axis(999, 0))

    def test_bad_sequence_shape(self) -> None:
        with pytest.raises(ValueError, match="frames, joints, 3"):
            extract_signal(np.zeros((5, 10)), joint_axis(0, 0))


# ---------------------------------------------------------------------------
# Layer 1: segment_by_peaks
# ---------------------------------------------------------------------------


class TestSegmentByPeaks:
    def test_three_humps_three_segments(self) -> None:
        signal = _triple_hump_signal()
        segs = segment_by_peaks(signal, min_prominence=0.1)
        assert len(segs) == 3
        # Segments should not overlap (in this synthetic case). Adjacent
        # segments are allowed to share the valley frame on their boundary,
        # so we use a ``>=`` comparison with one frame of slack.
        for prev, curr in itertools.pairwise(segs):
            assert curr.start >= prev.end - 1

    def test_first_segment_starts_at_zero_without_leading_valley(self) -> None:
        signal = _triple_hump_signal()
        segs = segment_by_peaks(signal, min_prominence=0.1)
        assert segs[0].start == 0

    def test_last_segment_ends_at_signal_length(self) -> None:
        signal = _triple_hump_signal()
        segs = segment_by_peaks(signal, min_prominence=0.1)
        assert segs[-1].end == len(signal)

    def test_peaks_lie_inside_segment(self) -> None:
        signal = _triple_hump_signal()
        segs = segment_by_peaks(signal, min_prominence=0.1)
        for seg in segs:
            assert seg.start <= seg.peak < seg.end

    def test_no_peaks_returns_empty(self) -> None:
        flat = np.zeros(50)
        segs = segment_by_peaks(flat)
        assert segs == []

    def test_min_distance_suppresses_close_peaks(self) -> None:
        # A signal with two very close peaks should give only one segment
        # when min_distance is large.
        signal = np.zeros(100)
        signal[20] = 1.0
        signal[25] = 1.0
        signal[70] = 1.0
        segs = segment_by_peaks(signal, min_distance=30)
        # Exact count depends on scipy's tie-breaking; main assertion is
        # "fewer than if no distance constraint".
        segs_unconstrained = segment_by_peaks(signal)
        assert len(segs) < len(segs_unconstrained)

    def test_pad_extends_segment(self) -> None:
        signal = _triple_hump_signal()
        base = segment_by_peaks(signal, min_prominence=0.1)
        padded = segment_by_peaks(signal, min_prominence=0.1, pad=5)
        assert len(base) == len(padded)
        for b, p in zip(base, padded, strict=True):
            assert p.start <= b.start
            assert p.end >= b.end

    def test_pad_is_clamped_to_bounds(self) -> None:
        signal = _triple_hump_signal()
        padded = segment_by_peaks(signal, min_prominence=0.1, pad=10_000)
        for seg in padded:
            assert seg.start >= 0
            assert seg.end <= len(signal)

    def test_negative_pad_rejected(self) -> None:
        with pytest.raises(ValueError, match="pad"):
            segment_by_peaks(np.zeros(10), pad=-1)

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            segment_by_peaks(np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# Layer 2: segment_predictions
# ---------------------------------------------------------------------------


class TestSegmentPredictions:
    def test_returns_segmentation_with_config(self) -> None:
        signal = _triple_hump_signal() * 1000.0  # mm scale
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"])
        result = segment_predictions(
            preds,
            joint_axis(JOINT_INDEX["lwri"], 1),
            min_prominence=50.0,
        )
        assert isinstance(result, Segmentation)
        assert len(result.segments) == 3
        assert result.config.extractor.kind == "joint_axis"
        assert result.config.min_prominence == 50.0
        assert result.config.method == "valley_to_valley_v1"

    def test_min_distance_seconds_converts_via_fps(self) -> None:
        # 300 frames at 30 fps = 10 seconds; humps are ~3.3 s apart.
        signal = _triple_hump_signal() * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"], fps=30.0)
        # A 5-second minimum distance should collapse the three humps
        # into at most two segments.
        result = segment_predictions(
            preds,
            joint_axis(JOINT_INDEX["lwri"], 1),
            min_prominence=50.0,
            min_distance_seconds=5.0,
        )
        assert len(result.segments) <= 2

    def test_pad_seconds_extends_segments(self) -> None:
        signal = _triple_hump_signal() * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"], fps=30.0)
        plain = segment_predictions(preds, joint_axis(JOINT_INDEX["lwri"], 1), min_prominence=50.0)
        padded = segment_predictions(
            preds,
            joint_axis(JOINT_INDEX["lwri"], 1),
            min_prominence=50.0,
            pad_seconds=0.2,  # ~6 frames at 30 fps
        )
        # At least one segment should have moved outward.
        assert any(
            p.start < b.start or p.end > b.end
            for p, b in zip(padded.segments, plain.segments, strict=True)
        )

    def test_requires_fps_when_time_params_used(self) -> None:
        signal = _triple_hump_signal() * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"], fps=0.0)
        with pytest.raises(ValueError, match="fps"):
            segment_predictions(
                preds,
                joint_axis(JOINT_INDEX["lwri"], 1),
                min_prominence=50.0,
                min_distance_seconds=1.0,
            )

    def test_no_fps_is_fine_without_time_params(self) -> None:
        signal = _triple_hump_signal() * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"], fps=0.0)
        # Without any time-based parameters we never need to multiply by
        # fps, so fps=0 is tolerated.
        result = segment_predictions(
            preds,
            joint_axis(JOINT_INDEX["lwri"], 1),
            min_prominence=50.0,
        )
        assert len(result.segments) == 3

    def test_config_roundtrips_through_json(self) -> None:
        signal = _triple_hump_signal() * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"])
        result = segment_predictions(
            preds,
            joint_pair_distance(JOINT_INDEX["lwri"], JOINT_INDEX["rwri"]),
            min_prominence=10.0,
        )
        serialized = result.model_dump(mode="json")
        rehydrated = Segmentation.model_validate(serialized)
        assert rehydrated == result


# ---------------------------------------------------------------------------
# slice_predictions
# ---------------------------------------------------------------------------


class TestSlicePredictions:
    def test_one_output_per_segment(self) -> None:
        signal = _triple_hump_signal() * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"])
        result = segment_predictions(preds, joint_axis(JOINT_INDEX["lwri"], 1), min_prominence=50.0)
        slices = slice_predictions(preds, result.segments)
        assert len(slices) == len(result.segments)

    def test_metadata_frame_count_matches_segment_length(self) -> None:
        signal = _triple_hump_signal() * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"])
        segments = [Segment(start=10, end=30, peak=20), Segment(start=50, end=90, peak=75)]
        slices = slice_predictions(preds, segments)
        assert slices[0].metadata.frame_count == 20
        assert slices[1].metadata.frame_count == 40

    def test_frames_are_rekeyed_from_zero(self) -> None:
        signal = _triple_hump_signal() * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"])
        segments = [Segment(start=100, end=110, peak=105)]
        sliced = slice_predictions(preds, segments)[0]
        assert sliced.frame_names()[0] == "frame_000000"
        assert sliced.frame_names()[-1] == "frame_000009"

    def test_sliced_segmentations_field_is_empty(self) -> None:
        # Parent has a segmentation attached; sliced copies intentionally
        # drop it because segment indices are only meaningful in the
        # parent's timeline.
        signal = _triple_hump_signal() * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"])
        result = segment_predictions(preds, joint_axis(JOINT_INDEX["lwri"], 1), min_prominence=50.0)
        parent = preds.model_copy(update={"segmentations": {"cup_lift": result}})
        slices = slice_predictions(parent, result.segments)
        assert all(s.segmentations == {} for s in slices)

    def test_out_of_bounds_segment_rejected(self) -> None:
        signal = _triple_hump_signal(num_frames=50) * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"])
        bad = [Segment(start=0, end=100, peak=25)]  # end > 50
        with pytest.raises(ValueError, match="exceeds frame count"):
            slice_predictions(preds, bad)

    def test_slice_preserves_pose_values(self) -> None:
        signal = _triple_hump_signal() * 1000.0
        preds = _make_predictions(signal, joint=JOINT_INDEX["lwri"])
        segments = [Segment(start=50, end=55, peak=52)]
        sliced = slice_predictions(preds, segments)[0]
        # frame_000000 of the slice must equal frame_000050 of the source
        assert sliced["frame_000000"].poses3d == preds["frame_000050"].poses3d
