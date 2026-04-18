"""Repetition segmentation for pose sequences.

Given a :class:`~neuropose.io.VideoPredictions` of a trial in which the
subject performs the same movement several times (e.g. lifting a cup
repeatedly), this module detects the individual repetitions and returns
them as a :class:`~neuropose.io.Segmentation` that can be persisted back
into ``results.json``.

The detection is a two-step pipeline:

1. **Extract a 1D segmentation signal** from the per-frame pose array,
   using one of the :class:`~neuropose.io.ExtractorSpec` variants
   (:class:`~neuropose.io.JointAxisExtractor`,
   :class:`~neuropose.io.JointPairDistanceExtractor`,
   :class:`~neuropose.io.JointSpeedExtractor`, or
   :class:`~neuropose.io.JointAngleExtractor`).
2. **Walk peaks to their neighbouring valleys** to form one ``[start,
   peak, end)`` window per repetition. The "valley-to-valley" method
   assumes the subject comes to rest between repetitions — the recording
   protocol for NeuroPose's clinical use cases makes this assumption
   reasonable; see :mod:`neuropose.io.SegmentationConfig.method` for the
   version stamp that travels with persisted segmentations.

Three layers of API are provided, in increasing order of convenience:

- :func:`segment_by_peaks` — pure 1D signal segmentation, no pose or
  metadata awareness. All parameters are in sample counts.
- :func:`segment_predictions` — the top-level entry point. Takes a
  :class:`~neuropose.io.VideoPredictions` plus an
  :class:`~neuropose.io.ExtractorSpec`, converts time-based parameters
  to frame counts using ``metadata.fps``, and returns a full
  :class:`~neuropose.io.Segmentation` ready to attach to the predictions.
- :func:`segment_gait_cycles` and :func:`segment_gait_cycles_bilateral`
  — clinical convenience wrappers over :func:`segment_predictions`
  that pre-fill a :func:`joint_axis` extractor with gait-appropriate
  defaults (heel joint, Y axis, 0.4 s minimum cycle). The bilateral
  variant returns both sides under ``"left_heel_strikes"`` and
  ``"right_heel_strikes"`` keys.
- :func:`slice_predictions` — split a :class:`~neuropose.io.VideoPredictions`
  into one per-repetition :class:`~neuropose.io.VideoPredictions`,
  useful when downstream code wants per-rep objects rather than windows
  over the original.

The four :class:`~neuropose.io.ExtractorSpec` variants have convenience
factories (:func:`joint_axis`, :func:`joint_pair_distance`,
:func:`joint_speed`, :func:`joint_angle`) re-exported at the subpackage
root so that Python callers never have to import the pydantic classes
directly::

    from neuropose.analyzer import (
        segment_predictions,
        joint_pair_distance,
        JOINT_INDEX,
    )

    seg = segment_predictions(
        predictions,
        joint_pair_distance(JOINT_INDEX["lwri"], JOINT_INDEX["rwri"]),
        min_distance_seconds=0.5,
        min_prominence=50.0,
    )

Dependency note
---------------
This module requires :mod:`scipy` for peak detection, which is part of
the ``analysis`` optional extra. The import is lazy so that
``import neuropose.analyzer.segment`` succeeds even when scipy is not
installed; a clear :class:`ImportError` surfaces at the first call to
:func:`segment_by_peaks` or :func:`segment_predictions`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from neuropose.analyzer.features import predictions_to_numpy
from neuropose.io import (
    ExtractorSpec,
    JointAngleExtractor,
    JointAxisExtractor,
    JointPairDistanceExtractor,
    JointSpeedExtractor,
    Segment,
    Segmentation,
    SegmentationConfig,
    VideoMetadata,
    VideoPredictions,
)

AxisLetter = Literal["x", "y", "z"]
"""Axis selector used by gait-cycle segmentation helpers."""

_AXIS_INDICES: dict[AxisLetter, int] = {"x": 0, "y": 1, "z": 2}

# ---------------------------------------------------------------------------
# berkeley_mhad_43 joint names
# ---------------------------------------------------------------------------
#
# The MeTRAbs SavedModel we pin in ``neuropose._model`` exposes a 43-joint
# skeleton named ``berkeley_mhad_43``. The names below are captured
# verbatim from that model and committed as a constant so that
# post-processing code (CLI flags, analyzer helpers) can translate
# human-readable joint names into indices without having to load a
# multi-gigabyte TensorFlow model just to resolve ``"rwri"``.
#
# An integration test (:mod:`tests.integration.test_joint_names_match_model`)
# asserts that this tuple still matches what the loaded model reports, so
# any upstream change in the MeTRAbs skeleton will fail CI the next time
# the slow tests run. When that happens, the expected fix is:
#
#   1. Update this tuple in the same commit that bumps the model pin in
#      ``neuropose._model``, and
#   2. Cross-check any CLI or docs that embed hardcoded joint names.

JOINT_NAMES: tuple[str, ...] = (
    "head",
    "lhead",
    "rhead",
    "rback",
    "backl",
    "backt",
    "lback",
    "lside",
    "bell",
    "chest",
    "rside",
    "lsho1",
    "lsho2",
    "larm",
    "lelb",
    "lwri",
    "lhan1",
    "lhan2",
    "lhan3",
    "rsho1",
    "rsho2",
    "rarm",
    "relb",
    "rwri",
    "rhan1",
    "rhan2",
    "rhan3",
    "lhipb",
    "lhipf",
    "lhipl",
    "lleg",
    "lkne",
    "lank",
    "lhee",
    "lfoo",
    "rhipb",
    "rhipf",
    "rhipl",
    "rleg",
    "rkne",
    "rank",
    "rhee",
    "rfoo",
)

JOINT_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(JOINT_NAMES)}


def joint_index(name: str) -> int:
    """Return the integer index of ``name`` in the berkeley_mhad_43 skeleton.

    Parameters
    ----------
    name
        Joint name as it appears in the MeTRAbs SavedModel
        ``per_skeleton_joint_names["berkeley_mhad_43"]`` tensor — for
        example ``"rwri"`` for the right wrist or ``"lkne"`` for the
        left knee.

    Returns
    -------
    int
        The 0-based index into a ``(frames, 43, 3)`` pose sequence.

    Raises
    ------
    KeyError
        If ``name`` is not one of the 43 known joint names. The error
        message lists all valid names to make recovery obvious.
    """
    try:
        return JOINT_INDEX[name]
    except KeyError:
        raise KeyError(
            f"unknown joint name: {name!r}. Known names: {sorted(JOINT_INDEX)}"
        ) from None


# ---------------------------------------------------------------------------
# Extractor factories
# ---------------------------------------------------------------------------
#
# Thin constructors for the four ExtractorSpec variants. Callers *could*
# instantiate the pydantic models directly, but the factories read a bit
# better at call sites and match the ergonomic pattern set by the rest
# of the analyzer subpackage (e.g. ``joint_pair_distance(j1, j2)`` reads
# like a function call rather than a schema construction).


def joint_axis(joint: int, axis: int, *, invert: bool = False) -> JointAxisExtractor:
    """Construct a :class:`~neuropose.io.JointAxisExtractor`.

    Parameters
    ----------
    joint
        Joint index into the ``(frames, J, 3)`` pose sequence.
    axis
        Spatial axis: ``0`` for x, ``1`` for y, ``2`` for z.
    invert
        If ``True``, negate the signal so valleys become peaks. Useful
        when the movement of interest is a *decrease* in the selected
        coordinate (e.g. a joint that dips during the repetition).
    """
    return JointAxisExtractor(joint=joint, axis=axis, invert=invert)


def joint_pair_distance(j1: int, j2: int) -> JointPairDistanceExtractor:
    """Construct a :class:`~neuropose.io.JointPairDistanceExtractor`.

    The two joints must be distinct; pydantic validation enforces this.
    """
    return JointPairDistanceExtractor(joints=(j1, j2))


def joint_speed(joint: int) -> JointSpeedExtractor:
    """Construct a :class:`~neuropose.io.JointSpeedExtractor`."""
    return JointSpeedExtractor(joint=joint)


def joint_angle(a: int, b: int, c: int) -> JointAngleExtractor:
    """Construct a :class:`~neuropose.io.JointAngleExtractor`.

    The angle is computed at joint ``b`` between the vectors ``(a - b)``
    and ``(c - b)``.
    """
    return JointAngleExtractor(triplet=(a, b, c))


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def extract_signal(sequence: np.ndarray, spec: ExtractorSpec) -> np.ndarray:
    """Reduce a pose sequence to a 1D segmentation signal per the ``spec``.

    Dispatches on the discriminator ``kind`` of ``spec``. All variants
    produce a 1D array whose length equals ``sequence.shape[0]``, so the
    segmentation engine can treat the output uniformly regardless of
    which extractor produced it.

    Parameters
    ----------
    sequence
        Array of shape ``(frames, joints, 3)`` — the output of
        :func:`neuropose.analyzer.features.predictions_to_numpy`.
    spec
        One of the :class:`~neuropose.io.ExtractorSpec` variants.

    Returns
    -------
    numpy.ndarray
        A 1D ``float`` array of length ``frames``.

    Raises
    ------
    ValueError
        If ``sequence`` is not a ``(frames, joints, 3)`` array, if any
        joint index in ``spec`` is out of range, or if
        ``JointSpeedExtractor`` is applied to a single-frame sequence.
    """
    if sequence.ndim != 3 or sequence.shape[-1] != 3:
        raise ValueError(f"expected (frames, joints, 3); got shape {sequence.shape}")
    num_frames, num_joints, _ = sequence.shape

    def _check_joint(idx: int) -> None:
        if not (0 <= idx < num_joints):
            raise ValueError(f"joint index {idx} out of range [0, {num_joints})")

    if isinstance(spec, JointAxisExtractor):
        _check_joint(spec.joint)
        signal = sequence[:, spec.joint, spec.axis].astype(float, copy=True)
        if spec.invert:
            signal = -signal
        return signal

    if isinstance(spec, JointPairDistanceExtractor):
        j1, j2 = spec.joints
        _check_joint(j1)
        _check_joint(j2)
        delta = sequence[:, j1, :] - sequence[:, j2, :]
        return np.linalg.norm(delta, axis=1).astype(float, copy=False)

    if isinstance(spec, JointSpeedExtractor):
        _check_joint(spec.joint)
        if num_frames < 2:
            raise ValueError("joint_speed requires at least two frames")
        diffs = np.diff(sequence[:, spec.joint, :], axis=0)
        mags = np.linalg.norm(diffs, axis=1)
        # Pad the first frame with 0 so the signal length matches the
        # input frame count; segmentation indices then line up with the
        # original frame indices without an off-by-one.
        return np.concatenate([[0.0], mags]).astype(float, copy=False)

    if isinstance(spec, JointAngleExtractor):
        # Deferred import to avoid a circular dependency — features.py
        # and segment.py both live under analyzer/, and extract_joint_angles
        # is defined in features.py.
        from neuropose.analyzer.features import extract_joint_angles

        a, b, c = spec.triplet
        _check_joint(a)
        _check_joint(b)
        _check_joint(c)
        return extract_joint_angles(sequence, [(a, b, c)])[:, 0].astype(float, copy=False)

    raise TypeError(f"unknown extractor kind: {type(spec).__name__}")


# ---------------------------------------------------------------------------
# Layer 1: pure 1D segmentation
# ---------------------------------------------------------------------------


def segment_by_peaks(
    signal: np.ndarray,
    *,
    min_distance: int | None = None,
    min_prominence: float | None = None,
    min_height: float | None = None,
    pad: int = 0,
) -> list[Segment]:
    """Segment a 1D signal into one window per detected peak.

    Implements the ``valley_to_valley_v1`` method: each peak found by
    :func:`scipy.signal.find_peaks` is walked outward to the nearest
    valley on each side (a local minimum of the signal), and the
    resulting ``[start, end)`` window is reported as one
    :class:`~neuropose.io.Segment`. If the signal has no valley before
    the first peak the segment starts at frame 0; likewise a trailing
    peak without a following valley extends to the end of the signal.

    Parameters
    ----------
    signal
        1D numpy array of segmentation feature values (e.g. wrist
        height across frames).
    min_distance, min_prominence, min_height
        Forwarded as ``distance``, ``prominence`` and ``height`` to
        :func:`scipy.signal.find_peaks`. Use these to reject noise; the
        defaults are permissive.
    pad
        Number of samples to extend each resulting segment on both
        sides. Segments are clamped to ``[0, len(signal)]``; adjacent
        padded segments may therefore overlap.

    Returns
    -------
    list[Segment]
        One :class:`~neuropose.io.Segment` per detected repetition, in
        ascending order of peak index. Empty if no peaks were found.

    Raises
    ------
    ValueError
        If ``signal`` is not 1D or if ``pad`` is negative.
    ImportError
        If :mod:`scipy` is not installed. The error message points at
        the ``analysis`` optional extra.
    """
    if signal.ndim != 1:
        raise ValueError(f"expected 1D array; got shape {signal.shape}")
    if pad < 0:
        raise ValueError(f"pad must be non-negative; got {pad}")

    try:
        from scipy.signal import find_peaks as _sp_find_peaks
    except ImportError as exc:
        raise ImportError(
            "neuropose.analyzer.segment.segment_by_peaks requires scipy. "
            "Install it with: pip install neuropose[analysis]"
        ) from exc

    peak_kwargs: dict[str, float | int] = {}
    # scipy requires ``distance >= 1``; treat ``None`` or 0 as "no constraint".
    if min_distance is not None and min_distance >= 1:
        peak_kwargs["distance"] = min_distance
    if min_prominence is not None:
        peak_kwargs["prominence"] = min_prominence
    if min_height is not None:
        peak_kwargs["height"] = min_height

    peaks, _ = _sp_find_peaks(signal, **peak_kwargs)
    if len(peaks) == 0:
        return []

    # Valleys = peaks of the negated signal. No extra filters — we want
    # every local minimum as a candidate boundary, and we'll pick the
    # nearest one on each side of each qualifying peak.
    valleys, _ = _sp_find_peaks(-signal)

    n = int(signal.shape[0])
    segments: list[Segment] = []
    for peak in peaks:
        peak_idx = int(peak)
        left = valleys[valleys < peak_idx]
        right = valleys[valleys > peak_idx]
        start = int(left.max()) if left.size > 0 else 0
        # ``end`` is exclusive; include the trailing valley frame so
        # the segment captures the return-to-rest the clinician cares
        # about.
        end = int(right.min()) + 1 if right.size > 0 else n

        start = max(0, start - pad)
        end = min(n, end + pad)
        segments.append(Segment(start=start, end=end, peak=peak_idx))

    return segments


# ---------------------------------------------------------------------------
# Layer 2: pose-aware convenience
# ---------------------------------------------------------------------------


def segment_predictions(
    predictions: VideoPredictions,
    extractor: ExtractorSpec,
    *,
    person_index: int = 0,
    min_distance_seconds: float | None = None,
    min_prominence: float | None = None,
    min_height: float | None = None,
    pad_seconds: float = 0.0,
) -> Segmentation:
    """Segment a :class:`~neuropose.io.VideoPredictions` into repetitions.

    This is the top-level entry point for post-hoc segmentation. It:

    1. Reduces ``predictions`` to a ``(frames, joints, 3)`` numpy array
       via :func:`~neuropose.analyzer.features.predictions_to_numpy`.
    2. Extracts a 1D segmentation signal using ``extractor``.
    3. Converts the time-based parameters (``min_distance_seconds``,
       ``pad_seconds``) to frame counts using
       ``predictions.metadata.fps``.
    4. Delegates to :func:`segment_by_peaks`.
    5. Wraps the result in a :class:`~neuropose.io.Segmentation` whose
       :class:`~neuropose.io.SegmentationConfig` carries the original
       (time-based) parameters and the extractor spec, so the output
       is self-describing when persisted.

    Parameters
    ----------
    predictions
        Per-video predictions to segment. The ``metadata.fps`` field
        is used to convert seconds to frames; callers with a video of
        unknown frame rate should fall back to :func:`segment_by_peaks`
        directly.
    extractor
        Serializable extractor spec — typically produced by one of the
        convenience factories (:func:`joint_axis`,
        :func:`joint_pair_distance`, :func:`joint_speed`,
        :func:`joint_angle`).
    person_index
        Which detected person to extract from each frame. Defaults to
        0 (the first detected person), matching the single-subject
        clinical case.
    min_distance_seconds, min_prominence, min_height, pad_seconds
        Forwarded to :func:`segment_by_peaks` after the time-based
        parameters are converted to sample counts via ``metadata.fps``.

    Returns
    -------
    Segmentation
        A :class:`~neuropose.io.Segmentation` pairing the segments with
        the exact :class:`~neuropose.io.SegmentationConfig` that
        produced them. Ready to attach under a name to
        ``VideoPredictions.segmentations``.

    Raises
    ------
    ValueError
        If ``predictions`` has zero frames, if a joint index in the
        extractor is out of range, or if ``predictions.metadata.fps``
        is non-positive and time-based parameters were supplied.
    ImportError
        If :mod:`scipy` is not installed.
    """
    sequence = predictions_to_numpy(predictions, person_index=person_index)
    signal = extract_signal(sequence, extractor)

    fps = predictions.metadata.fps
    needs_fps = (
        min_distance_seconds is not None and min_distance_seconds > 0.0
    ) or pad_seconds > 0.0
    if needs_fps and fps <= 0.0:
        raise ValueError(
            "cannot convert seconds to frames: metadata.fps is "
            f"{fps}. Pass min_distance_seconds=None and pad_seconds=0.0, "
            "or call segment_by_peaks directly with sample-count units."
        )

    if min_distance_seconds is None or min_distance_seconds <= 0.0:
        min_distance: int | None = None
    else:
        min_distance = max(1, round(min_distance_seconds * fps))

    pad = round(pad_seconds * fps) if pad_seconds > 0.0 else 0

    segments = segment_by_peaks(
        signal,
        min_distance=min_distance,
        min_prominence=min_prominence,
        min_height=min_height,
        pad=pad,
    )

    config = SegmentationConfig(
        extractor=extractor,
        person_index=person_index,
        min_distance_seconds=min_distance_seconds,
        min_prominence=min_prominence,
        min_height=min_height,
        pad_seconds=pad_seconds,
    )
    return Segmentation(config=config, segments=segments)


# ---------------------------------------------------------------------------
# Gait-cycle segmentation
# ---------------------------------------------------------------------------


def segment_gait_cycles(
    predictions: VideoPredictions,
    *,
    joint: str = "rhee",
    axis: AxisLetter = "y",
    invert: bool = False,
    min_cycle_seconds: float = 0.4,
    min_prominence: float | None = None,
) -> Segmentation:
    """Segment gait cycles from a single heel's vertical trace.

    Runs valley-to-valley peak detection (the same engine used by
    :func:`segment_predictions`) on the chosen joint's coordinate along
    the chosen spatial axis. By default, each detected peak corresponds
    to one heel-strike — the frame where the heel reaches its lowest
    point on the Y-down MeTRAbs world-coordinate convention — and the
    returned :class:`~neuropose.io.Segment` windows span one full gait
    cycle from the preceding toe-off valley to the following toe-off
    valley.

    The function is a **thin wrapper** over :func:`segment_predictions`
    with a :func:`joint_axis` extractor; it exists to give clinical
    callers a gait-specific entry point with meaningful defaults
    (``joint="rhee"``, ``axis="y"``, ``min_cycle_seconds=0.4``)
    rather than forcing them to construct the extractor by hand.

    Parameters
    ----------
    predictions
        Per-video predictions to segment. ``metadata.fps`` is used to
        translate ``min_cycle_seconds`` into a sample-count distance
        threshold.
    joint
        Joint name in the berkeley_mhad_43 skeleton — typically
        ``"rhee"`` (right heel) or ``"lhee"`` (left heel). Resolved
        via :func:`joint_index`.
    axis
        Spatial axis to track, as ``"x"``, ``"y"``, or ``"z"``. The
        default ``"y"`` matches the vertical axis in MeTRAbs's output
        (Y-down world coordinates).
    invert
        If ``True``, negate the extracted signal so that minima
        become peaks. Needed when the recording convention makes a
        heel-strike appear as a *decrease* in the chosen coordinate
        — for example, a camera orientation where the vertical axis
        runs bottom-to-top instead of MeTRAbs's default top-to-bottom.
    min_cycle_seconds
        Minimum gait-cycle duration. Used as scipy's
        ``find_peaks(distance=...)`` parameter after conversion to
        frame count via ``metadata.fps``. Defaults to ``0.4`` seconds,
        which rejects noise peaks on even the fastest human gaits
        (~120 strides/min) while retaining every real cadence.
    min_prominence
        Forwarded to :func:`segment_by_peaks` to filter out shallow
        local maxima that aren't real heel-strikes. In MeTRAbs units
        (millimetres) a threshold of 20 to 50 mm is typical for
        able-bodied gait; leave ``None`` to accept every peak scipy
        identifies.

    Returns
    -------
    Segmentation
        A :class:`~neuropose.io.Segmentation` paired with the full
        :class:`~neuropose.io.SegmentationConfig` that produced it, so
        the output is self-describing when persisted. The segments
        list is **empty** rather than an exception when no peaks are
        detected — a common outcome for shuffling gaits or
        walker-assisted trials.

    Raises
    ------
    KeyError
        If ``joint`` is not a known berkeley_mhad_43 joint name.
    ValueError
        If ``axis`` is not one of ``"x"``, ``"y"``, ``"z"``, or if
        ``predictions`` has zero frames, or if ``metadata.fps`` is
        non-positive.
    ImportError
        If :mod:`scipy` is not installed.
    """
    if axis not in _AXIS_INDICES:
        raise ValueError(f"axis must be one of 'x', 'y', 'z'; got {axis!r}")
    joint_idx = joint_index(joint)
    axis_idx = _AXIS_INDICES[axis]
    extractor = joint_axis(joint_idx, axis_idx, invert=invert)
    return segment_predictions(
        predictions,
        extractor,
        min_distance_seconds=min_cycle_seconds,
        min_prominence=min_prominence,
    )


def segment_gait_cycles_bilateral(
    predictions: VideoPredictions,
    *,
    axis: AxisLetter = "y",
    invert: bool = False,
    min_cycle_seconds: float = 0.4,
    min_prominence: float | None = None,
) -> dict[str, Segmentation]:
    """Segment gait cycles for both heels.

    Runs :func:`segment_gait_cycles` twice — once with ``joint="lhee"``
    and once with ``joint="rhee"`` — and returns the two results under
    the keys ``"left_heel_strikes"`` and ``"right_heel_strikes"``. The
    returned dict is shape-compatible with
    :class:`~neuropose.io.VideoPredictions.segmentations` so it can be
    merged directly into a predictions object and persisted to
    ``results.json`` via the usual save path.

    Parameters
    ----------
    predictions, axis, invert, min_cycle_seconds, min_prominence
        Forwarded to :func:`segment_gait_cycles`; see that function's
        docstring for details.

    Returns
    -------
    dict[str, Segmentation]
        Two-keyed mapping with the left and right heel segmentations
        under ``"left_heel_strikes"`` and ``"right_heel_strikes"``.
        Either side may carry an empty segments list if its heel's
        trace contained no detectable strikes.
    """
    return {
        "left_heel_strikes": segment_gait_cycles(
            predictions,
            joint="lhee",
            axis=axis,
            invert=invert,
            min_cycle_seconds=min_cycle_seconds,
            min_prominence=min_prominence,
        ),
        "right_heel_strikes": segment_gait_cycles(
            predictions,
            joint="rhee",
            axis=axis,
            invert=invert,
            min_cycle_seconds=min_cycle_seconds,
            min_prominence=min_prominence,
        ),
    }


# ---------------------------------------------------------------------------
# Slicing: one VideoPredictions per segment
# ---------------------------------------------------------------------------


def slice_predictions(
    predictions: VideoPredictions,
    segments: Sequence[Segment],
) -> list[VideoPredictions]:
    """Split a :class:`~neuropose.io.VideoPredictions` into one per segment.

    Each output :class:`~neuropose.io.VideoPredictions` contains only
    the frames in ``[segment.start, segment.end)`` of the input,
    re-keyed starting at ``frame_000000`` and with
    :class:`~neuropose.io.VideoMetadata.frame_count` rewritten to match.
    Other metadata fields (``fps``, ``width``, ``height``) are copied
    verbatim. The input is not modified.

    The per-slice ``segmentations`` field is intentionally reset to
    empty; a segment is meaningful only against its parent video's
    timeline, and copying the parent's segmentations into a sliced view
    would be confusing rather than useful.

    Parameters
    ----------
    predictions
        Source predictions.
    segments
        Segments to slice out. Indices are interpreted into the input's
        ``frame_names()`` ordering.

    Returns
    -------
    list[VideoPredictions]
        One :class:`~neuropose.io.VideoPredictions` per segment, in the
        order of the input list.

    Raises
    ------
    ValueError
        If any segment's ``end`` exceeds the input frame count.
    """
    frame_names = predictions.frame_names()
    total_frames = len(frame_names)
    results: list[VideoPredictions] = []
    for idx, seg in enumerate(segments):
        if seg.end > total_frames:
            raise ValueError(f"segment {idx} end={seg.end} exceeds frame count {total_frames}")
        sliced_frames = {
            f"frame_{i:06d}": predictions[src_name]
            for i, src_name in enumerate(frame_names[seg.start : seg.end])
        }
        new_metadata = VideoMetadata(
            frame_count=seg.end - seg.start,
            fps=predictions.metadata.fps,
            width=predictions.metadata.width,
            height=predictions.metadata.height,
        )
        results.append(
            VideoPredictions(
                metadata=new_metadata,
                frames=sliced_frames,
                segmentations={},
            )
        )
    return results
