"""Dynamic Time Warping helpers for pose sequence comparison.

Three entry points, ordered by increasing precision (and increasing cost):

- :func:`dtw_all` — DTW on the flattened per-frame feature vector. Fast
  but coarse; collapses every joint axis (or every angle triplet) into
  a single per-frame vector.
- :func:`dtw_per_joint` — DTW on each joint (or angle triplet)
  independently. Preserves per-unit temporal alignment at the cost of
  one DTW call per unit.
- :func:`dtw_relation` — DTW on the displacement vector between two
  specific joints. This is the right tool when the research question is
  about the *relative* motion of a specific pair of joints (e.g. the
  hand-to-hip vector during a reach-and-grasp trial).

All three return a :class:`DTWResult` dataclass with the DTW distance
and the warping path. Inputs are expected to be ``(frames, joints, 3)``
numpy arrays — the shape :func:`~neuropose.analyzer.features.predictions_to_numpy`
produces.

Three orthogonal preprocessing knobs are available on the entry points:

- **``align``** routes the inputs through
  :func:`~neuropose.analyzer.features.procrustes_align` before DTW runs,
  yielding translation- and rotation-invariant distances.
  ``align="none"`` (the default) preserves the raw-coordinate behaviour
  shipped in 0.1.
- **``representation``** (on :func:`dtw_all` and :func:`dtw_per_joint`)
  selects what each frame is reduced to before DTW. ``"coords"`` uses
  the raw joint coordinates; ``"angles"`` replaces them with joint
  angles computed at caller-supplied triplets via
  :func:`~neuropose.analyzer.features.extract_joint_angles`, giving
  DTW distances that are directly interpretable as clinical joint-range
  comparisons.
- **``nan_policy``** decides how the DTW path handles non-finite values
  in its input — typically a concern only for the angle representation,
  where degenerate (zero-length) vectors produce NaN. See :data:`NanPolicy`.

Dependency note
---------------
This module requires :mod:`fastdtw` and :mod:`scipy`, which are part of
the ``analysis`` optional extra. Imports are performed lazily inside
:func:`_require_fastdtw` so that ``import neuropose.analyzer.dtw``
succeeds even when the extra is not installed; the error surfaces with
a clear installation hint the first time a DTW function is actually
called.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from neuropose.analyzer.features import extract_joint_angles, procrustes_align

AlignMode = Literal["none", "procrustes_per_frame", "procrustes_per_sequence"]
"""Alignment selector for DTW entry points.

- ``"none"`` — feed raw coordinates directly to DTW.
- ``"procrustes_per_frame"`` — per-frame Kabsch alignment before DTW.
- ``"procrustes_per_sequence"`` — single sequence-wide Kabsch
  alignment before DTW.
"""

Representation = Literal["coords", "angles"]
"""Per-frame feature representation for :func:`dtw_all` and :func:`dtw_per_joint`.

- ``"coords"`` — use the raw joint coordinates (the input's last two
  axes). Preserves the 0.1 behaviour.
- ``"angles"`` — replace joints with joint angles at caller-supplied
  triplets. Translation- and rotation-invariant by construction,
  scale-invariant modulo the upstream normalization, and directly
  interpretable in clinical terms ("knee flexion during swing phase").
  The ``angle_triplets`` keyword becomes mandatory in this mode.
"""

NanPolicy = Literal["propagate", "interpolate", "drop"]
"""Per-feature NaN handling for the DTW input.

NaN typically appears when ``representation="angles"`` encounters a
degenerate (zero-length) vector — the angle is undefined and
:func:`extract_joint_angles` propagates NaN rather than quietly returning
a stand-in value.

- ``"propagate"`` (default) — pass NaN straight through to the DTW
  engine. fastdtw validates its input via
  :func:`numpy.asarray_chkfinite` and raises :class:`ValueError`
  the moment a NaN appears, which is the safest default because it
  makes the problem visible instead of quietly corrupting a
  distance.
- ``"interpolate"`` — linearly interpolate NaN frames along each
  feature column using neighbouring finite values. Reasonable when a
  small number of frames are corrupted and the surrounding motion is
  smooth; inappropriate when long stretches are missing.
- ``"drop"`` — remove any frame where *any* feature is NaN before DTW
  runs. Simple, but compresses the time axis, so warping-path indices
  refer to the *compacted* sequence rather than the original.
"""


@dataclass(frozen=True)
class DTWResult:
    """Result of a single DTW computation.

    Attributes
    ----------
    distance
        Scalar DTW distance between the two input sequences.
    path
        Warping path as a list of ``(i, j)`` index pairs, where ``i`` is
        an index into the first sequence and ``j`` is an index into the
        second.
    """

    distance: float
    path: list[tuple[int, int]]


def _require_fastdtw() -> tuple[Callable, Callable]:
    """Lazily import fastdtw and scipy.spatial.distance.euclidean.

    Returns
    -------
    tuple
        ``(fastdtw_callable, euclidean_callable)``.

    Raises
    ------
    ImportError
        If either ``fastdtw`` or ``scipy`` is unavailable. The message
        points the user at the ``analysis`` optional-dependencies extra.
    """
    try:
        from fastdtw import fastdtw  # type: ignore[attr-defined]
        from scipy.spatial.distance import euclidean
    except ImportError as exc:
        raise ImportError(
            "neuropose.analyzer.dtw requires fastdtw and scipy. "
            "Install them with: pip install neuropose[analysis]"
        ) from exc
    return fastdtw, euclidean


def dtw_all(
    a: np.ndarray,
    b: np.ndarray,
    *,
    align: AlignMode = "none",
    representation: Representation = "coords",
    angle_triplets: Sequence[tuple[int, int, int]] | None = None,
    nan_policy: NanPolicy = "propagate",
) -> DTWResult:
    """DTW on the flattened per-frame feature vector.

    Under the default ``representation="coords"`` each frame's joints
    are collapsed into a single vector before DTW is applied — fast
    (one DTW call regardless of joint count) but coarse, since a small
    timing mismatch on one joint can dominate the distance metric.
    Switching to ``representation="angles"`` computes joint angles at
    the supplied triplets first and flattens those instead.

    Parameters
    ----------
    a, b
        Pose sequences as ``(frames, joints, 3)`` numpy arrays. The two
        sequences do not need to have the same number of frames, but
        they must have the same number of joints. When ``align`` is not
        ``"none"``, the two sequences must additionally share a frame
        count (Procrustes requires a 1:1 correspondence).
    align
        Procrustes alignment mode applied before DTW. See
        :data:`AlignMode`.
    representation
        Per-frame feature representation. See :data:`Representation`.
    angle_triplets
        Required when ``representation="angles"``. Sequence of
        ``(a, b, c)`` joint-index triplets passed through to
        :func:`~neuropose.analyzer.features.extract_joint_angles`.
        Ignored otherwise.
    nan_policy
        How to handle NaN values in the DTW input. See :data:`NanPolicy`.

    Returns
    -------
    DTWResult
        The DTW distance and warping path between the flattened
        sequences.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` do not have the same joint count, if
        ``align`` requires a matching frame count that is not present,
        if ``representation="angles"`` is requested without
        ``angle_triplets``, or if ``nan_policy="interpolate"``
        encounters an all-NaN column.
    """
    _validate_same_joint_count(a, b)
    a, b = _maybe_align(a, b, align=align)
    feat_a = _apply_representation(a, representation, angle_triplets=angle_triplets)
    feat_b = _apply_representation(b, representation, angle_triplets=angle_triplets)
    feat_a = _apply_nan_policy(feat_a, nan_policy)
    feat_b = _apply_nan_policy(feat_b, nan_policy)
    fastdtw, euclidean = _require_fastdtw()
    distance, path = fastdtw(feat_a, feat_b, dist=euclidean)
    return DTWResult(distance=float(distance), path=[tuple(p) for p in path])


def dtw_per_joint(
    a: np.ndarray,
    b: np.ndarray,
    *,
    align: AlignMode = "none",
    representation: Representation = "coords",
    angle_triplets: Sequence[tuple[int, int, int]] | None = None,
    nan_policy: NanPolicy = "propagate",
) -> list[DTWResult]:
    """DTW on each joint (or angle triplet) independently.

    Performs one DTW computation per unit, yielding a list of
    :class:`DTWResult` objects in input order. More precise than
    :func:`dtw_all` because each unit's temporal alignment is optimised
    separately, at the cost of J times more DTW calls for J units.

    Under the default ``representation="coords"`` a "unit" is one of
    the input's joints (xyz treated jointly). Under
    ``representation="angles"`` a "unit" is one scalar angle column
    computed from one ``angle_triplets`` entry.

    Parameters
    ----------
    a, b
        Pose sequences as ``(frames, joints, 3)`` numpy arrays. The two
        sequences do not need to have the same number of frames but
        must have the same number of joints. When ``align`` is not
        ``"none"``, they must additionally share a frame count.
    align
        Procrustes alignment mode applied before DTW. See
        :data:`AlignMode`.
    representation
        Per-frame feature representation. See :data:`Representation`.
    angle_triplets
        Required when ``representation="angles"``; see
        :func:`dtw_all` for details.
    nan_policy
        How to handle NaN values in the DTW input. See :data:`NanPolicy`.

    Returns
    -------
    list[DTWResult]
        One DTW result per joint or per angle triplet, in input order.

    Raises
    ------
    ValueError
        Same conditions as :func:`dtw_all`.
    """
    _validate_same_joint_count(a, b)
    a, b = _maybe_align(a, b, align=align)

    if representation == "coords":
        feat_a = a
        feat_b = b
        # (frames, joints, 3) — one DTW per joint over its (frames, 3) slice.
        num_units = feat_a.shape[1]
        slicers: list[Callable[[np.ndarray], np.ndarray]] = [
            (lambda arr, idx=i: arr[:, idx, :]) for i in range(num_units)
        ]
    else:  # "angles"
        if angle_triplets is None:
            raise ValueError("representation='angles' requires angle_triplets")
        feat_a = extract_joint_angles(a, angle_triplets)  # (frames, num_triplets)
        feat_b = extract_joint_angles(b, angle_triplets)
        num_units = feat_a.shape[1]
        slicers = [
            # Scalar columns become 2D for DTW (fastdtw expects a
            # sequence of vectors, not a sequence of scalars).
            (lambda arr, idx=i: arr[:, idx : idx + 1])
            for i in range(num_units)
        ]

    fastdtw, euclidean = _require_fastdtw()
    results: list[DTWResult] = []
    for slicer in slicers:
        unit_a = _apply_nan_policy(slicer(feat_a), nan_policy)
        unit_b = _apply_nan_policy(slicer(feat_b), nan_policy)
        distance, path = fastdtw(unit_a, unit_b, dist=euclidean)
        results.append(DTWResult(distance=float(distance), path=[tuple(p) for p in path]))
    return results


def dtw_relation(
    a: np.ndarray,
    b: np.ndarray,
    joint_i: int,
    joint_j: int,
    *,
    align: AlignMode = "none",
    nan_policy: NanPolicy = "propagate",
) -> DTWResult:
    """DTW on the displacement vector between two specific joints.

    For each frame, the input is reduced to the vector from ``joint_i``
    to ``joint_j``. DTW is then applied to the two sequences of
    displacement vectors. This is the right tool when the question is
    "how does the relationship between joint A and joint B change over
    time?" — for example, "does the subject's hand track a consistent
    distance from the hip during the reach trial?"

    Parameters
    ----------
    a, b
        Pose sequences as ``(frames, joints, 3)`` numpy arrays.
    joint_i, joint_j
        Indices of the two joints whose relative position should be
        compared. Must be valid indices into ``a`` and ``b``'s joint
        axis.
    align
        Procrustes alignment mode applied to the full sequences
        before the displacement vectors are extracted. See
        :data:`AlignMode`. Note that displacement vectors are already
        translation-invariant; alignment is still useful for cancelling
        camera rotation between trials.
    nan_policy
        How to handle NaN values in the DTW input. See :data:`NanPolicy`.

    Returns
    -------
    DTWResult
        DTW distance and path between the two displacement sequences.

    Raises
    ------
    ValueError
        If the sequences have different joint counts, either joint
        index is out of range, or ``align`` requires a matching frame
        count that is not present.
    """
    _validate_same_joint_count(a, b)
    num_joints = a.shape[1]
    if not (0 <= joint_i < num_joints) or not (0 <= joint_j < num_joints):
        raise ValueError(
            f"joint indices must be in [0, {num_joints}); got joint_i={joint_i}, joint_j={joint_j}"
        )
    a, b = _maybe_align(a, b, align=align)
    disp_a = _apply_nan_policy(a[:, joint_j, :] - a[:, joint_i, :], nan_policy)
    disp_b = _apply_nan_policy(b[:, joint_j, :] - b[:, joint_i, :], nan_policy)
    fastdtw, euclidean = _require_fastdtw()
    distance, path = fastdtw(disp_a, disp_b, dist=euclidean)
    return DTWResult(distance=float(distance), path=[tuple(p) for p in path])


def _validate_same_joint_count(a: np.ndarray, b: np.ndarray) -> None:
    """Raise :class:`ValueError` if ``a`` and ``b`` disagree on joint count."""
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(
            f"expected 3D arrays of shape (frames, joints, 3); got a.ndim={a.ndim}, b.ndim={b.ndim}"
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"input arrays disagree on joint count: "
            f"a has {a.shape[1]} joints, b has {b.shape[1]} joints"
        )


def _maybe_align(
    a: np.ndarray,
    b: np.ndarray,
    *,
    align: AlignMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Procrustes alignment if ``align`` requests it.

    Procrustes requires a frame-by-frame correspondence, so this
    helper rejects calls where the two sequences disagree on frame
    count and ``align`` is not ``"none"``. Pad upstream with
    :func:`~neuropose.analyzer.features.pad_sequences` if the lengths
    differ.
    """
    if align == "none":
        return a, b
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"align={align!r} requires matching frame counts; "
            f"got a with {a.shape[0]} frames and b with {b.shape[0]} frames"
        )
    mode = "per_frame" if align == "procrustes_per_frame" else "per_sequence"
    aligned_a, _target, _diag = procrustes_align(a, b, mode=mode)
    return aligned_a, b


def _apply_representation(
    sequence: np.ndarray,
    representation: Representation,
    *,
    angle_triplets: Sequence[tuple[int, int, int]] | None,
) -> np.ndarray:
    """Reduce a ``(frames, joints, 3)`` sequence to DTW-ready 2D features.

    ``"coords"`` reshapes to ``(frames, joints * 3)``; ``"angles"``
    runs :func:`extract_joint_angles` to produce
    ``(frames, len(angle_triplets))``.
    """
    if representation == "coords":
        return sequence.reshape(sequence.shape[0], -1)
    if representation == "angles":
        if angle_triplets is None:
            raise ValueError("representation='angles' requires angle_triplets")
        return extract_joint_angles(sequence, angle_triplets)
    raise ValueError(f"unknown representation {representation!r}")


def _apply_nan_policy(features: np.ndarray, policy: NanPolicy) -> np.ndarray:
    """Handle NaN values in a ``(frames, features)`` array per ``policy``.

    ``"propagate"`` is a no-op. ``"interpolate"`` runs 1D linear
    interpolation along the frame axis within each feature column,
    leaving finite data untouched. ``"drop"`` removes any frame where
    *any* feature is NaN.

    Raises
    ------
    ValueError
        If ``"interpolate"`` encounters a column that is entirely NaN
        (no finite anchors to interpolate between), or if ``"drop"``
        leaves an empty sequence.
    """
    if policy == "propagate":
        return features
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if policy == "drop":
        keep = np.isfinite(features).all(axis=1)
        dropped = features[keep]
        if dropped.shape[0] == 0:
            raise ValueError(
                "nan_policy='drop' removed every frame; DTW needs a non-empty sequence"
            )
        return dropped
    if policy == "interpolate":
        out = features.astype(float, copy=True)
        num_frames = out.shape[0]
        indices = np.arange(num_frames, dtype=float)
        for col in range(out.shape[1]):
            column = out[:, col]
            finite = np.isfinite(column)
            if finite.all():
                continue
            if not finite.any():
                raise ValueError(
                    f"nan_policy='interpolate' cannot fill column {col}: all values are NaN"
                )
            out[:, col] = np.interp(indices, indices[finite], column[finite])
        return out
    raise ValueError(f"unknown nan_policy {policy!r}")
