"""Dynamic Time Warping helpers for pose sequence comparison.

Three entry points, ordered by increasing precision (and increasing cost):

- :func:`dtw_all` — DTW on the flattened per-frame joint vector. Fast but
  coarse; collapses every joint axis into a single per-frame vector.
- :func:`dtw_per_joint` — DTW on each joint independently. Preserves
  per-joint temporal alignment at the cost of one DTW call per joint.
- :func:`dtw_relation` — DTW on the displacement vector between two
  specific joints. This is the right tool when the research question is
  about the *relative* motion of a specific pair of joints (e.g. the
  hand-to-hip vector during a reach-and-grasp trial).

All three return a :class:`DTWResult` dataclass with the DTW distance
and the warping path. Inputs are expected to be ``(frames, joints, 3)``
numpy arrays — the shape :func:`~neuropose.analyzer.features.predictions_to_numpy`
produces.

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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


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


def dtw_all(a: np.ndarray, b: np.ndarray) -> DTWResult:
    """DTW on the flattened per-frame joint vector.

    Each frame's joints are collapsed into a single vector before DTW
    is applied. This is fast — one DTW call regardless of the joint
    count — but loses per-joint temporal structure, so a small
    timing mismatch on one joint can dominate the distance metric.

    Parameters
    ----------
    a, b
        Pose sequences as ``(frames, joints, 3)`` numpy arrays. The two
        sequences do not need to have the same number of frames, but
        they must have the same number of joints.

    Returns
    -------
    DTWResult
        The DTW distance and warping path between the flattened
        sequences.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` do not have the same joint count.
    """
    _validate_same_joint_count(a, b)
    fastdtw, euclidean = _require_fastdtw()
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    distance, path = fastdtw(a_flat, b_flat, dist=euclidean)
    return DTWResult(distance=float(distance), path=[tuple(p) for p in path])


def dtw_per_joint(a: np.ndarray, b: np.ndarray) -> list[DTWResult]:
    """DTW on each joint independently.

    Performs one DTW computation per joint, yielding a list of
    :class:`DTWResult` objects in joint-index order. More precise than
    :func:`dtw_all` because each joint's temporal alignment is optimised
    separately, at the cost of J times more DTW calls for J joints.

    Parameters
    ----------
    a, b
        Pose sequences as ``(frames, joints, 3)`` numpy arrays. The two
        sequences do not need to have the same number of frames but
        must have the same number of joints.

    Returns
    -------
    list[DTWResult]
        One DTW result per joint, in index order.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` do not have the same joint count.
    """
    _validate_same_joint_count(a, b)
    fastdtw, euclidean = _require_fastdtw()
    results: list[DTWResult] = []
    for joint_idx in range(a.shape[1]):
        a_joint = a[:, joint_idx, :]
        b_joint = b[:, joint_idx, :]
        distance, path = fastdtw(a_joint, b_joint, dist=euclidean)
        results.append(DTWResult(distance=float(distance), path=[tuple(p) for p in path]))
    return results


def dtw_relation(
    a: np.ndarray,
    b: np.ndarray,
    joint_i: int,
    joint_j: int,
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

    Returns
    -------
    DTWResult
        DTW distance and path between the two displacement sequences.

    Raises
    ------
    ValueError
        If the sequences have different joint counts or if either joint
        index is out of range.
    """
    _validate_same_joint_count(a, b)
    num_joints = a.shape[1]
    if not (0 <= joint_i < num_joints) or not (0 <= joint_j < num_joints):
        raise ValueError(
            f"joint indices must be in [0, {num_joints}); got joint_i={joint_i}, joint_j={joint_j}"
        )
    fastdtw, euclidean = _require_fastdtw()
    disp_a = a[:, joint_j, :] - a[:, joint_i, :]
    disp_b = b[:, joint_j, :] - b[:, joint_i, :]
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
