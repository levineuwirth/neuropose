"""Feature extraction helpers for pose sequences.

All functions in this module operate on numpy arrays of shape
``(frames, joints, 3)`` — the output of
:func:`predictions_to_numpy`. They are pure functions: none of them
mutate their inputs, and none of them touch the filesystem or the
model.

The following helpers are provided:

- :func:`predictions_to_numpy` — convert a validated
  :class:`~neuropose.io.VideoPredictions` into a numpy pose sequence.
- :func:`normalize_pose_sequence` — scale a sequence so joint positions
  fit in the unit cube (either per-axis or uniform).
- :func:`pad_sequences` — edge-pad a batch of sequences to a common
  length, suitable for downstream tensor-based analysis.
- :func:`extract_joint_angles` — compute joint angles at specified
  triplet positions across a pose sequence.
- :func:`extract_feature_statistics` — summary statistics
  (mean / std / min / max / range) for a 1D feature series.
- :func:`find_peaks` — thin :mod:`scipy.signal` wrapper returning only
  the peak indices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from neuropose.io import VideoPredictions


# ---------------------------------------------------------------------------
# VideoPredictions → numpy
# ---------------------------------------------------------------------------


def predictions_to_numpy(
    predictions: VideoPredictions,
    *,
    person_index: int = 0,
) -> np.ndarray:
    """Convert a :class:`VideoPredictions` to a 3D pose sequence.

    Parameters
    ----------
    predictions
        The predictions to convert.
    person_index
        Which detected person to extract per frame. Defaults to ``0``
        (the first detected person) which matches the single-subject
        clinical case. Frames that do not have at least
        ``person_index + 1`` detections raise :class:`ValueError`.

    Returns
    -------
    numpy.ndarray
        A ``(frames, joints, 3)`` array in the same physical units as
        the underlying predictions (millimetres for MeTRAbs output).

    Raises
    ------
    ValueError
        If any frame lacks sufficient detections for ``person_index``,
        or if the predictions contain no frames.
    """
    if len(predictions) == 0:
        raise ValueError("predictions contains zero frames")
    frames: list[list[list[float]]] = []
    for frame_name in predictions.frame_names():
        per_person = predictions[frame_name].poses3d
        if person_index >= len(per_person):
            raise ValueError(
                f"frame {frame_name} has {len(per_person)} detections; "
                f"person_index={person_index} is out of range"
            )
        frames.append(per_person[person_index])
    return np.asarray(frames, dtype=float)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_pose_sequence(
    sequence: np.ndarray,
    *,
    axis_wise: bool = False,
) -> np.ndarray:
    """Translate and scale a pose sequence so joints fit in the unit cube.

    The minimum coordinate along each spatial axis is subtracted, and
    the result is divided by a single scalar (the range of the largest
    axis, when ``axis_wise=False``) or by the per-axis range (when
    ``axis_wise=True``).

    Parameters
    ----------
    sequence
        Array of shape ``(frames, joints, 3)``.
    axis_wise
        If ``False`` (default), preserve the geometric aspect ratio by
        using a single scalar denominator (the maximum axis extent).
        If ``True``, scale each axis independently to ``[0, 1]``, which
        distorts the geometry but guarantees full-range normalization
        on every axis.

    Returns
    -------
    numpy.ndarray
        A new array of the same shape as ``sequence``, with joint
        positions translated to start at the origin and scaled as
        described above. The input is not modified.

    Raises
    ------
    ValueError
        If ``sequence`` does not have a final axis of size 3, or if the
        sequence is degenerate (zero extent on every axis).
    """
    if sequence.ndim != 3 or sequence.shape[-1] != 3:
        raise ValueError(
            f"expected (frames, joints, 3); got shape {sequence.shape}"
        )
    result = sequence.astype(float, copy=True)
    mins = result.reshape(-1, 3).min(axis=0)
    maxs = result.reshape(-1, 3).max(axis=0)
    ranges = maxs - mins

    if np.all(ranges == 0):
        raise ValueError("cannot normalize a degenerate (zero-extent) sequence")

    result -= mins  # broadcasts over (frames, joints, 3)

    if axis_wise:
        # Replace zero ranges with 1 to avoid division-by-zero on axes
        # where all joints share a coordinate; those axes will remain 0.
        safe_ranges = np.where(ranges == 0, 1.0, ranges)
        result = result / safe_ranges
    else:
        scale = float(ranges.max())
        result = result / scale

    return result


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def pad_sequences(
    sequences: Sequence[np.ndarray],
    *,
    target_length: int | None = None,
) -> list[np.ndarray]:
    """Edge-pad a list of pose sequences to a common length.

    Each input sequence is extended by repeating its last frame until
    it reaches ``target_length``. Sequences that are already longer
    than ``target_length`` are **truncated** to that length. The input
    list itself and the input arrays are never mutated.

    Parameters
    ----------
    sequences
        List of ``(frames_i, joints, 3)`` arrays. ``frames_i`` may
        differ per sequence, but all sequences must share the same
        joint count and spatial dimensionality.
    target_length
        Desired number of frames. If ``None``, the maximum
        ``frames_i`` in the input is used.

    Returns
    -------
    list[numpy.ndarray]
        A new list of arrays, each with ``target_length`` frames.

    Raises
    ------
    ValueError
        If ``sequences`` is empty and ``target_length`` is ``None``, or
        if any sequence has mismatched trailing dimensions.
    """
    if not sequences:
        if target_length is None:
            raise ValueError(
                "cannot infer target_length from an empty sequence list"
            )
        return []

    first = sequences[0]
    trailing_shape = first.shape[1:]
    for idx, seq in enumerate(sequences):
        if seq.shape[1:] != trailing_shape:
            raise ValueError(
                f"sequence {idx} has trailing shape {seq.shape[1:]}; "
                f"expected {trailing_shape}"
            )

    length = target_length if target_length is not None else max(
        s.shape[0] for s in sequences
    )

    padded: list[np.ndarray] = []
    for seq in sequences:
        if seq.shape[0] == length:
            padded.append(seq.copy())
        elif seq.shape[0] > length:
            padded.append(seq[:length].copy())
        else:
            pad_amount = length - seq.shape[0]
            padding = [(0, pad_amount)] + [(0, 0)] * (seq.ndim - 1)
            padded.append(np.pad(seq, padding, mode="edge"))
    return padded


# ---------------------------------------------------------------------------
# Joint angles
# ---------------------------------------------------------------------------


def extract_joint_angles(
    sequence: np.ndarray,
    triplets: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    """Compute angles at specified joints across a pose sequence.

    For each triplet ``(a, b, c)``, the angle at ``b`` is defined as
    the angle between the vectors ``(a - b)`` and ``(c - b)``, in
    radians. Frames where either vector has zero length (e.g. joint
    degeneracy) produce ``NaN`` in the output rather than raising.

    Parameters
    ----------
    sequence
        Array of shape ``(frames, joints, 3)``.
    triplets
        Iterable of ``(a, b, c)`` joint index tuples. Each index must
        be in ``[0, joints)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(frames, len(triplets))``, where each column
        is the time-series of angles at the corresponding triplet's
        centre joint. Angles are in radians, in ``[0, pi]``.

    Raises
    ------
    ValueError
        If any joint index in ``triplets`` is out of range.
    """
    if sequence.ndim != 3 or sequence.shape[-1] != 3:
        raise ValueError(
            f"expected (frames, joints, 3); got shape {sequence.shape}"
        )
    num_joints = sequence.shape[1]
    columns: list[np.ndarray] = []
    for a_idx, b_idx, c_idx in triplets:
        for idx in (a_idx, b_idx, c_idx):
            if not (0 <= idx < num_joints):
                raise ValueError(
                    f"joint index {idx} out of range [0, {num_joints})"
                )
        v1 = sequence[:, a_idx, :] - sequence[:, b_idx, :]
        v2 = sequence[:, c_idx, :] - sequence[:, b_idx, :]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.sum(v1 * v2, axis=1) / (n1 * n2)
            cosine = np.clip(cosine, -1.0, 1.0)
            angle = np.arccos(cosine)
        columns.append(angle)
    return np.stack(columns, axis=1)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureStatistics:
    """Summary statistics for a 1D feature series.

    Attributes
    ----------
    mean, std, min, max
        Standard summary statistics of the input values.
    range
        ``max - min``. Precomputed for convenience.
    """

    mean: float
    std: float
    min: float
    max: float
    range: float


def extract_feature_statistics(values: np.ndarray) -> FeatureStatistics:
    """Compute summary statistics for a 1D feature series.

    Parameters
    ----------
    values
        A 1D numpy array. Higher-dimensional inputs are rejected to
        keep the semantics unambiguous — callers that want per-column
        statistics should reduce along their axis of interest first.

    Returns
    -------
    FeatureStatistics
        Mean, standard deviation, minimum, maximum, and range of
        ``values``.

    Raises
    ------
    ValueError
        If ``values`` is not 1D or is empty.
    """
    if values.ndim != 1:
        raise ValueError(f"expected 1D array; got shape {values.shape}")
    if values.size == 0:
        raise ValueError("cannot compute statistics of an empty array")
    mn = float(values.min())
    mx = float(values.max())
    return FeatureStatistics(
        mean=float(values.mean()),
        std=float(values.std()),
        min=mn,
        max=mx,
        range=mx - mn,
    )


# ---------------------------------------------------------------------------
# Peak finding
# ---------------------------------------------------------------------------


def find_peaks(values: np.ndarray, **kwargs: object) -> np.ndarray:
    """Return indices of local maxima in a 1D series.

    Thin wrapper around :func:`scipy.signal.find_peaks` that returns
    just the peak-index array (scipy's function also returns a
    properties dict, which callers rarely need).

    Parameters
    ----------
    values
        1D numpy array of feature values (e.g. a joint's Y-coordinate
        across frames).
    **kwargs
        Forwarded to :func:`scipy.signal.find_peaks`. Common options
        include ``height``, ``threshold``, ``distance``, and
        ``prominence``.

    Returns
    -------
    numpy.ndarray
        1D integer array of peak indices, in ascending order.

    Raises
    ------
    ImportError
        If scipy is not installed. The error message points at the
        ``analysis`` optional extra.
    ValueError
        If ``values`` is not 1D.
    """
    if values.ndim != 1:
        raise ValueError(f"expected 1D array; got shape {values.shape}")
    try:
        from scipy.signal import find_peaks as _sp_find_peaks
    except ImportError as exc:
        raise ImportError(
            "neuropose.analyzer.features.find_peaks requires scipy. "
            "Install it with: pip install neuropose[analysis]"
        ) from exc
    indices, _properties = _sp_find_peaks(values, **kwargs)
    return indices
