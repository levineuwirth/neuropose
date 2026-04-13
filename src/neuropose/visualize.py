"""Matplotlib-based visualization of NeuroPose predictions.

Separate module so that importing :mod:`neuropose.estimator` does not pull
in matplotlib or incur its global backend side effect. Callers that want
visualization import this module explicitly.

The old prototype's ``_visualize`` helper had an in-place numpy-view
aliasing bug where ``poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2],
-poses3d[..., 1]`` mutated the caller's prediction array. This rewrite
makes an explicit copy before any axis reordering, so the input
:class:`VideoPredictions` is never touched.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from neuropose.estimator import VideoDecodeError
from neuropose.io import VideoPredictions

logger = logging.getLogger(__name__)

VALID_VIEWS = frozenset({"normal", "depth"})


def visualize_predictions(
    video_path: Path,
    predictions: VideoPredictions,
    output_dir: Path,
    *,
    view: str = "normal",
    joint_edges: Sequence[tuple[int, int]] | None = None,
    frame_indices: Sequence[int] | None = None,
) -> list[Path]:
    """Render per-frame 2D + 3D visualizations as PNG files.

    Parameters
    ----------
    video_path
        Path to the source video, used to recover frame pixels for the 2D
        overlay. Must be the same video the predictions were computed from.
    predictions
        Predictions to overlay. The dict ordering of ``predictions.frames``
        determines which frames are drawn unless ``frame_indices`` is set.
    output_dir
        Directory to write the rendered PNGs into. Created if absent.
    view
        ``"normal"`` (default) or ``"depth"``. Controls the 3D subplot's
        axis limits. Anything else raises ``ValueError``.
    joint_edges
        Optional list of ``(i, j)`` index pairs specifying skeleton edges to
        draw as lines connecting joints. If ``None``, only scatter points
        are drawn. For ``berkeley_mhad_43`` the edges can be obtained from
        ``model.per_skeleton_joint_edges[skeleton]``.
    frame_indices
        Optional subset of frame indices to render. If ``None``, every
        frame in ``predictions`` is rendered. Out-of-range indices are
        silently skipped.

    Returns
    -------
    list[Path]
        Paths of the written PNG files, in render order.

    Raises
    ------
    FileNotFoundError
        If ``video_path`` does not exist.
    VideoDecodeError
        If OpenCV cannot open the video.
    ValueError
        If ``view`` is not one of the supported values.
    """
    if view not in VALID_VIEWS:
        raise ValueError(f"view must be one of {sorted(VALID_VIEWS)}; got {view!r}")
    if not video_path.exists():
        raise FileNotFoundError(f"video file not found: {video_path}")

    # Late imports: matplotlib carries a global backend side effect, which
    # we want to avoid at module load time. pyplot is also slow to import.
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_names = predictions.frame_names()
    selected = _select_indices(frame_indices, len(frame_names))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoDecodeError(f"OpenCV could not open video: {video_path}")

    written: list[Path] = []
    try:
        next_index_to_render = iter(selected)
        target = next(next_index_to_render, None)
        frame_index = 0
        while target is not None:
            ok, bgr_frame = cap.read()
            if not ok:
                break
            if frame_index == target:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                frame_name = frame_names[frame_index]
                out_path = output_dir / f"{frame_name}.png"
                _render_frame(
                    rgb_frame,
                    predictions[frame_name],
                    out_path,
                    view=view,
                    joint_edges=joint_edges,
                    plt=plt,
                    Rectangle=Rectangle,
                )
                written.append(out_path)
                target = next(next_index_to_render, None)
            frame_index += 1
    finally:
        cap.release()

    logger.info("Wrote %d visualization frame(s) to %s", len(written), output_dir)
    return written


def _select_indices(frame_indices: Sequence[int] | None, total: int) -> list[int]:
    """Normalise the ``frame_indices`` argument to a sorted, in-range list."""
    if frame_indices is None:
        return list(range(total))
    return sorted({i for i in frame_indices if 0 <= i < total})


def _render_frame(
    rgb_frame: Any,
    frame_prediction: Any,
    out_path: Path,
    *,
    view: str,
    joint_edges: Sequence[tuple[int, int]] | None,
    plt: Any,
    Rectangle: Any,
) -> None:
    """Render one frame's 2D overlay + 3D scatter to ``out_path``."""
    # Explicit copies. The previous prototype mutated the caller's data via
    # numpy-view tuple-assignment; we take a fresh numpy array per person
    # so the caller's VideoPredictions is never touched.
    boxes = np.asarray(frame_prediction.boxes, dtype=float)
    poses3d = np.asarray(frame_prediction.poses3d, dtype=float).copy()
    poses2d = np.asarray(frame_prediction.poses2d, dtype=float)

    # Rotate for visualization: swap Y and Z so the ground plane is horizontal.
    # Do this on the copy so the original predictions object is untouched.
    if poses3d.size > 0:
        original_y = poses3d[..., 1].copy()
        poses3d[..., 1] = poses3d[..., 2]
        poses3d[..., 2] = -original_y

    fig = plt.figure(figsize=(10, 5.2))

    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(rgb_frame)
    for box in boxes:
        if len(box) < 4:
            continue
        x, y, w, h = box[:4]
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    pose_ax = fig.add_subplot(1, 2, 2, projection="3d")
    pose_ax.view_init(5, -85)
    if view == "depth":
        pose_ax.set_xlim3d(200, 17500)
        pose_ax.set_zlim3d(-1500, 1500)
        pose_ax.set_ylim3d(0, 3000)
    else:
        pose_ax.set_xlim3d(-1500, 1500)
        pose_ax.set_zlim3d(-1500, 1500)
        pose_ax.set_ylim3d(0, 3000)
    pose_ax.set_box_aspect((1, 1, 1))

    for pose3d, pose2d in zip(poses3d, poses2d, strict=False):
        if joint_edges is not None:
            for i_start, i_end in joint_edges:
                if 0 <= i_start < len(pose2d) and 0 <= i_end < len(pose2d):
                    image_ax.plot(
                        [pose2d[i_start][0], pose2d[i_end][0]],
                        [pose2d[i_start][1], pose2d[i_end][1]],
                        marker="o",
                        markersize=2,
                    )
                if 0 <= i_start < len(pose3d) and 0 <= i_end < len(pose3d):
                    pose_ax.plot(
                        [pose3d[i_start][0], pose3d[i_end][0]],
                        [pose3d[i_start][1], pose3d[i_end][1]],
                        [pose3d[i_start][2], pose3d[i_end][2]],
                        marker="o",
                        markersize=2,
                    )
        image_ax.scatter(pose2d[:, 0], pose2d[:, 1], s=2)
        pose_ax.scatter(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], s=2)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
