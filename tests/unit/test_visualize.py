"""Tests for :mod:`neuropose.visualize`.

Smoke tests only: we verify that the visualize function runs end-to-end
against a synthetic video, writes PNG files, honours ``frame_indices``, and
does NOT mutate the caller's :class:`VideoPredictions`. Actual pixel-level
correctness is out of scope for unit tests.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from neuropose.estimator import Estimator, VideoDecodeError
from neuropose.io import VideoPredictions
from neuropose.visualize import visualize_predictions


@pytest.fixture
def predictions_for_synthetic(
    synthetic_video: Path,
    fake_metrabs_model,
) -> VideoPredictions:
    """Run the fake estimator over the synthetic video and return predictions."""
    estimator = Estimator(model=fake_metrabs_model)
    return estimator.process_video(synthetic_video).predictions


class TestVisualizePredictions:
    def test_writes_one_png_per_frame(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        predictions_for_synthetic: VideoPredictions,
    ) -> None:
        output_dir = tmp_path / "viz"
        written = visualize_predictions(synthetic_video, predictions_for_synthetic, output_dir)
        assert len(written) == 5
        for path in written:
            assert path.exists()
            assert path.suffix == ".png"
            assert path.stat().st_size > 0

    def test_frame_indices_limits_output(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        predictions_for_synthetic: VideoPredictions,
    ) -> None:
        output_dir = tmp_path / "viz"
        written = visualize_predictions(
            synthetic_video,
            predictions_for_synthetic,
            output_dir,
            frame_indices=[0, 2, 4],
        )
        assert len(written) == 3
        names = sorted(p.stem for p in written)
        assert names == ["frame_000000", "frame_000002", "frame_000004"]

    def test_out_of_range_indices_silently_skipped(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        predictions_for_synthetic: VideoPredictions,
    ) -> None:
        output_dir = tmp_path / "viz"
        written = visualize_predictions(
            synthetic_video,
            predictions_for_synthetic,
            output_dir,
            frame_indices=[0, 999, -1],
        )
        assert len(written) == 1
        assert written[0].stem == "frame_000000"

    def test_does_not_mutate_input_predictions(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        predictions_for_synthetic: VideoPredictions,
    ) -> None:
        """The aliasing bug from the previous prototype mutated poses3d in place.

        This test guards against its regression by deep-copying poses3d
        before visualization and comparing against the live value after.
        Without the deepcopy, any in-place mutation would be invisible
        because ``before`` and ``after`` would refer to the same list.
        """
        frame_name = predictions_for_synthetic.frame_names()[0]
        before = copy.deepcopy(predictions_for_synthetic[frame_name].poses3d)
        visualize_predictions(
            synthetic_video,
            predictions_for_synthetic,
            tmp_path / "viz",
            frame_indices=[0],
        )
        after = predictions_for_synthetic[frame_name].poses3d
        assert before == after

    def test_rejects_invalid_view(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        predictions_for_synthetic: VideoPredictions,
    ) -> None:
        with pytest.raises(ValueError, match="view must be one of"):
            visualize_predictions(
                synthetic_video,
                predictions_for_synthetic,
                tmp_path / "viz",
                view="orthographic",
            )

    def test_depth_view_runs(
        self,
        tmp_path: Path,
        synthetic_video: Path,
        predictions_for_synthetic: VideoPredictions,
    ) -> None:
        written = visualize_predictions(
            synthetic_video,
            predictions_for_synthetic,
            tmp_path / "viz",
            view="depth",
            frame_indices=[0],
        )
        assert len(written) == 1

    def test_missing_video_raises(
        self,
        tmp_path: Path,
        predictions_for_synthetic: VideoPredictions,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            visualize_predictions(
                tmp_path / "nope.avi",
                predictions_for_synthetic,
                tmp_path / "viz",
            )

    def test_unreadable_video_raises(
        self,
        tmp_path: Path,
        predictions_for_synthetic: VideoPredictions,
    ) -> None:
        fake_video = tmp_path / "fake.avi"
        fake_video.write_bytes(b"not a video")
        with pytest.raises(VideoDecodeError):
            visualize_predictions(
                fake_video,
                predictions_for_synthetic,
                tmp_path / "viz",
            )
