"""Frames to an mp4.

Split from `ttrl.train.record`, which owns the eval protocol, the checkpoint
and the panel: this owns the container and the codec and knows nothing about
what is in the picture.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Mp4Writer"]


class Mp4Writer:
    """Frames to an mp4, opened on the first frame so it carries its size."""

    def __init__(self, path: str, *, fps: float) -> None:
        try:
            # Optional dependency, imported on first use; the refusal below names the extra.
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "recording a video needs OpenCV, which is not installed "
                f"({exc.name}): pip install opencv-python-headless (the "
                "'atari' extra already carries it)."
            ) from exc
        self._cv2, self._path, self._fps = cv2, path, float(fps)
        self._writer = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = self._cv2.VideoWriter(
                self._path, self._cv2.VideoWriter_fourcc(*"mp4v"), self._fps,
                (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"could not open {self._path} for writing: the directory "
                    f"has to exist and the name has to end in .mp4.")
        self._writer.write(frame[:, :, ::-1])  # cv2 wants BGR

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
