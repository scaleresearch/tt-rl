"""What a recording looks like: one native frame and a readout beside it.

Split from `ttrl.train.record`, which owns the checkpoint, the environment and
the rollout, as `ttrl.train.video` owns the container and the codec. This
module owns the picture and knows nothing about either.

The readout sits beside the playfield rather than over it, so nothing it draws
can cover the ball, and every element has a fixed position so the layout does
not jitter.

The action block is the one part of the panel that an action space changes. A
discrete policy has a set to spread its choices over, so the bars are the
share of the last `history` steps each action took; a torque vector has no
such set, and `ContinuousFilm` draws one signed bar per actuator instead.
Everything else, including every colour and coordinate, is shared.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

__all__ = ["Film", "ContinuousFilm"]


class Film:
    """Composites one native game frame and a side panel into a video frame.

    Magnification is nearest-neighbour: the source is 160x210 pixel art that a
    smooth filter turns to mush. It magnifies towards `TARGET_HEIGHT` rather
    than by a fixed factor, because the panel's coordinates are laid out
    against a canvas of about that height, and a source that is already large
    (a rendered 3-D scene rather than a sprite) would otherwise push the
    readout into a corner of a four-megapixel frame.
    """

    SCALE, PANEL, MARGIN, VIEW, ROW = 4, 340, 28, 2, 22
    #: The canvas height the panel was laid out against: an Atari frame, whose
    #: 210 rows at SCALE is the magnification the look was approved at.
    TARGET_HEIGHT = 210 * SCALE
    BG, RULE, DIM = (18, 18, 20), (58, 58, 64), (128, 128, 138)
    TEXT, ACCENT = (236, 236, 240), (120, 215, 255)

    def __init__(self, *, title: str, subtitle: str = "", history: int = 60,
                 action_names: Sequence[str] | None = None) -> None:
        try:
            # Optional dependency, imported on first use; the refusal below names the extra.
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "drawing the panel needs OpenCV, which is not installed "
                f"({exc.name}): pip install opencv-python-headless (the "
                "'atari' and 'mujoco' extras already carry it)."
            ) from exc

        self._cv2 = cv2
        self.title, self.subtitle, self.history = title, subtitle, history
        self._names = None if action_names is None else list(action_names)
        self._recent: deque[int] = deque(maxlen=history)

    def label(self, action) -> str:
        named = self._names is not None and 0 <= action < len(self._names)
        return self._names[action] if named else str(action)

    def scale(self, height: int) -> int:
        return max(1, round(self.TARGET_HEIGHT / height))

    def _observe(self, action) -> None:
        self._recent.append(int(action))

    def _heading(self) -> str:
        return f"ACTION MIX, LAST {self.history}"

    def frame(self, rgb: np.ndarray, *, ret: float, step: int, action,
              n_actions: int, agent_view: np.ndarray | None = None) -> np.ndarray:
        cv2 = self._cv2
        self._observe(action)
        scale = self.scale(rgb.shape[0])
        game = cv2.resize(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST)
        height, width = game.shape[:2]
        canvas = np.empty((height, width + self.PANEL, 3), dtype=np.uint8)
        canvas[:, :width], canvas[:, width:] = game, self.BG
        canvas[:, width:width + 1] = self.RULE
        left = width + self.MARGIN
        for text, y, size, colour, weight in (
            (self.title, 48, 0.62, self.TEXT, 1),
            (self.subtitle, 76, 0.44, self.DIM, 1),
            ("RETURN", 156, 0.45, self.DIM, 1),
            (f"{ret:+.0f}", 214, 1.7, self.TEXT, 2),
            (f"STEP {step:>5d}", 252, 0.5, self.DIM, 1),
            ("ACTION", 312, 0.45, self.DIM, 1),
            (self.label(action), 346, 0.72, self.ACCENT, 1),
            (self._heading(), 454, 0.4, self.DIM, 1),
        ):
            if text:
                self._text(canvas, text, (left, y), size, colour, weight)
        top = height - self.MARGIN - 84 * self.VIEW
        self._bars(canvas, left, 470, min(n_actions, (top - 510) // self.ROW))
        if agent_view is not None:
            self._view(canvas, left, top, agent_view)
        return canvas

    def _text(self, canvas, text, origin, size, colour, weight) -> None:
        cv2 = self._cv2
        font = cv2.FONT_HERSHEY_DUPLEX if size > 0.6 else cv2.FONT_HERSHEY_SIMPLEX
        for shade, thick in ((0, 0, 0), weight + 2), (colour, weight):
            cv2.putText(canvas, text, origin, font, size, shade, thick, cv2.LINE_AA)

    def _row(self, canvas, left, y, label: str):
        """One labelled bar's track, returning where the bar may be drawn."""
        bar, span = left + 104, self.PANEL - 2 * self.MARGIN - 104
        self._text(canvas, label[:9], (left, y + 10), 0.36, self.DIM, 1)
        canvas[y:y + 12, bar:bar + span] = (36, 36, 42)
        return bar, span

    def _bars(self, canvas, left, top, rows) -> None:
        """How the last `history` actions were spread over the action set."""
        for index in range(max(0, rows)):
            share = sum(a == index for a in self._recent) / max(1, len(self._recent))
            y = top + index * self.ROW
            bar, span = self._row(canvas, left, y, self.label(index))
            canvas[y:y + 12, bar:bar + int(round(share * span))] = self.ACCENT

    def _view(self, canvas, left, top, agent_view) -> None:
        """The policy's own 84x84 input, so the clip shows what it sees."""
        side = 84 * self.VIEW
        grey = self._cv2.resize(agent_view, (side, side),
                                interpolation=self._cv2.INTER_NEAREST)
        self._text(canvas, "WHAT THE AGENT SEES", (left, top - 14), 0.4, self.DIM, 1)
        canvas[top - 1:top + side + 1, left - 1:left + side + 1] = self.RULE
        canvas[top:top + side, left:left + side] = grey[:, :, None]


class ContinuousFilm(Film):
    """The same panel, with the action block reading a torque vector.

    A continuous action has no set to take a histogram over, so the bars are
    the current action itself: one row per actuator, drawn from the centre of
    its track so a negative torque reads as one. The single-line readout is
    the action's norm, which is the one summary whose width does not grow with
    the number of actuators.
    """

    def __init__(self, *, bounds, **kwargs) -> None:
        super().__init__(**kwargs)
        self._bounds = np.asarray(bounds, dtype=np.float64).reshape(-1)
        if self._bounds.size < 1 or not np.all(self._bounds > 0.0):
            raise ValueError(
                f"ContinuousFilm draws each actuator against its own bound, "
                f"so it needs one positive bound per actuator, got "
                f"{bounds!r}")
        self._action = np.zeros_like(self._bounds)

    def _observe(self, action) -> None:
        self._action = np.asarray(action, dtype=np.float64).reshape(-1)
        if self._action.shape != self._bounds.shape:
            raise ValueError(
                f"ContinuousFilm was built for {self._bounds.shape[0]} "
                f"actuators and given an action of shape {self._action.shape}")

    def _heading(self) -> str:
        return "ACTION, PER ACTUATOR"

    def label(self, action) -> str:
        return f"|a| {np.linalg.norm(np.asarray(action, dtype=np.float64)):.2f}"

    def _bars(self, canvas, left, top, rows) -> None:
        for index in range(max(0, min(rows, self._action.shape[0]))):
            y = top + index * self.ROW
            bar, span = self._row(canvas, left, y, str(index))
            # A value outside the actuator's bound is clamped by the track
            # rather than drawn past its end.
            middle = bar + span // 2
            value = self._action[index]
            reach = int(round(min(1.0, abs(value) / self._bounds[index])
                              * (span // 2)))
            lo = middle if value >= 0.0 else middle - reach
            canvas[y:y + 12, lo:lo + max(1, reach)] = self.ACCENT
            canvas[y:y + 12, middle:middle + 1] = self.DIM
