"""Confirm vehicle detections across frames before a video trusts them.

A single frame gives no way to tell a real vehicle from a detector artefact.
Both are a box with a plausible score, and the discriminators one would reach
for do not separate them: measured over unseen cameras, detections that landed
in a bay the dataset labels *vacant* were no smaller than genuine ones (relative
footprint 1.70 against 1.01) and barely less confident (0.63 against 0.71).  So
no size, aspect or confidence rule is applied to a still image, because none is
supported by the evidence.

Video supplies the signal a still image cannot.  A parked vehicle is in the same
place in frame after frame; a spurious detection on a painted numeral, a drain
cover or a shadow appears and disappears.  Requiring a track to be seen in
several of the recent frames before it counts therefore removes exactly the
errors that a still image cannot rule out, and costs almost nothing on real
vehicles, which are seen in all of them.

The tracker is deliberately simple -- greedy overlap matching against a short
history -- because the camera is fixed and the sampling rate is low.  Anything
motion-based would be modelling movement this footage does not contain.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# Overlap at which a detection is taken to continue an existing track.  A parked
# vehicle barely moves between samples, so this can be strict.
TRACK_IOU = 0.45

# Frames of history kept, and how many of them a track must appear in before it
# is confirmed.  Two of three tolerates a single missed frame -- an occlusion by
# a passing vehicle, a gust of rain -- while still rejecting a one-frame
# artefact.
HISTORY = 3
REQUIRED_SIGHTINGS = 2


def _iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    overlap = max(right - left, 0.0) * max(bottom - top, 0.0)
    if overlap <= 0:
        return 0.0
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    second_area = max(second[2] - second[0], 0.0) * max(second[3] - second[1], 0.0)
    union = first_area + second_area - overlap
    return overlap / union if union > 0 else 0.0


@dataclass
class VehicleTracker:
    """Confirms detections that recur across recent frames of one fixed camera."""

    history: int = HISTORY
    required: int = REQUIRED_SIGHTINGS
    frames: deque[list[tuple[float, float, float, float]]] = field(default_factory=deque)

    def confirm(
        self, boxes: list[tuple[float, float, float, float]]
    ) -> list[bool]:
        """Which of this frame's detections are corroborated by recent frames.

        The current frame counts as one sighting, so with ``required`` of two a
        detection needs support from at least one earlier frame.  Until the
        history has filled, everything is accepted: refusing to report vehicles
        for the first frames of a clip would be a worse failure than admitting a
        few unconfirmed ones.
        """
        confirmed: list[bool] = []
        warming = len(self.frames) < self.required - 1
        for box in boxes:
            sightings = 1 + sum(
                1
                for previous in self.frames
                if any(_iou(box, earlier) >= TRACK_IOU for earlier in previous)
            )
            confirmed.append(warming or sightings >= self.required)

        self.frames.append(list(boxes))
        while len(self.frames) > self.history:
            self.frames.popleft()
        return confirmed
