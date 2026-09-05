from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEMPORAL_CONFIG = "parking-temporal-v1.json"


@dataclass(frozen=True)
class TemporalParameters:
    median_window: int = 3
    ema_alpha: float = 0.4
    hysteresis: float = 0.08
    persistence: int = 2


def load_temporal_parameters(model_root: Path) -> TemporalParameters:
    path = model_root / TEMPORAL_CONFIG
    if not path.is_file():
        return TemporalParameters()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0" or not payload.get("validation_locked"):
        raise RuntimeError("Temporal configuration is not validation-locked")
    selected = payload["selected"]
    return TemporalParameters(
        median_window=int(selected["median_window"]),
        ema_alpha=float(selected["ema_alpha"]),
        hysteresis=float(selected["hysteresis"]),
        persistence=int(selected["persistence"]),
    )


class TemporalState:
    def __init__(self, slot_count: int, threshold: float, parameters: TemporalParameters) -> None:
        self.threshold = threshold
        self.parameters = parameters
        self.windows = [deque(maxlen=parameters.median_window) for _ in range(slot_count)]
        self.ema = np.full(slot_count, 0.5)
        self.states = np.zeros(slot_count, dtype=bool)
        self.pending = np.zeros(slot_count, dtype=np.int8)
        self.initialized = False

    def update(
        self, probabilities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, bool, bool]]]:
        medians = []
        for window, probability in zip(self.windows, probabilities, strict=True):
            window.append(float(probability))
            medians.append(float(np.median(window)))
        if not self.initialized:
            self.ema = np.asarray(medians)
            self.states = self.ema >= self.threshold
            self.initialized = True
            return self.ema.copy(), self.states.copy(), []
        alpha = self.parameters.ema_alpha
        self.ema = (1 - alpha) * self.ema + alpha * np.asarray(medians)
        events = []
        for index, probability in enumerate(self.ema):
            desired = self.states[index]
            if (
                not self.states[index]
                and probability >= self.threshold + self.parameters.hysteresis
            ):
                desired = True
            elif self.states[index] and probability <= self.threshold - self.parameters.hysteresis:
                desired = False
            if desired != self.states[index]:
                self.pending[index] += 1
                if self.pending[index] >= self.parameters.persistence:
                    previous = bool(self.states[index])
                    self.states[index] = desired
                    self.pending[index] = 0
                    events.append((index, previous, bool(desired)))
            else:
                self.pending[index] = 0
        return self.ema.copy(), self.states.copy(), events
