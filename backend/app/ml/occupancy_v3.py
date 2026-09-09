"""Serve the enhanced occupancy classifier from its exported ONNX artifact.

This module is on the request path, so it holds only what answering a request
needs: metadata validation, checksum verification, a cached inference session
and the decision rules around its output.  The PyTorch model definitions and
the ONNX export that produced the artifact live in
``app.ml.occupancy_v3_training``, which the FastAPI application never imports.

The separation is a performance boundary, not tidiness.  Torch and ONNX Runtime
each size a thread pool to the machine and then compete for it, and the module
that merely *defines* a torch model is enough to pull that competition into the
serving process.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.ml import registry
from app.ml.preprocessing import imagenet_batch
from app.ml.registry import inference_providers, session_options

OCCUPANCY_MODEL_NAME = "parking-occupancy-enhanced-v3"
OCCUPANCY_SCHEMA_VERSION = "3.0"
OCCUPANCY_ONNX = f"{OCCUPANCY_MODEL_NAME}.onnx"
OCCUPANCY_METADATA = f"{OCCUPANCY_MODEL_NAME}.json"
IMAGE_SIZE = 128


class OccupancyV3NotReadyError(RuntimeError):
    """Raised when the independently evaluated V3 model is unavailable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def occupancy_v3_metadata(model_root: Path) -> dict[str, object]:
    """Validate and return enhanced-model metadata without building a session."""
    model_path = model_root / OCCUPANCY_ONNX
    metadata_path = model_root / OCCUPANCY_METADATA
    if not model_path.is_file() or not metadata_path.is_file():
        raise OccupancyV3NotReadyError(
            "Enhanced occupancy model is not installed; "
            "the reproducible V2 baseline remains available"
        )
    try:
        metadata = registry.read_metadata(metadata_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise OccupancyV3NotReadyError("Enhanced model metadata is unreadable") from exc
    if metadata.get("schema_version") != OCCUPANCY_SCHEMA_VERSION:
        raise OccupancyV3NotReadyError("Enhanced model schema is unsupported")
    if not metadata.get("frozen") or "final_holdout" not in metadata:
        raise OccupancyV3NotReadyError("Enhanced model has not passed the frozen holdout gate")
    return metadata


def verify_checksum(path: Path, expected: str) -> None:
    """Checksum a model file once per on-disk version.

    Integrity is still enforced -- a file whose bytes change gets a new
    signature and is hashed again -- but an unchanged file is not re-read on
    every request.
    """

    def build() -> str:
        return sha256_file(path)

    digest = registry.cached(f"sha256::{path}", registry.file_signature(path), build)
    if digest != expected:
        raise OccupancyV3NotReadyError("Enhanced model checksum is invalid")


@dataclass
class OccupancyV3Predictor:
    session: ort.InferenceSession
    threshold: float
    temperature: float
    metadata: dict[str, object]

    @classmethod
    def load(cls, model_root: Path) -> OccupancyV3Predictor:
        """Return the shared predictor, building it once per model file version."""
        model_path = model_root / OCCUPANCY_ONNX
        metadata_path = model_root / OCCUPANCY_METADATA
        signature = registry.file_signature(model_path, metadata_path)
        cached_error = registry.cached_failure("occupancy_v3", signature)
        if cached_error is not None:
            raise cached_error

        def build() -> OccupancyV3Predictor:
            metadata = occupancy_v3_metadata(model_root)
            verify_checksum(model_path, str(metadata.get("onnx_sha256", "")))
            return cls(
                session=ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options(),
                    providers=inference_providers(),
                ),
                threshold=float(metadata["decision_threshold"]),
                temperature=max(float(metadata.get("temperature", 1.0)), 1e-3),
                metadata=metadata,
            )

        try:
            return registry.cached("occupancy_v3", signature, build)
        except OccupancyV3NotReadyError as exc:
            registry.remember_failure("occupancy_v3", signature, exc)
            raise

    def probabilities(self, images: list[Image.Image]) -> tuple[np.ndarray, float]:
        if not images:
            return np.empty(0, dtype=np.float32), 0.0
        batch = imagenet_batch(images, IMAGE_SIZE)
        started = perf_counter()
        logits = self.session.run(["logit"], {"image": batch})[0]
        elapsed = (perf_counter() - started) * 1_000
        logits = np.asarray(logits, dtype=np.float64).reshape(-1) / self.temperature
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        return probabilities.astype(np.float32), elapsed

    @property
    def uncertain_band(self) -> tuple[float, float] | None:
        """Probability range in which no vacant/occupied verdict is asserted.

        Selected on the validation split; absent means the model reports a
        plain binary decision.
        """
        band = self.metadata.get("uncertain_band")
        if not isinstance(band, dict):
            return None
        try:
            lower, upper = float(band["lower"]), float(band["upper"])
        except (KeyError, TypeError, ValueError):
            return None
        return (lower, upper) if 0.0 < lower < upper < 1.0 else None

    def state_for(self, probability: float) -> str:
        """Three-state verdict for one calibrated probability."""
        band = self.uncertain_band
        if band is not None and band[0] <= probability <= band[1]:
            return "uncertain"
        return "occupied" if probability >= self.threshold else "vacant"

    def predict(self, images: list[Image.Image]) -> tuple[list[dict[str, object]], float]:
        probabilities, elapsed = self.probabilities(images)
        results = []
        for probability in probabilities:
            value = float(probability)
            state = self.state_for(value)
            results.append(
                {
                    # Kept for every existing consumer: "uncertain" spaces are
                    # not counted as occupied.
                    "predicted_occupied": state == "occupied",
                    "occupancy_state": state,
                    "occupied_probability": round(value, 6),
                    "confidence": round(float(max(value, 1.0 - value)), 6),
                }
            )
        return results, elapsed


def occupancy_v3_status(model_root: Path) -> dict[str, object]:
    """Report enhanced-model readiness from metadata alone.

    This runs on every readiness and status request, so it must not construct
    an inference session; the checksum is verified through the cache.
    """
    try:
        metadata = occupancy_v3_metadata(model_root)
        verify_checksum(model_root / OCCUPANCY_ONNX, str(metadata.get("onnx_sha256", "")))
    except OccupancyV3NotReadyError as exc:
        return {"ready": False, "model_name": OCCUPANCY_MODEL_NAME, "reason": str(exc)}
    return {
        "ready": True,
        "model_name": OCCUPANCY_MODEL_NAME,
        "architecture": metadata.get("architecture"),
        "decision_threshold": float(metadata["decision_threshold"]),
        "onnx_sha256": metadata.get("onnx_sha256"),
        "final_holdout": metadata.get("final_holdout"),
        "calibration": metadata.get("calibration"),
        "uncertain_band": metadata.get("uncertain_band"),
    }
