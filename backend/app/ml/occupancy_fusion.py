"""Combine the independent occupancy signals into one calibrated verdict.

Each signal fails in a different way, which is the whole reason to keep more
than one.  The crop classifier is strong but sees only a rectangle of pixels, so
a shadow, a puddle or a bay it was never trained on can fool it.  The vehicle
detector sees the whole frame and is unfooled by those, but it misses small,
distant and heavily occluded vehicles.  Geometry confidence says how much either
of them should be trusted, because a badly placed bay polygon feeds a bad crop
to the classifier and a bad footprint to the association.

The combination is a logistic model over log-odds rather than a rule ladder.
That matters for one specific reason: rules make evidence *decisive*, and no
signal here deserves that.  A missing vehicle detection must lower confidence in
occupancy, never veto it, because a motorcycle behind a van is invisible to the
detector and obvious to the classifier.  Adding log-odds does exactly that and
nothing more.

The coefficients are fitted on development data by
``ml/fit_occupancy_fusion.py`` and shipped as an artifact.  The defaults below
reduce the model to the classifier alone, so a system with no fitted artifact
and no vehicle detector behaves precisely as it did before fusion existed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from app.ml import registry

FUSION_ARTIFACT = "occupancy-fusion-v3.json"
FUSION_SCHEMA = "1.0"

VACANT = "vacant"
OCCUPIED = "occupied"
UNCERTAIN = "uncertain"

# Probability clamp, so a saturated input cannot produce an infinite log-odds.
_EPSILON = 1e-6


def logit(probability: float) -> float:
    clamped = min(max(float(probability), _EPSILON), 1.0 - _EPSILON)
    return math.log(clamped / (1.0 - clamped))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 40.0), -40.0)))


@dataclass(frozen=True)
class OccupancyEvidence:
    """Everything known about one bay at one moment.

    ``vehicle_detected`` is deliberately separate from
    ``vehicle_evidence_available``.  "No vehicle was found" and "no detector
    ran" are different states, and treating the second as the first would
    silently push every bay towards vacant on a machine where the detector
    failed to load.
    """

    occupancy_probability: float
    geometry_confidence: float = 1.0
    vehicle_evidence_available: bool = False
    vehicle_detected: bool = False
    vehicle_confidence: float = 0.0
    slot_coverage: float = 0.0
    vehicle_coverage: float = 0.0
    prior_probability: float | None = None


@dataclass(frozen=True)
class FusedOccupancy:
    state: str
    probability: float
    confidence: float
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def occupied(self) -> bool:
        """Only a settled occupied verdict counts as occupied."""
        return self.state == OCCUPIED

    def as_dict(self) -> dict[str, object]:
        return {
            "occupancy_state": self.state,
            "predicted_occupied": self.occupied,
            "occupied_probability": round(self.probability, 6),
            "confidence": round(self.confidence, 6),
            "evidence": {key: round(value, 4) for key, value in self.contributions.items()},
        }


# Reduces to "believe the classifier": the fused probability equals the
# classifier probability until a fitted artifact says otherwise.
DEFAULT_COEFFICIENTS: dict[str, float] = {
    "intercept": 0.0,
    "occupancy_logit": 1.0,
    "vehicle_present": 0.0,
    "vehicle_absent": 0.0,
    "slot_coverage": 0.0,
    "vehicle_coverage": 0.0,
    "geometry_confidence": 0.0,
    "prior_logit": 0.0,
}


@dataclass(frozen=True)
class FusionPolicy:
    coefficients: dict[str, float]
    lower_threshold: float
    upper_threshold: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        """Whether real coefficients were loaded rather than the fallback."""
        return bool(self.metadata.get("fitted", False))

    def fuse(self, evidence: OccupancyEvidence) -> FusedOccupancy:
        """Turn one bay's evidence into a three-state verdict."""
        weights = self.coefficients
        contributions: dict[str, float] = {
            "classifier": weights["occupancy_logit"] * logit(evidence.occupancy_probability)
        }

        if evidence.vehicle_evidence_available:
            if evidence.vehicle_detected:
                contributions["vehicle"] = weights["vehicle_present"] * evidence.vehicle_confidence
                contributions["coverage"] = (
                    weights["slot_coverage"] * evidence.slot_coverage
                    + weights["vehicle_coverage"] * evidence.vehicle_coverage
                )
            else:
                # Absence is weak evidence of vacancy, never a veto: a small or
                # occluded vehicle is exactly what this detector misses.
                contributions["vehicle"] = weights["vehicle_absent"]

        # Centred on 1.0 so a perfectly trusted polygon contributes nothing and
        # only doubtful geometry moves the verdict.
        contributions["geometry"] = weights["geometry_confidence"] * (
            evidence.geometry_confidence - 1.0
        )

        if evidence.prior_probability is not None:
            contributions["prior"] = weights["prior_logit"] * logit(evidence.prior_probability)

        total = weights["intercept"] + sum(contributions.values())
        probability = sigmoid(total)
        return FusedOccupancy(
            state=self.state_for(probability),
            probability=probability,
            confidence=max(probability, 1.0 - probability),
            contributions=contributions,
        )

    def state_for(self, probability: float) -> str:
        if probability >= self.upper_threshold:
            return OCCUPIED
        if probability <= self.lower_threshold:
            return VACANT
        return UNCERTAIN


def default_policy(lower: float = 0.35, upper: float = 0.65) -> FusionPolicy:
    """Classifier-only policy, used when no fitted artifact is installed."""
    return FusionPolicy(
        coefficients=dict(DEFAULT_COEFFICIENTS),
        lower_threshold=lower,
        upper_threshold=upper,
        metadata={"fitted": False, "reason": "no fusion artifact installed"},
    )


def load_policy(model_root: Path) -> FusionPolicy:
    """Load the fitted fusion policy, falling back to classifier-only.

    A missing or unreadable artifact is not an error: the system degrades to its
    pre-fusion behaviour rather than refusing to report occupancy.
    """
    artifact = model_root / FUSION_ARTIFACT
    signature = registry.file_signature(artifact)

    def build() -> FusionPolicy:
        if not artifact.is_file():
            return default_policy()
        try:
            payload = registry.read_metadata(artifact)
        except (OSError, json.JSONDecodeError):
            return default_policy()
        if payload.get("schema_version") != FUSION_SCHEMA:
            return default_policy()
        raw = payload.get("coefficients")
        if not isinstance(raw, dict):
            return default_policy()
        coefficients = dict(DEFAULT_COEFFICIENTS)
        for key in coefficients:
            if key in raw:
                coefficients[key] = float(raw[key])
        band = payload.get("uncertain_band", {})
        lower = float(band.get("lower", 0.35)) if isinstance(band, dict) else 0.35
        upper = float(band.get("upper", 0.65)) if isinstance(band, dict) else 0.65
        if not 0.0 < lower < upper < 1.0:
            lower, upper = 0.35, 0.65
        return FusionPolicy(
            coefficients=coefficients,
            lower_threshold=lower,
            upper_threshold=upper,
            metadata={
                "fitted": True,
                "fitted_on": payload.get("fitted_on"),
                "development": payload.get("development"),
            },
        )

    return registry.cached("occupancy_fusion", signature, build)
