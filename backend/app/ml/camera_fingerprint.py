"""Decide whether a new image comes from a camera with a verified layout.

A verified layout may only be reused when the next image really is the same
view.  Getting that wrong reinstates the defect this work exists to remove -- one
lot's geometry stamped onto another -- so the matcher is tuned for precision and
is allowed to say "not recognised" freely.  The cost of a miss is that the
operator verifies the layout again; the cost of a false match is a confident
wrong answer.

Two independent signals are measured, and a match needs only one of them:

* a perceptual hash of the whole frame, which is cheap and settles most cases;
* an ORB + RANSAC homography inlier ratio, which survives large appearance
  changes because it matches structure rather than pixels.

Both thresholds were chosen from measured distributions over the prepared
sources (``backend/app/cli/measure_fingerprint.py``).  On same-camera pairs
separated by months of weather and occupancy change, each signal alone
recognised roughly one pair in nine with **no** cross-camera false match at
these settings; taken together they recognise more while keeping that property.
Images captured under similar conditions match far more readily than that
worst-case figure suggests.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

import cv2
import imagehash
import numpy as np
from PIL import Image

# Perceptual-hash Hamming distance below which two frames are the same camera.
# Measured cross-camera minimum was 16 bits, so 12 keeps a clear margin.
MATCH_DISTANCE = 12

# Homography inlier ratio above which two frames are the same view.  Measured
# cross-camera maximum was 0.234, so 0.30 keeps a margin above every observed
# false pairing.
MATCH_INLIER_RATIO = 0.30

# Frames whose hashes are further apart than this are never structurally
# compared; it prunes the expensive path without discarding real matches.
STRUCTURE_PREFILTER_DISTANCE = 40

# Aspect ratios differing by more than this fraction are never the same camera.
MAX_ASPECT_DELTA = 0.12

# Working width for structural matching.  Large enough to keep parking-bay
# corners and markings, small enough to stay fast.
STRUCTURE_WIDTH = 900
_ORB_FEATURES = 1200


@dataclass(frozen=True)
class CameraSignature:
    """Everything needed to recognise this view again."""

    fingerprint: str
    width: int
    height: int
    descriptors: np.ndarray | None = None
    keypoints: tuple[tuple[float, float], ...] = ()

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)


def _structure(image: Image.Image) -> tuple[np.ndarray | None, tuple[tuple[float, float], ...]]:
    grey = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    scale = STRUCTURE_WIDTH / max(grey.shape[1], 1)
    if scale < 1.0:
        grey = cv2.resize(grey, (STRUCTURE_WIDTH, max(1, round(grey.shape[0] * scale))))
    detector = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    keypoints, descriptors = detector.detectAndCompute(grey, None)
    if descriptors is None or len(keypoints) < 20:
        return None, ()
    return descriptors, tuple((float(point.pt[0]), float(point.pt[1])) for point in keypoints)


def camera_signature(image: Image.Image, *, with_structure: bool = True) -> CameraSignature:
    """Fingerprint the scene behind an image."""
    rgb = image.convert("RGB")
    descriptors, keypoints = _structure(rgb) if with_structure else (None, ())
    return CameraSignature(
        fingerprint=str(imagehash.phash(rgb, hash_size=8)),
        width=rgb.width,
        height=rgb.height,
        descriptors=descriptors,
        keypoints=keypoints,
    )


def encode_structure(signature: CameraSignature) -> str:
    """Serialise the structural part of a signature for storage."""
    if signature.descriptors is None:
        return json.dumps({"descriptors": None, "keypoints": []})
    return json.dumps(
        {
            "descriptors": base64.b64encode(signature.descriptors.tobytes()).decode("ascii"),
            "shape": list(signature.descriptors.shape),
            "keypoints": [[round(x, 2), round(y, 2)] for x, y in signature.keypoints],
        }
    )


def decode_structure(
    payload: str | None, width: int, height: int, fingerprint: str
) -> CameraSignature | None:
    """Rebuild a stored signature; ``None`` when no structure was saved."""
    if not payload:
        return None
    try:
        data = json.loads(payload)
        if not data.get("descriptors"):
            return None
        raw = base64.b64decode(data["descriptors"])
        descriptors = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(data["shape"]))
        keypoints = tuple((float(x), float(y)) for x, y in data["keypoints"])
    except (ValueError, TypeError, KeyError):
        return None
    return CameraSignature(fingerprint, width, height, descriptors, keypoints)


def fingerprint_distance(first: str, second: str) -> int:
    """Hamming distance between two perceptual fingerprints, in bits."""
    try:
        return int(imagehash.hex_to_hash(first) - imagehash.hex_to_hash(second))
    except (ValueError, TypeError):
        return 64


def structural_agreement(first: CameraSignature, second: CameraSignature) -> float:
    """Homography inlier ratio between two views, 0 when they cannot be aligned."""
    if first.descriptors is None or second.descriptors is None:
        return 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
        first.descriptors, second.descriptors
    )
    if len(matches) < 15:
        return 0.0
    matches = sorted(matches, key=lambda match: match.distance)[:200]
    source = np.float32([first.keypoints[match.queryIdx] for match in matches])
    target = np.float32([second.keypoints[match.trainIdx] for match in matches])
    _, inliers = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    return float(np.mean(inliers)) if inliers is not None else 0.0


def compatible_aspect(candidate: CameraSignature, width: int, height: int) -> bool:
    """Whether two frames could come from the same camera at all."""
    stored_aspect = width / max(height, 1)
    return abs(candidate.aspect - stored_aspect) <= MAX_ASPECT_DELTA * max(stored_aspect, 1.0)


def same_camera(
    candidate: CameraSignature,
    stored_fingerprint: str,
    stored_width: int,
    stored_height: int,
    *,
    stored_signature: CameraSignature | None = None,
    max_distance: int = MATCH_DISTANCE,
) -> bool:
    """Whether a new image comes from a camera with this stored fingerprint.

    Aspect ratio is a hard gate.  After that, either signal alone is enough,
    because each was measured to produce no cross-camera false match on its own.
    """
    if not compatible_aspect(candidate, stored_width, stored_height):
        return False
    distance = fingerprint_distance(candidate.fingerprint, stored_fingerprint)
    if distance <= max_distance:
        return True
    if stored_signature is None or distance > STRUCTURE_PREFILTER_DISTANCE:
        return False
    return structural_agreement(candidate, stored_signature) >= MATCH_INLIER_RATIO
