from __future__ import annotations

import numpy as np
from PIL import Image

FEATURE_VERSION = "hog-color-v1"
PATCH_SIZE = 32
HOG_BINS = 9
HOG_CELL_SIZE = 8


def crop_slot(image: Image.Image, polygon: list[list[float]]) -> Image.Image:
    """Crop a normalized parking polygon's padded bounding box."""
    width, height = image.size
    xs = [min(1.0, max(0.0, float(point[0]))) * width for point in polygon]
    ys = [min(1.0, max(0.0, float(point[1]))) * height for point in polygon]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    padding_x = max(2.0, (right - left) * 0.08)
    padding_y = max(2.0, (bottom - top) * 0.08)
    box = (
        max(0, int(left - padding_x)),
        max(0, int(top - padding_y)),
        min(width, int(right + padding_x + 1)),
        min(height, int(bottom + padding_y + 1)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Parking-space polygon produces an empty crop")
    return (
        image.crop(box).convert("RGB").resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BILINEAR)
    )


def _hog_features(grayscale: np.ndarray) -> np.ndarray:
    gradient_y, gradient_x = np.gradient(grayscale)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = np.mod(np.arctan2(gradient_y, gradient_x), np.pi)
    bin_edges = np.linspace(0.0, np.pi, HOG_BINS + 1)
    cells: list[np.ndarray] = []
    for top in range(0, PATCH_SIZE, HOG_CELL_SIZE):
        for left in range(0, PATCH_SIZE, HOG_CELL_SIZE):
            angles = orientation[top : top + HOG_CELL_SIZE, left : left + HOG_CELL_SIZE]
            weights = magnitude[top : top + HOG_CELL_SIZE, left : left + HOG_CELL_SIZE]
            histogram, _ = np.histogram(angles, bins=bin_edges, weights=weights)
            histogram = histogram.astype(np.float32)
            histogram /= np.linalg.norm(histogram) + 1e-6
            cells.append(histogram)
    return np.concatenate(cells)


def extract_features(patch: Image.Image) -> np.ndarray:
    pixels = (
        np.asarray(
            patch.convert("RGB").resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )
    grayscale = pixels @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    hog = _hog_features(grayscale)
    color_histograms = []
    for channel in range(3):
        histogram, _ = np.histogram(pixels[:, :, channel], bins=8, range=(0.0, 1.0))
        color_histograms.append(histogram.astype(np.float32) / pixels[:, :, channel].size)
    color_statistics = np.concatenate([pixels.mean(axis=(0, 1)), pixels.std(axis=(0, 1))]).astype(
        np.float32
    )
    grayscale_thumbnail = (
        np.asarray(
            Image.fromarray(np.uint8(grayscale * 255.0)).resize((8, 8), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).reshape(-1)
        / 255.0
    )
    return np.concatenate([hog, *color_histograms, color_statistics, grayscale_thumbnail]).astype(
        np.float32
    )


def scenario_features(image: Image.Image, slots: list[dict[str, object]]) -> np.ndarray:
    return np.stack([extract_features(crop_slot(image, slot["polygon"])) for slot in slots])
