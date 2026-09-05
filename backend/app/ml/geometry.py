from __future__ import annotations

from dataclasses import dataclass
from math import atan2

import cv2
import numpy as np
from PIL import Image, ImageDraw

RECTIFIED_PATCH_SIZE = 128
MIN_POLYGON_AREA = 1e-5


@dataclass(frozen=True)
class GeometryQuality:
    valid: bool
    area: float
    self_intersecting: bool
    reason: str | None = None


def polygon_area(polygon: list[list[float]]) -> float:
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = b - a
    second = c - a
    return float(first[0] * second[1] - first[1] * second[0])


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    return (_orientation(a, b, c) * _orientation(a, b, d) < 0) and (
        _orientation(c, d, a) * _orientation(c, d, b) < 0
    )


def polygon_self_intersects(polygon: list[list[float]]) -> bool:
    points = np.asarray(polygon, dtype=np.float64)
    count = len(points)
    if count < 4:
        return False
    for index in range(count):
        a, b = points[index], points[(index + 1) % count]
        for other in range(index + 1, count):
            if other in {index, (index + 1) % count}:
                continue
            if (other + 1) % count in {index, (index + 1) % count}:
                continue
            c, d = points[other], points[(other + 1) % count]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def order_polygon(polygon: list[list[float]]) -> list[list[float]]:
    """Return a stable clockwise polygon beginning near the top-left corner."""
    points = [[float(point[0]), float(point[1])] for point in polygon]
    if len(points) < 3:
        return points
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    ordered = sorted(points, key=lambda point: atan2(point[1] - center_y, point[0] - center_x))
    start = min(range(len(ordered)), key=lambda index: ordered[index][0] + ordered[index][1])
    ordered = ordered[start:] + ordered[:start]
    if polygon_area(ordered) == 0:
        return ordered
    return ordered


def validate_polygon(polygon: list[list[float]]) -> GeometryQuality:
    if len(polygon) < 3:
        return GeometryQuality(False, 0.0, False, "fewer_than_three_points")
    if any(len(point) != 2 for point in polygon):
        return GeometryQuality(False, 0.0, False, "invalid_point")
    if any(not np.isfinite(float(value)) for point in polygon for value in point):
        return GeometryQuality(False, 0.0, False, "non_finite_coordinate")
    if any(not 0.0 <= float(value) <= 1.0 for point in polygon for value in point):
        return GeometryQuality(False, 0.0, False, "coordinate_out_of_bounds")
    ordered = order_polygon(polygon)
    intersects = polygon_self_intersects(ordered)
    area = polygon_area(ordered)
    if intersects:
        return GeometryQuality(False, area, True, "self_intersection")
    if area < MIN_POLYGON_AREA:
        return GeometryQuality(False, area, False, "area_too_small")
    return GeometryQuality(True, area, False)


def _pixel_points(image: Image.Image, polygon: list[list[float]]) -> np.ndarray:
    width, height = image.size
    return np.asarray(
        [[float(x) * (width - 1), float(y) * (height - 1)] for x, y in order_polygon(polygon)],
        dtype=np.float32,
    )


def _quad_destination(points: np.ndarray, size: int) -> np.ndarray:
    edge_width = max(
        np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3]), 1.0
    )
    edge_height = max(
        np.linalg.norm(points[2] - points[1]), np.linalg.norm(points[3] - points[0]), 1.0
    )
    if edge_height > edge_width:
        width = max(24, int(round(size * edge_width / edge_height)))
        height = size
    else:
        width = size
        height = max(24, int(round(size * edge_height / edge_width)))
    left = (size - width) // 2
    top = (size - height) // 2
    return np.asarray(
        [
            [left, top],
            [left + width - 1, top],
            [left + width - 1, top + height - 1],
            [left, top + height - 1],
        ],
        dtype=np.float32,
    )


def rectify_slot(
    image: Image.Image,
    polygon: list[list[float]],
    *,
    size: int = RECTIFIED_PATCH_SIZE,
) -> Image.Image:
    """Perspective-rectify a normalized slot polygon and mask non-slot pixels."""
    quality = validate_polygon(polygon)
    if not quality.valid:
        raise ValueError(f"Invalid parking-space polygon: {quality.reason}")
    rgb = image.convert("RGB")
    points = _pixel_points(rgb, polygon)
    if len(points) == 4:
        destination = _quad_destination(points, size)
        matrix = cv2.getPerspectiveTransform(points, destination)
        pixels = np.asarray(rgb)
        warped = cv2.warpPerspective(
            pixels,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(127, 127, 127),
        )
        mask = np.zeros((rgb.height, rgb.width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
        warped_mask = cv2.warpPerspective(mask, matrix, (size, size), flags=cv2.INTER_NEAREST)
        warped[warped_mask == 0] = 127
        return Image.fromarray(warped, mode="RGB")

    left, top = np.floor(points.min(axis=0)).astype(int)
    right, bottom = np.ceil(points.max(axis=0)).astype(int)
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    crop = rgb.crop((left, top, right + 1, bottom + 1))
    translated = [(float(x - left), float(y - top)) for x, y in points]
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon(translated, fill=255)
    neutral = Image.new("RGB", crop.size, (127, 127, 127))
    return Image.composite(crop, neutral, mask).resize((size, size), Image.Resampling.BILINEAR)


def polygon_iou(first: list[list[float]], second: list[list[float]]) -> float:
    """Fast convex-polygon IoU for validation and proposal suppression."""
    first_points = np.asarray(order_polygon(first), dtype=np.float32)
    second_points = np.asarray(order_polygon(second), dtype=np.float32)
    first_area = float(cv2.contourArea(first_points))
    second_area = float(cv2.contourArea(second_points))
    if first_area <= 0 or second_area <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first_points, second_points)
    union = first_area + second_area - float(intersection)
    return float(intersection) / union if union > 0 else 0.0
