"""Score parking-space candidates on the geometry a product actually needs.

Box mAP is the wrong selection signal for this product.  It answers "did a
detection land near a bay", not "does the proposed boundary follow the bay",
and it averages a camera the detector handles well together with one it cannot
see at all.  Every number here is computed from the predicted quadrilateral
against the annotated quadrilateral, and every number is reported split by the
two axes where the incumbent was measured to fail:

* **occupancy state** -- the shipped detector recalls occupied bays at 0.711
  and vacant bays at 0.433, because it learned to find vehicles rather than
  bays.  A parking product that cannot see empty bays is useless, so vacant
  recall is reported on its own rather than averaged away.
* **camera family** -- aggregate scores are carried by CNRPark+EXT, whose
  annotations are axis-aligned and easy; PKLot unseen recall sits near zero
  behind the same average.

Candidates of any Ultralytics task are accepted and converted to quadrilaterals
through one code path, so pose, segmentation, oriented box and plain detection
are compared on identical matching rules.

    python ml/evaluate_space_geometry.py --weights runs/.../best.pt --task pose
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_MANIFEST = Path(
    "D:/Projects/AI Based Smart Parking System Data/prepared/v2-protocol/source-manifest.jsonl"
)

# Match threshold at which a proposal is taken to have found a bay.  Reported at
# both the conventional 0.50 and the stricter 0.75, because 0.50 accepts a
# rectangle sitting askew over a trapezoid and 0.75 largely does not.
MATCH_IOU = (0.50, 0.75)


def camera_family(stem: str) -> str:
    if "GOPR" in stem:
        return "ACPDS"
    if stem.startswith("camera"):
        return "CNRPark+EXT"
    return "PKLot"


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_area = float(cv2.contourArea(first))
    second_area = float(cv2.contourArea(second))
    if first_area <= 0 or second_area <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = first_area + second_area - float(intersection)
    return float(intersection) / union if union > 0 else 0.0


def corner_error(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Mean corner displacement over the bay's own scale.

    Both quadrilaterals are already ordered clockwise, so only the starting
    corner can differ; the best cyclic rotation is taken so the number measures
    geometry rather than indexing.
    """
    scale = max(float(np.sqrt(abs(cv2.contourArea(truth)))), 1e-6)
    best = min(
        float(np.mean(np.linalg.norm(truth - np.roll(predicted, shift, axis=0), axis=1)))
        for shift in range(4)
    )
    return best / scale


def order_clockwise(points: np.ndarray) -> np.ndarray:
    """Clockwise from the top-left, matching how the labels are stored."""
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def mask_to_quad(mask: np.ndarray) -> np.ndarray | None:
    """Recover four corners from a predicted instance mask.

    A polygon simplification is tried first so a genuinely trapezoidal bay keeps
    its trapezoid; the minimum-area rectangle is the fallback when the contour
    will not reduce to four points, which is also the honest failure mode to
    report for this architecture.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None
    perimeter = cv2.arcLength(contour, True)
    for fraction in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        approximation = cv2.approxPolyDP(contour, fraction * perimeter, True)
        if len(approximation) == 4:
            return order_clockwise(approximation.reshape(4, 2).astype(np.float32))
    return order_clockwise(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32))


def predict_quads(
    model, task: str, image_path: Path, imgsz: int, conf: float, iou: float = 0.7
) -> list[np.ndarray]:
    """Run one candidate and return its bays as pixel-space quadrilaterals.

    ``iou`` is the suppression threshold, exposed because it is not neutral
    between the candidates: a pose head suppresses on *axis-aligned* boxes,
    and the axis-aligned box around a slanted bay overlaps its neighbours
    heavily, so the same threshold is far more aggressive there than for an
    oriented-box head suppressing on rotated overlap.
    """
    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
        device=getattr(model, "device", None),
    )[0]
    quads: list[np.ndarray] = []
    if task in {"pose", "keypointrcnn", "onnx", "onnx-segment"} and result.keypoints is not None:
        for points in result.keypoints.xy.cpu().numpy():
            if len(points) == 4:
                quads.append(order_clockwise(points.astype(np.float32)))
    elif task == "segment" and result.masks is not None:
        for mask in result.masks.data.cpu().numpy():
            resized = cv2.resize(
                mask, (result.orig_shape[1], result.orig_shape[0]), interpolation=cv2.INTER_NEAREST
            )
            quad = mask_to_quad(resized)
            if quad is not None:
                quads.append(quad)
    elif task == "obb" and result.obb is not None:
        for corners in result.obb.xyxyxyxy.cpu().numpy():
            quads.append(order_clockwise(corners.reshape(4, 2).astype(np.float32)))
    elif result.boxes is not None:
        for x1, y1, x2, y2 in result.boxes.xyxy.cpu().numpy():
            quads.append(
                np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            )
    return quads


class _RuntimeDetectorAdapter:
    """Scores the artifact the application actually loads.

    This exists because the other adapters score a training checkpoint, and the
    two are not the same thing.  Ultralytics feeds a ``.pt`` a letterboxed
    *rectangle* -- a 1280x720 frame becomes 1024x576 -- while the exported ONNX
    graph has a fixed square input, so the deployed model sees the frame padded
    to 1024x1024.  Measured on the top detection of one frame the two agree at
    0.92 polygon IoU, not 1.0: close, but a product figure should describe what
    ships rather than what it was exported from.
    """

    def __init__(self, detector) -> None:
        self.detector = detector
        self.device = "cpu"

    def predict(self, source, imgsz, conf, iou, verbose, device):
        from PIL import Image as PILImage

        with PILImage.open(source) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        polygons = [
            np.array(
                [[x * width, y * height] for x, y in item["polygon"]], dtype=np.float32
            )
            for item in self.detector.detect(image, confidence=conf)
        ]
        return [
            type(
                "Result",
                (),
                {"keypoints": type("K", (), {"xy": _Numpy(np.asarray(polygons))})()},
            )()
        ]


def _load_runtime_detector(model_root: Path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.ml.generalized_localizer import GeneralizedDetector

    return _RuntimeDetectorAdapter(GeneralizedDetector.load(model_root))


class _SegmentationOnnxAdapter:
    """Scores an exported segmentation graph the way the runtime would run it.

    The product decodes every detector through one path: a square letterbox, a
    fixed-shape graph, and suppression on polygon overlap.  Scoring the
    segmentation candidate through Ultralytics instead would compare it against
    the others under different preprocessing and different NMS, so the mask
    decode is reimplemented here against the same conventions.

    The decoder lives in the evaluator rather than in the application because no
    segmentation model has been selected: shipping a mask decode the product
    does not use would be speculative complexity.  If this candidate wins, the
    decode moves into ``app.ml.generalized_localizer`` with it.
    """

    def __init__(self, weights: Path) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
        import onnxruntime as ort
        from app.ml.registry import inference_providers

        self.session = ort.InferenceSession(str(weights), providers=inference_providers())
        self.device = "cpu"
        self.input_size = self.session.get_inputs()[0].shape[-1]
        if not isinstance(self.input_size, int):
            self.input_size = 768

    def predict(self, source, imgsz, conf, iou, verbose, device):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
        from app.ml.generalized_localizer import letterbox, polygon_nms
        from app.ml.geometry import order_polygon, validate_polygon
        from PIL import Image as PILImage

        with PILImage.open(source) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        canvas, scale, pad_x, pad_y = letterbox(image, self.input_size)
        batch = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: batch})

        # Ultralytics segmentation emits detections plus mask prototypes; the
        # prototype tensor is the four-dimensional one.
        detections = next(o for o in outputs if np.asarray(o).ndim == 3)
        prototypes = next(o for o in outputs if np.asarray(o).ndim == 4)
        predictions = np.asarray(detections, dtype=np.float32)[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose()
        protos = np.asarray(prototypes, dtype=np.float32)[0]
        channels, mask_h, mask_w = protos.shape

        scores = predictions[:, 4]
        keep = scores >= conf
        if not np.any(keep):
            return [_empty_result()]
        selected = predictions[keep]
        scores = scores[keep]
        coefficients = selected[:, -channels:]

        flat = protos.reshape(channels, -1)
        masks = 1.0 / (1.0 + np.exp(-(coefficients @ flat)))
        masks = masks.reshape(-1, mask_h, mask_w)

        quads: list[np.ndarray] = []
        for row, mask, score in zip(selected, masks, scores, strict=True):
            binary = (mask > 0.5).astype(np.uint8)
            if binary.sum() < 4:
                continue
            quad = mask_to_quad(binary)
            if quad is None:
                continue
            # Mask space -> canvas -> source image, then normalise.
            ratio = self.input_size / mask_w
            polygon = []
            for x, y in quad:
                source_x = (x * ratio - pad_x) / scale / width
                source_y = (y * ratio - pad_y) / scale / height
                polygon.append(
                    [min(max(float(source_x), 0.0), 1.0), min(max(float(source_y), 0.0), 1.0)]
                )
            polygon = order_polygon(polygon)
            if not validate_polygon(polygon).valid:
                continue
            quads.append(
                {"polygon": polygon, "confidence": float(score)}  # type: ignore[arg-type]
            )

        kept = polygon_nms(quads, iou, 400)  # type: ignore[arg-type]
        pixels = np.asarray(
            [
                [[x * width, y * height] for x, y in item["polygon"]]  # type: ignore[index]
                for item in kept
            ],
            dtype=np.float32,
        )
        return [
            type(
                "Result",
                (),
                {"keypoints": type("K", (), {"xy": _Numpy(pixels)})()},
            )()
        ]


def _empty_result():
    return type(
        "Result",
        (),
        {"keypoints": type("K", (), {"xy": _Numpy(np.zeros((0, 4, 2), np.float32))})()},
    )()


class _KeypointRCNNAdapter:
    """Presents a torchvision keypoint model through the same call the others use.

    The adapter exists so the scoring path stays literally identical across
    architectures; a separate evaluator for the non-YOLO candidate would make
    its numbers incomparable with everything else in the table.
    """

    def __init__(self, model, device: str) -> None:
        self.model = model
        self.device = device

    def predict(self, source, imgsz, conf, iou, verbose, device):
        import torch
        from PIL import Image as PILImage
        from torchvision.transforms import functional

        with PILImage.open(source) as handle:
            tensor = functional.to_tensor(handle.convert("RGB")).to(self.device)
        with torch.inference_mode():
            output = self.model([tensor])[0]
        keep = output["scores"] >= conf
        points = output["keypoints"][keep][:, :, :2].cpu().numpy()
        return [type("Result", (), {"keypoints": type("K", (), {"xy": _Numpy(points)})()})()]


class _Numpy:
    """Mimics the ``.cpu().numpy()`` chain the Ultralytics results expose."""

    def __init__(self, value) -> None:
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _load_keypoint_rcnn(weights: str, device: str, max_size: int | None = None):
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml.train_keypoint_rcnn import build_model

    model = build_model(3)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    if max_size is not None:
        # Match the effective scale the exported YOLO graphs receive.
        model.transform.max_size = max_size
        model.transform.min_size = (int(max_size * 0.5625),)
    model.eval().to(device)
    return _KeypointRCNNAdapter(model, device)


def score(
    model,
    task: str,
    rows: list[dict],
    image_root: Path,
    imgsz: int,
    conf: float,
    limit: int | None,
    iou: float = 0.7,
) -> dict[str, object]:
    tally: collections.Counter = collections.Counter()
    matched_iou: dict[str, list[float]] = collections.defaultdict(list)
    matched_error: dict[str, list[float]] = collections.defaultdict(list)
    latencies: list[float] = []
    images = 0

    # Filter to the rows this split actually holds *before* applying the limit.
    # Slicing the manifest first samples whatever the manifest happens to list
    # earliest, which for a limit smaller than the training block is a set of
    # images that appear in no evaluation split at all.
    present = [
        (row, image_root / f"{str(row['source_id']).replace('/', '__').replace(' ', '_')}.jpg")
        for row in rows
    ]
    present = [(row, path) for row, path in present if path.is_file()]
    if limit is not None and limit < len(present):
        # Evenly spaced indices across the whole split.  Taking every n-th and
        # then truncating drops the tail, and the manifest is ordered by dataset,
        # so the truncated tail is an entire camera family -- a subset that
        # silently excluded every ACPDS image while still reporting an "ALL" row.
        present = [
            present[round(index * (len(present) - 1) / (limit - 1))]
            for index in range(limit)
        ]

    for row, image_path in present:
        stem = image_path.stem
        with Image.open(image_path) as handle:
            width, height = handle.size
        family = camera_family(stem)

        started = time.perf_counter()
        predictions = predict_quads(model, task, image_path, imgsz, conf, iou)
        latencies.append((time.perf_counter() - started) * 1000)
        images += 1

        truths = []
        for slot in row["slots"]:
            occupied = slot.get("occupied")
            if occupied is None:
                continue
            quad = np.array(
                [[x * width, y * height] for x, y in slot["polygon"]], dtype=np.float32
            )
            if cv2.contourArea(quad) <= 0:
                quad = quad[::-1].copy()
            truths.append((quad, "occupied" if occupied else "vacant"))

        # Greedy one-to-one matching, best overlap first, so a single sprawling
        # proposal cannot claim credit for several neighbouring bays.
        pairs = []
        for truth_index, (truth, _) in enumerate(truths):
            for prediction_index, prediction in enumerate(predictions):
                overlap = polygon_iou(truth, prediction)
                if overlap > 0:
                    pairs.append((overlap, truth_index, prediction_index))
        pairs.sort(reverse=True)
        truth_taken: dict[int, float] = {}
        prediction_taken: set[int] = set()
        for overlap, truth_index, prediction_index in pairs:
            if truth_index in truth_taken or prediction_index in prediction_taken:
                continue
            truth_taken[truth_index] = overlap
            prediction_taken.add(prediction_index)
            truth, state = truths[truth_index]
            matched_iou[family].append(overlap)
            matched_iou["ALL"].append(overlap)
            matched_iou[f"state:{state}"].append(overlap)
            error = corner_error(order_clockwise(truth), predictions[prediction_index])
            matched_error[family].append(error)
            matched_error["ALL"].append(error)

        tally[(family, "predictions")] += len(predictions)
        tally[("ALL", "predictions")] += len(predictions)
        for truth_index, (_, state) in enumerate(truths):
            overlap = truth_taken.get(truth_index, 0.0)
            for key in (family, "ALL", f"state:{state}", f"{family}|{state}"):
                tally[(key, "truths")] += 1
                for threshold in MATCH_IOU:
                    if overlap >= threshold:
                        tally[(key, f"hit{int(threshold * 100)}")] += 1

    groups = sorted({key for key, _ in tally})
    report_rows = []
    for key in groups:
        truths_count = tally[(key, "truths")]
        if not truths_count:
            continue
        entry: dict[str, object] = {"group": key, "bays": truths_count}
        for threshold in MATCH_IOU:
            suffix = int(threshold * 100)
            entry[f"recall{suffix}"] = round(tally[(key, f"hit{suffix}")] / truths_count, 4)
        predictions_count = tally[(key, "predictions")]
        if predictions_count:
            entry["proposals"] = predictions_count
            entry["precision50"] = round(tally[(key, "hit50")] / predictions_count, 4)
        if matched_iou.get(key):
            entry["mean_matched_polygon_iou"] = round(statistics.fmean(matched_iou[key]), 4)
        if matched_error.get(key):
            entry["mean_corner_error"] = round(statistics.fmean(matched_error[key]), 4)
        report_rows.append(entry)

    return {
        "images": images,
        "latency_ms_mean": round(statistics.fmean(latencies), 2) if latencies else None,
        "latency_ms_median": round(statistics.median(latencies), 2) if latencies else None,
        "groups": report_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--task",
        choices=(
            "pose",
            "segment",
            "obb",
            "detect",
            "keypointrcnn",
            "onnx",
            "onnx-segment",
        ),
        required=True,
        help="the 'onnx' tasks score an exported graph through the deployed decode",
    )
    parser.add_argument("--images", type=Path, required=True, help="images/<split> directory")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    # Torchvision resizes to min 800 / max 1333 by default, so Keypoint
    # R-CNN sees roughly a third more pixels than the YOLO candidates get at
    # 1024.  Constraining it isolates how much of its geometry advantage is
    # architecture and how much is resolution.
    parser.add_argument("--rcnn-max-size", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    if arguments.task == "onnx":
        model = _load_runtime_detector(Path(arguments.weights))
    elif arguments.task == "onnx-segment":
        model = _SegmentationOnnxAdapter(Path(arguments.weights))
    elif arguments.task == "keypointrcnn":
        model = _load_keypoint_rcnn(
            arguments.weights, arguments.device, arguments.rcnn_max_size
        )
    else:
        from ultralytics import YOLO

        model = YOLO(arguments.weights)
        model.to(arguments.device)
    rows = [
        json.loads(line)
        for line in arguments.manifest.read_text(encoding="utf-8").splitlines()
    ]
    result = score(
        model,
        arguments.task,
        rows,
        arguments.images,
        arguments.imgsz,
        arguments.conf,
        arguments.limit,
        arguments.iou,
    )
    # Two perspectives, because the corpus supports two different questions.
    #
    # "Dataset geometry" asks how closely predictions match the annotations that
    # exist.  "Product geometry" asks how closely they follow the real bay
    # boundaries, and only ACPDS can answer it: its bays are drawn on the
    # painted markings (edge response 2.27x the surrounding ground), while
    # PKLot's are drawn around where vehicles stand (0.91x), so a PKLot score
    # penalises a model for predicting the boundary correctly.
    result["geometry_perspectives"] = {
        "dataset_geometry": {
            "group": "ALL",
            "meaning": "agreement with the annotations as they exist",
        },
        "product_geometry": {
            "group": "ACPDS",
            "meaning": "agreement with real bay boundaries, on the only "
            "geometry-faithful source in this corpus",
        },
    }
    result["candidate"] = arguments.label or Path(arguments.weights).stem
    result["task"] = arguments.task
    result["weights"] = str(arguments.weights)
    result["split"] = arguments.images.name
    result["device"] = arguments.device
    result["conf"] = arguments.conf
    result["nms_iou"] = arguments.iou
    result["rcnn_max_size"] = arguments.rcnn_max_size
    print(json.dumps(result, indent=2))
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
