"""Profile the product pipeline stage by stage, on an otherwise idle machine.

An earlier attempt at this was run while GPU training was in progress and
reported space detection at 3,493 ms against 842 ms measured quietly -- a four-
fold error caused entirely by dataloader workers competing for the CPU.  This
script therefore refuses to run while a training process is alive, rather than
producing a number that looks like a measurement.

Two deployments are reported.  **CPU** is the real one: the application loads
ONNX Runtime, and the installed build exposes only the CPU execution provider,
so every millisecond the product actually spends is a CPU millisecond.  **GPU**
is measured through the PyTorch path the training environment uses, which is not
the deployed path -- it says what the hardware could do if a CUDA ONNX build
were installed, and is labelled as such rather than presented as product
latency.

    python ml/profile_pipeline.py --out <scratch>/latency-profile.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

DEFAULT_IMAGES = Path(
    "D:/Projects/AI Based Smart Parking System Data/prepared/obb-single/images/val_unseen"
)


def training_is_running() -> bool:
    """Whether anything that would distort a measurement is alive.

    The process query is restricted to ``python.exe``.  Without that it matches
    the PowerShell process running the query itself, because the query string
    contains the very script names it searches for -- which made this guard
    report a training run on a completely idle machine.
    """
    query = (
        "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { "
        "$_.CommandLine -like '*train_keypoint_rcnn.py*' -or "
        "$_.CommandLine -like '*train_space_model.py*' } "
        "| Measure-Object).Count"
    )
    try:
        result = subprocess.run(
            ["powershell", "-Command", query],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return int(result.stdout.strip() or "0") > 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def timed(function, *args, **kwargs) -> tuple[object, float]:
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-busy", action="store_true")
    arguments = parser.parse_args()

    if training_is_running() and not arguments.allow_busy:
        print("A training process is running; a profile taken now would be wrong.")
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.core.config import get_settings
    from app.ml.association import associate
    from app.ml.generalized_localizer import GeneralizedDetector
    from app.ml.geometry import rectify_slots
    from app.ml.occupancy_fusion import OccupancyEvidence, load_policy
    from app.ml.occupancy_v3 import OccupancyV3Predictor
    from app.ml.parking_area import classify_vehicle_positions, infer_parking_area
    from app.ml.vehicle_detector import VehicleDetector

    settings = get_settings()
    paths = sorted(arguments.images.glob("*.jpg"))
    if not paths:
        print(f"No images under {arguments.images}")
        return 1
    step = max(1, len(paths) // arguments.samples)
    paths = paths[::step][: arguments.samples]

    space = GeneralizedDetector.load(settings.model_root)
    vehicle = VehicleDetector.load(settings.model_root)
    occupancy = OccupancyV3Predictor.load(settings.model_root)
    policy = load_policy(settings.model_root)

    stages: dict[str, list[float]] = {
        "slot localization": [],
        "vehicle detection": [],
        "parking-area inference": [],
        "slot rectification": [],
        "occupancy classification": [],
        "association + fusion": [],
    }

    # One untimed pass, so a first-call cost inside a library is not charged to
    # the first image.
    space.detect(Image.open(paths[0]).convert("RGB"))

    for path in paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size

        slots, elapsed = timed(space.detect, image)
        stages["slot localization"].append(elapsed)
        polygons = [item["polygon"] for item in slots] or [
            [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]
        ]

        vehicles, elapsed = timed(vehicle.detect, image)
        stages["vehicle detection"].append(elapsed)
        boxes = [item.box for item in vehicles]

        area, elapsed = timed(infer_parking_area, polygons, boxes)
        stages["parking-area inference"].append(elapsed)

        patches, elapsed = timed(rectify_slots, image, polygons)
        stages["slot rectification"].append(elapsed)

        (probabilities, _), elapsed = timed(occupancy.probabilities, patches)
        stages["occupancy classification"].append(elapsed)

        started = time.perf_counter()
        association = associate(
            [[[x * width, y * height] for x, y in polygon] for polygon in polygons],
            [
                (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
                for box in boxes
            ],
        )
        mapped = {
            link.vehicle_index
            for index in range(len(polygons))
            for link in association.occupying_links(index)
        }
        classify_vehicle_positions(area, boxes, mapped)
        for index, probability in enumerate(probabilities):
            policy.fuse(
                OccupancyEvidence(
                    occupancy_probability=float(probability),
                    vehicle_evidence_available=True,
                    vehicle_detected=bool(association.occupying_links(index)),
                )
            )
        stages["association + fusion"].append((time.perf_counter() - started) * 1000)

    report: dict[str, object] = {
        "images": len(paths),
        "deployment": "CPU (ONNX Runtime; the installed build exposes no CUDA provider)",
        "stages": {
            name: {
                "median_ms": round(statistics.median(values), 1),
                "mean_ms": round(statistics.fmean(values), 1),
            }
            for name, values in stages.items()
        },
    }
    total = sum(statistics.median(values) for values in stages.values())
    report["cpu_total_ms"] = round(total, 1)

    print(f"{'stage':<28}{'median ms':>11}{'share':>8}")
    for name, values in stages.items():
        median = statistics.median(values)
        print(f"{name:<28}{median:>11.1f}{100 * median / total:>7.1f}%")
    print(f"{'TOTAL (CPU, per image)':<28}{total:>11.1f}")

    # GPU, through the training environment's PyTorch path.  Reported as a
    # hardware capability rather than as product latency, because the shipped
    # runtime cannot reach it.
    try:
        import torch
        from ultralytics import YOLO

        if torch.cuda.is_available():
            gpu: dict[str, float] = {}
            for label, weights in (
                (
                    "slot localization",
                    "runs/obb/ml/runs/generalized-localizer/yolo11n-obb-2/weights/best.pt",
                ),
                ("vehicle detection", "weights/yolo11n.pt"),
            ):
                if not Path(weights).is_file():
                    continue
                model = YOLO(weights)
                model.to("cuda")
                model.predict(str(paths[0]), imgsz=1024, verbose=False, device="cuda")
                timings = []
                for path in paths:
                    _, elapsed = timed(
                        model.predict,
                        str(path),
                        imgsz=1024,
                        verbose=False,
                        device="cuda",
                    )
                    timings.append(elapsed)
                gpu[label] = round(statistics.median(timings), 1)
            if gpu:
                report["gpu_pytorch_stages_ms"] = gpu
                report["gpu_total_ms"] = round(sum(gpu.values()), 1)
                report["gpu_note"] = (
                    "PyTorch on RTX 4070, not the deployed ONNX path; detectors only"
                )
                print()
                for label, value in gpu.items():
                    print(f"{label + ' (GPU, PyTorch)':<28}{value:>11.1f}")
    except ImportError:
        report["gpu_note"] = "torch is not installed in this environment"

    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
