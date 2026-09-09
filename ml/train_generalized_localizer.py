"""Train and evaluate generalized parking-space detectors.

Runs in the separate CUDA training environment, not the application runtime, so
installing a GPU build of PyTorch cannot destabilise serving.  The application
keeps its CPU-only ONNX path.

Selection rule: candidates are compared on ``val_unseen`` -- cameras that never
appear in training.  ``val_known`` is reported alongside it so the generalisation
gap is visible rather than averaged away.  ``test_unseen`` is not read here.

Usage (from the training environment):
    python ml/train_generalized_localizer.py --model yolo11n-obb.pt --epochs 80
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_DATA = Path("D:/Projects/AI Based Smart Parking System Data/prepared/obb-single")


def evaluate(model, data_yaml: Path, split_name: str, imgsz: int) -> dict[str, float]:
    """Validate one weights file against one split descriptor."""
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, split="val", verbose=False)
    box = metrics.box
    return {
        "split": split_name,
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
        "map50": round(float(box.map50), 6),
        "map50_95": round(float(box.map), 6),
        "f1_50": round(
            float(2 * box.mp * box.mr / (box.mp + box.mr)) if (box.mp + box.mr) else 0.0, 6
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n-obb.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=20)
    # Windows spawns a full process per worker; 4 keeps host memory bounded
    # while still keeping the GPU fed.
    parser.add_argument("--workers", type=int, default=4)
    # Unseen-camera accuracy peaked at epoch 3 of the first run while training
    # loss kept falling, so the backbone can be frozen to keep generic features
    # and fit only the detection head.
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.01)
    arguments = parser.parse_args()

    from ultralytics import YOLO

    run_name = arguments.name or Path(arguments.model).stem
    project = Path("ml/runs/generalized-localizer")

    model = YOLO(arguments.model)
    started = time.perf_counter()
    model.train(
        data=str(arguments.data / "data.yaml"),
        epochs=arguments.epochs,
        imgsz=arguments.imgsz,
        batch=arguments.batch,
        device=arguments.device,
        project=str(project),
        name=run_name,
        patience=arguments.patience,
        workers=arguments.workers,
        cache=False,
        freeze=arguments.freeze or None,
        lr0=arguments.lr0,
        # Parking geometry has a consistent up direction and the detector must
        # stay sensitive to small, densely packed boxes.
        flipud=0.0,
        fliplr=0.5,
        degrees=5.0,
        scale=0.4,
        translate=0.1,
        mosaic=0.6,
        mixup=0.0,
        erasing=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        cos_lr=True,
        plots=False,
        val=True,
        verbose=True,
    )
    train_seconds = time.perf_counter() - started

    best = Path(model.trainer.best)
    evaluated = YOLO(str(best))
    results = {
        "model": arguments.model,
        "weights": str(best),
        "imgsz": arguments.imgsz,
        "epochs_requested": arguments.epochs,
        "train_seconds": round(train_seconds, 1),
        "splits": [
            evaluate(evaluated, arguments.data / "data-known.yaml", "val_known", arguments.imgsz),
            evaluate(evaluated, arguments.data / "data.yaml", "val_unseen", arguments.imgsz),
        ],
    }
    report = best.parent.parent / "development-report.json"
    report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
