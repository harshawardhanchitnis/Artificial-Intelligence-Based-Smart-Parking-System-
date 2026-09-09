"""Train one parking-space geometry candidate on the CUDA training machine.

Every candidate architecture -- four-corner keypoint, instance segmentation,
oriented box -- is trained from this one entry point on the camera-aware splits
so the comparison is not confounded by different schedules, image sizes or
augmentation.  Selection reads ``val_unseen`` only.  ``val_known`` is reported
beside it so the generalisation gap stays visible, and ``test_unseen`` is never
opened here.

    python ml/train_space_model.py --task pose --data .../quad-pose-geom
    python ml/train_space_model.py --task segment --data .../quad-seg-geom

The application runtime stays CPU-only; nothing here is imported by the server.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_MODELS = {
    "pose": "yolo11n-pose.pt",
    "segment": "yolo11n-seg.pt",
    "obb": "yolo11n-obb.pt",
    "detect": "yolo11n.pt",
}


def evaluate(model, data_yaml: Path, split_name: str, imgsz: int) -> dict[str, float]:
    """Ultralytics metrics for one split.

    These are reported for continuity with the incumbent, but they are not the
    selection signal: a box-level mAP says nothing about whether the predicted
    corners follow the bay.  ``ml/evaluate_space_geometry.py`` does that.
    """
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, split="val", verbose=False)
    box = metrics.box
    return {
        "split": split_name,
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
        "map50": round(float(box.map50), 6),
        "map50_95": round(float(box.map), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(DEFAULT_MODELS), required=True)
    parser.add_argument("--model", default=None, help="defaults to the nano model for the task")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=15)
    # Windows spawns a process per dataloader worker; 4 keeps host memory
    # bounded while still keeping the GPU fed.
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.01)
    # Mosaic stitches four images into one, which multiplies the instance
    # count by four.  A dense car park already carries ~100 bays per image,
    # and instance segmentation allocates a full mask per instance, so the
    # combination exhausted host memory building a (266, 256, 256) array.
    parser.add_argument("--mosaic", type=float, default=1.0)
    arguments = parser.parse_args()

    from ultralytics import YOLO

    weights = arguments.model or DEFAULT_MODELS[arguments.task]
    name = arguments.name or f"{Path(weights).stem}-{arguments.data.name}"
    project = Path("runs") / "space-models"

    model = YOLO(weights)
    started = time.perf_counter()
    model.train(
        data=str(arguments.data / "data.yaml"),
        epochs=arguments.epochs,
        imgsz=arguments.imgsz,
        batch=arguments.batch,
        device=arguments.device,
        workers=arguments.workers,
        patience=arguments.patience,
        freeze=arguments.freeze or None,
        lr0=arguments.lr0,
        mosaic=arguments.mosaic,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    train_seconds = round(time.perf_counter() - started, 1)

    # Ultralytics resolves a relative project under its own runs directory, so
    # the trainer is the only reliable source of the checkpoint path.
    best = Path(model.trainer.best)
    trained = YOLO(str(best))
    splits = [
        evaluate(trained, arguments.data / "data-known.yaml", "val_known", arguments.imgsz),
        evaluate(trained, arguments.data / "data.yaml", "val_unseen", arguments.imgsz),
    ]

    report = {
        "task": arguments.task,
        "model": weights,
        "weights": str(best),
        "data": str(arguments.data),
        "imgsz": arguments.imgsz,
        "epochs_requested": arguments.epochs,
        "freeze": arguments.freeze,
        "mosaic": arguments.mosaic,
        "train_seconds": train_seconds,
        "splits": splits,
    }
    print(json.dumps(report, indent=2))
    destination = best.parent.parent / "development-report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("report:", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
