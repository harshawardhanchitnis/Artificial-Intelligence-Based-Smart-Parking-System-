"""Export a trained parking-space detector to the application's ONNX runtime.

The application serves on CPU through ONNX Runtime, so the deployable artifact
is an ONNX graph plus a metadata document recording how it was selected.  The
development metrics travel with the model so the interface can report what the
detector is actually known to do.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_MODEL_ROOT = Path("D:/Projects/AI Based Smart Parking System Data/models")
DETECTOR_NAME = "parking-space-detector-v3"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--evaluation", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--architecture", default="yolo11n-obb")
    # How the graph expresses a bay.  The runtime decodes both, so a model
    # can be swapped by reinstalling an artifact.
    parser.add_argument("--representation", choices=("obb", "pose"), default="obb")
    parser.add_argument("--backup", action="store_true",
                        help="keep the artifact currently installed alongside the new one")
    arguments = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(arguments.weights))
    exported = Path(model.export(format="onnx", imgsz=arguments.imgsz, opset=18, simplify=True))

    arguments.model_root.mkdir(parents=True, exist_ok=True)
    destination = arguments.model_root / f"{DETECTOR_NAME}.onnx"
    metadata_path = arguments.model_root / f"{DETECTOR_NAME}.json"
    if arguments.backup and destination.is_file():
        # The previous detector is a measured artifact in its own right and
        # the audit trail refers to it, so it is preserved rather than
        # overwritten.
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(destination, destination.with_name(f"{DETECTOR_NAME}-{stamp}.onnx"))
        if metadata_path.is_file():
            shutil.copy2(
                metadata_path, metadata_path.with_name(f"{DETECTOR_NAME}-{stamp}.json")
            )
    shutil.copy2(exported, destination)

    evaluation_path = arguments.evaluation or arguments.weights.parent.parent / "evaluation.json"
    development = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.is_file()
        else {}
    )

    metadata = {
        "schema_version": "1.0",
        "model_name": DETECTOR_NAME,
        "architecture": arguments.architecture,
        "representation": arguments.representation,
        "input_size": arguments.imgsz,
        "confidence_threshold": arguments.confidence,
        "iou_threshold": arguments.iou,
        "max_detections": 400,
        "onnx_file": f"{DETECTOR_NAME}.onnx",
        "onnx_size_mb": round(destination.stat().st_size / 1024 / 1024, 3),
        "trained_at": datetime.now(UTC).isoformat(),
        "selection_policy": (
            "selected on val_unseen (cameras absent from training); "
            "test_unseen and the occupancy holdout were not used"
        ),
        "development": development,
        "moving_camera_supported": False,
        "scope": (
            "Detects parking-space quadrilaterals from pixels. Known cameras are "
            "near-exact; unseen cameras are proposals that require verification."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "development"}, indent=2))
    print("installed:", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
