"""Export the selected vehicle detector to the ONNX artifact the runtime loads.

The detector ships with its source taxonomy intact in the metadata and is folded
into the three product classes at inference time by ``app.ml.vehicle_taxonomy``.
Keeping the mapping in code rather than baking it into the graph means the
policy is reviewable, testable and identical for every detector that might
replace this one; the runtime discards every class that does not map, so no
unsupported class can reach the interface regardless of what the graph emits.

    python ml/export_vehicle_detector.py --weights weights/yolo11n.pt --imgsz 960
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_MODEL_ROOT = Path("D:/Projects/AI Based Smart Parking System Data/models")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--development", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    arguments = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.ml.vehicle_detector import (
        DETECTOR_METADATA,
        DETECTOR_ONNX,
        DETECTOR_SCHEMA,
    )
    from app.ml.vehicle_taxonomy import (
        EXTENDED_CLASS_MAP,
        PRODUCT_CLASSES,
        UNSUPPORTED_VEHICLE_LABELS,
    )
    from ultralytics import YOLO

    model = YOLO(arguments.weights)
    class_names = [str(model.names[index]) for index in sorted(model.names)]
    exported = Path(
        model.export(format="onnx", imgsz=arguments.imgsz, opset=arguments.opset, simplify=True)
    )

    arguments.model_root.mkdir(parents=True, exist_ok=True)
    destination = arguments.model_root / DETECTOR_ONNX
    shutil.copy2(exported, destination)

    mapped = {
        name: EXTENDED_CLASS_MAP[name.lower()]
        for name in class_names
        if name.lower() in EXTENDED_CLASS_MAP
    }
    development: dict[str, object] = {}
    if arguments.development and arguments.development.is_file():
        development = json.loads(arguments.development.read_text(encoding="utf-8"))

    metadata = {
        "schema_version": DETECTOR_SCHEMA,
        "model_name": "vehicle-detector-v3",
        "architecture": Path(arguments.weights).stem,
        "input_size": arguments.imgsz,
        "confidence_threshold": arguments.conf,
        "iou_threshold": arguments.iou,
        "max_detections": 300,
        "onnx_file": DETECTOR_ONNX,
        "onnx_size_mb": round(destination.stat().st_size / 1e6, 3),
        "trained_at": datetime.now(UTC).isoformat(),
        "source_taxonomy": "COCO-80 (pretrained, not fine-tuned)",
        "class_names": class_names,
        "product_classes": list(PRODUCT_CLASSES),
        "class_mapping": mapped,
        "selection_policy": (
            "selected on unseen cameras by occupied-bay vehicle recall and vacant-bay "
            "false rate derived from dataset occupancy labels; the protected holdout "
            "was not used"
        ),
        "development": development,
        "unsupported_vehicle_labels": sorted(UNSUPPORTED_VEHICLE_LABELS),
        "scope": (
            "Detects vehicles across the whole frame, independently of parking-space "
            "geometry. Exactly CAR / TWO_WHEELER / TRUCK are surfaced. A source class "
            "outside that set is never renamed into it -- bus is not a truck and "
            "bicycle is not a two-wheeler -- but a vehicle-shaped object carrying one "
            "of those labels is retained without a class, because it still occupies "
            "the bay it stands in. Non-vehicle classes are discarded at decode time."
        ),
    }
    (arguments.model_root / DETECTOR_METADATA).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "class_names"}, indent=2))
    print("installed:", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
