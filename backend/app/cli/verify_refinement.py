from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import inspect

from app.core.config import get_settings
from app.datasets.integrity import read_jsonl, sha256_file, verify_partition_integrity
from app.datasets.preparation import verify_prepared_data
from app.db.session import engine, initialize_database
from app.ml.localization import localizer_status
from app.ml.occupancy_v3 import occupancy_v3_status

REQUIRED_TABLES = {
    "media_assets",
    "analysis_jobs",
    "detected_layouts",
    "detected_slots",
    "video_analyses",
    "occupancy_events",
    "layout_corrections",
}


def verify_refinement(data_root: Path, model_root: Path) -> dict[str, object]:
    scenario = verify_prepared_data(data_root)
    if scenario["scenario_count"] < 30 or any(
        count < 10 for count in scenario["dataset_counts"].values()
    ):
        raise RuntimeError("Prepared image catalogue does not meet the 10-per-dataset contract")
    qc_path = data_root / "prepared" / "scenario-qc" / "scenario-qc-report.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if not qc.get("geometry_valid") or qc["manual_overlay_qc"]["status"] != "confirmed":
        raise RuntimeError("Prepared scenario geometry and manual overlay QC are incomplete")

    protocol_root = data_root / "prepared" / "v2-protocol"
    occupancy_rows = read_jsonl(protocol_root / "occupancy-manifest.jsonl")
    integrity = verify_partition_integrity(occupancy_rows)
    exposure = json.loads((protocol_root / "exposure-report.json").read_text(encoding="utf-8"))
    if exposure.get("historically_virgin") is not False:
        raise RuntimeError("V2 protocol historical-exposure disclosure is missing")

    occupancy = occupancy_v3_status(model_root)
    localizer = localizer_status(model_root)
    if not occupancy["ready"] or not localizer["ready"]:
        raise RuntimeError(
            "Enhanced occupancy and automatic localisation models must both be ready"
        )
    regression_path = data_root / "prepared" / "scenario-regression-report.json"
    regression = json.loads(regression_path.read_text(encoding="utf-8"))
    if regression.get("catalogue_sha256") != sha256_file(data_root / "demo" / "catalogue.json"):
        raise RuntimeError("Prepared-scenario regression does not match the current catalogue")
    if regression.get("model_artifacts", {}).get("enhanced_onnx_sha256") != occupancy.get(
        "onnx_sha256"
    ):
        raise RuntimeError("Prepared-scenario regression does not match the enhanced model")

    video_path = data_root / "demo" / "videos" / "catalogue.json"
    videos = json.loads(video_path.read_text(encoding="utf-8"))
    if videos.get("by_dataset") != {"PKLot": 3, "CNRPark+EXT": 3, "ACPDS": 0}:
        raise RuntimeError("Prepared time-lapse inventory does not meet the traceable 3+3 contract")
    for row in videos["videos"]:
        path = data_root / row["video_path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Prepared video is missing or changed: {row['id']}")

    initialize_database()
    tables = set(inspect(engine).get_table_names())
    missing_tables = REQUIRED_TABLES - tables
    if missing_tables:
        raise RuntimeError(f"Refinement database tables are missing: {sorted(missing_tables)}")
    return {
        "valid": True,
        "prepared_images": scenario,
        "scenario_qc_sha256": sha256_file(qc_path),
        "protocol_integrity": integrity,
        "historical_exposure": exposure["protection_statement"],
        "enhanced_occupancy": occupancy,
        "automatic_localizer": localizer,
        "prepared_scenario_regression": regression["overall"],
        "prepared_videos": videos["by_dataset"],
        "database_tables": sorted(REQUIRED_TABLES),
    }


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Verify the complete integrated refinement")
    parser.add_argument("--data-root", type=Path, default=settings.parking_data_root)
    parser.add_argument("--model-root", type=Path)
    arguments = parser.parse_args()
    root = arguments.data_root.resolve()
    print(json.dumps(verify_refinement(root, arguments.model_root or root / "models"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
