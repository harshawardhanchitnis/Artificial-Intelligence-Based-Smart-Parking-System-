from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.ml.features import extract_features
from app.ml.inference import analyse_scenario, verify_model
from app.ml.model_store import ModelNotReadyError, load_model, model_paths, model_status
from app.ml.training import train_model


def build_catalogue(data_root: Path) -> list[dict[str, object]]:
    scenarios = []
    media_root = data_root / "demo" / "media" / "fixture"
    media_root.mkdir(parents=True)
    for scenario_index in range(4):
        image = Image.new("RGB", (240, 80), (190, 190, 190))
        draw = ImageDraw.Draw(image)
        slots = []
        for slot_index in range(6):
            left = slot_index / 6
            right = (slot_index + 1) / 6
            occupied = slot_index % 2 == scenario_index % 2
            color = (35, 45, 65) if occupied else (220, 210, 150)
            x0, x1 = int(left * 240) + 3, int(right * 240) - 3
            draw.rectangle((x0, 10, x1, 70), fill=color, outline=(255, 255, 255))
            if occupied:
                draw.rectangle((x0 + 8, 20, x1 - 8, 60), fill=(180, 40, 35))
            slots.append(
                {
                    "id": str(slot_index + 1),
                    "polygon": [
                        [left, 0.05],
                        [right, 0.05],
                        [right, 0.95],
                        [left, 0.95],
                    ],
                    "occupied": occupied,
                }
            )
        image_path = media_root / f"scenario-{scenario_index}.jpg"
        image.save(image_path, quality=95)
        occupied_count = sum(bool(slot["occupied"]) for slot in slots)
        scenarios.append(
            {
                "id": f"fixture-{scenario_index}",
                "dataset": ["PKLot", "CNRPark+EXT", "ACPDS"][scenario_index % 3],
                "lot": f"lot-{scenario_index}",
                "condition": "Synthetic",
                "image_path": image_path.relative_to(data_root).as_posix(),
                "total_spaces": len(slots),
                "occupied_spaces": occupied_count,
                "vacant_spaces": len(slots) - occupied_count,
                "slots": slots,
            }
        )
    catalogue = {
        "schema_version": "1.0",
        "profile": "test",
        "scenario_count": len(scenarios),
        "datasets": [],
        "scenarios": scenarios,
    }
    catalogue_path = data_root / "demo" / "catalogue.json"
    catalogue_path.write_text(json.dumps(catalogue), encoding="utf-8")
    return scenarios


def test_feature_vector_is_finite_and_stable() -> None:
    vector = extract_features(Image.new("RGB", (40, 60), (100, 120, 140)))
    assert vector.shape == (238,)
    assert vector.dtype.name == "float32"
    assert bool((vector == vector).all())


def test_training_inference_and_verification(tmp_path: Path) -> None:
    scenarios = build_catalogue(tmp_path)
    model_root = tmp_path / "models"
    assert model_status(model_root)["ready"] is False

    report = train_model(tmp_path, model_root, progress=None)
    assert report["ready"] is True
    assert report["training_samples"] == 24
    assert model_status(model_root)["ready"] is True

    result = analyse_scenario(tmp_path, model_root, str(scenarios[0]["id"]))
    assert result["total_spaces"] == 6
    assert len(result["predictions"]) == 6
    assert 0.0 <= result["ground_truth_agreement"] <= 1.0
    assert all(0.5 <= row["confidence"] <= 1.0 for row in result["predictions"])

    verification = verify_model(tmp_path, model_root)
    assert verification["valid"] is True
    assert verification["scenario_count"] == 4
    assert verification["slot_count"] == 24
    assert verification["catalogue_ground_truth_agreement"] >= 0.9


def test_modified_weights_are_rejected(tmp_path: Path) -> None:
    build_catalogue(tmp_path)
    model_root = tmp_path / "models"
    train_model(tmp_path, model_root, progress=None)
    weights_path, _ = model_paths(model_root)
    with weights_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ModelNotReadyError, match="checksum"):
        load_model(model_root)
