"""Train a Keypoint R-CNN four-corner parking-space detector.

The non-YOLO candidate.  It is a deliberately different family from everything
else in the comparison -- a two-stage region-proposal detector on a ResNet50-FPN
backbone, against one-stage anchor-free YOLO heads -- while solving exactly the
same task on exactly the same data, so a difference in the result is a
difference between the architectures rather than between the problems they were
given.

It reads the same ``quad-pose`` dataset the YOLO pose model reads, so the label
format, the camera-aware splits and the evaluation are all shared.

    python ml/train_keypoint_rcnn.py --data .../quad-pose-geom --epochs 15
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class QuadKeypointDataset(Dataset):
    """YOLO pose labels presented as torchvision keypoint targets."""

    def __init__(self, root: Path, split: str) -> None:
        self.image_dir = root / "images" / split
        self.label_dir = root / "labels" / split
        self.stems = sorted(
            path.stem for path in self.image_dir.glob("*.jpg")
            if (self.label_dir / f"{path.stem}.txt").is_file()
        )

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        from torchvision.transforms import functional

        stem = self.stems[index]
        image = Image.open(self.image_dir / f"{stem}.jpg").convert("RGB")
        width, height = image.size
        boxes: list[list[float]] = []
        keypoints: list[list[list[float]]] = []
        for line in (self.label_dir / f"{stem}.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) != 17:
                continue
            values = [float(value) for value in parts[1:]]
            corners = [
                [values[4 + i * 3] * width, values[5 + i * 3] * height, 1.0] for i in range(4)
            ]
            xs = [corner[0] for corner in corners]
            ys = [corner[1] for corner in corners]
            # A degenerate box makes the RPN loss non-finite, so drop it here.
            if max(xs) - min(xs) < 2 or max(ys) - min(ys) < 2:
                continue
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            keypoints.append(corners)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "keypoints": torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 4, 3),
        }
        return functional.to_tensor(image), target


def collate(batch):
    return tuple(zip(*batch, strict=True))


def build_model(trainable_layers: int):
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.keypoint_rcnn import (
        KeypointRCNNPredictor,
        keypointrcnn_resnet50_fpn,
    )

    # COCO weights give the backbone the same head start the YOLO candidates get
    # from their own pretraining; only the keypoint head is replaced, because a
    # parking bay has four corners and a person has seventeen joints.
    model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1,
        num_keypoints=17,
    )
    in_channels = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_channels, 4)
    # A dense car park holds far more bays than COCO holds people.
    model.roi_heads.detections_per_img = 300
    model.rpn._pre_nms_top_n = {"training": 2000, "testing": 2000}
    model.rpn._post_nms_top_n = {"training": 2000, "testing": 1000}
    for name, parameter in model.backbone.body.named_parameters():
        if not any(name.startswith(f"layer{index}") for index in range(5 - trainable_layers, 5)):
            parameter.requires_grad_(False)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--trainable-layers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/keypoint-rcnn"))
    arguments = parser.parse_args()

    device = torch.device(arguments.device if torch.cuda.is_available() else "cpu")
    dataset = QuadKeypointDataset(arguments.data, "train")
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch,
        shuffle=True,
        num_workers=arguments.workers,
        collate_fn=collate,
        persistent_workers=arguments.workers > 0,
    )
    model = build_model(arguments.trainable_layers).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(parameters, lr=arguments.lr, momentum=0.9, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=arguments.epochs)

    arguments.out.mkdir(parents=True, exist_ok=True)
    print(f"training on {len(dataset)} images, device {device}")
    started = time.perf_counter()
    peak_memory = 0.0
    for epoch in range(arguments.epochs):
        model.train()
        running = 0.0
        seen = 0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()} for target in targets
            ]
            # An image whose bays were all dropped has no boxes to train on.
            if any(target["boxes"].numel() == 0 for target in targets):
                continue
            losses = model(images, targets)
            total = sum(losses.values())
            if not torch.isfinite(total):
                continue
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimiser.step()
            running += float(total)
            seen += 1
        schedule.step()
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1e9)
        print(
            f"epoch {epoch + 1}/{arguments.epochs}  loss {running / max(seen, 1):.4f}"
            f"  peak VRAM {peak_memory:.2f} GB  {time.perf_counter() - started:.0f}s",
            flush=True,
        )
        torch.save(model.state_dict(), arguments.out / "last.pt")

    report = {
        "architecture": "keypointrcnn_resnet50_fpn",
        "family": "two-stage region proposal (non-YOLO)",
        "data": str(arguments.data),
        "epochs": arguments.epochs,
        "batch": arguments.batch,
        "trainable_backbone_layers": arguments.trainable_layers,
        "train_seconds": round(time.perf_counter() - started, 1),
        "peak_vram_gb": round(peak_memory, 2),
        "weights": str(arguments.out / "last.pt"),
    }
    (arguments.out / "development-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
