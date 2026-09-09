"""Measure automatic multi-frame calibration on cameras absent from training.

Answers the questions a fixed-camera deployment actually asks: how many frames
does the system need before it will trust a layout, how much does that layout
move if it sees different frames, and what does it cost.

Stability is measured by calibrating twice from disjoint halves of the sequence
and comparing the two layouts.  Agreement between independent halves is a
stronger claim than repeatability on one sample, because it shows the layout is
a property of the camera rather than of the frames that happened to be chosen.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

DEFAULT_SEQUENCES = Path("D:/Projects/AI Based Smart Parking System Data/development/sequences")


def read_frames(video_path: Path, count: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    frames: list[Image.Image] = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or count
        for index in range(count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index * max(total - 1, 1) / max(count - 1, 1)))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="sequence id under the development root")
    parser.add_argument("--root", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--counts", type=int, nargs="+", default=[3, 5, 7, 9, 12])
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.core.config import get_settings
    from app.ml.geometry import polygon_iou
    from app.services.auto_calibration import calibrate_from_frames

    settings = get_settings()
    catalogue = json.loads((arguments.root / f"{arguments.sequence}.json").read_text())
    video_path = Path(str(catalogue["video_path"]))

    print(f"sequence {catalogue['id']}  camera {catalogue['camera']}  {catalogue['frame_count']} frames")
    print(f"{'frames':>7}{'state':<26}{'slots':>7}{'consensus':>11}{'seconds':>9}")
    results = []
    for count in arguments.counts:
        frames = read_frames(video_path, count)
        started = time.perf_counter()
        result = calibrate_from_frames(frames, settings)
        elapsed = time.perf_counter() - started
        results.append((count, result, elapsed))
        print(
            f"{count:>7}{result.state:<26}{len(result.slots):>7}"
            f"{result.consensus:>11.3f}{elapsed:>9.1f}"
        )

    # Independent halves, to show the layout belongs to the camera.
    frames = read_frames(video_path, 12)
    first = calibrate_from_frames(frames[0::2], settings)
    second = calibrate_from_frames(frames[1::2], settings)
    print()
    print(f"disjoint halves: {len(first.slots)} vs {len(second.slots)} spaces")
    if first.slots and second.slots:
        matched = []
        for slot in first.slots:
            best = max(
                (polygon_iou(slot["polygon"], other["polygon"]) for other in second.slots),
                default=0.0,
            )
            matched.append(best)
        agreed = sum(1 for value in matched if value >= 0.5)
        print(
            f"  spaces recovered by both halves at IoU 0.5: {agreed}/{len(first.slots)}"
            f"  ({agreed / len(first.slots):.1%})"
        )
        print(f"  median polygon IoU between halves: {statistics.median(matched):.3f}")
        if arguments.out:
            arguments.out.parent.mkdir(parents=True, exist_ok=True)
            arguments.out.write_text(
                json.dumps(
                    {
                        "sequence": catalogue["id"],
                        "camera": catalogue["camera"],
                        "frames_available": catalogue["frame_count"],
                        "by_frame_count": [
                            {
                                "frames": count,
                                "state": result.state,
                                "spaces": len(result.slots),
                                "consensus": round(result.consensus, 4),
                                "seconds": round(elapsed, 2),
                            }
                            for count, result, elapsed in results
                        ],
                        "halves_spaces": [len(first.slots), len(second.slots)],
                        "halves_agreement": round(agreed / len(first.slots), 4),
                        "halves_median_polygon_iou": round(statistics.median(matched), 4),
                        "caveat": (
                            "reconstructed time-lapse, frames minutes apart; "
                            "the camera is unseen to the generalized detector"
                        ),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
