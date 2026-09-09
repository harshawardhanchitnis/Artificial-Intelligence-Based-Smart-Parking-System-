"""The serving path must stay free of the training framework, and stay exact.

Two things are asserted, and they only mean something together.

The first is a boundary: importing the FastAPI application, or any module on the
request path, must not pull PyTorch or torchvision into the process.  This is a
performance property with a measured cost behind it -- the vehicle detector runs
at 123 ms in a torch-free process and 2,103 ms in one where torch has started
its thread pool -- and it is the kind of property that decays silently, because
adding ``from app.ml.occupancy_v3_training import ...`` to a service would
reintroduce it without breaking a single functional test.

The second is equivalence: the numpy preprocessing that replaced torchvision
must produce the *same numbers*, not similar ones.  A boundary held by quietly
changing what the model sees would be worse than no boundary at all, so the
comparison is made against torchvision itself and required to be exact.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.ml.preprocessing import imagenet_batch, imagenet_chw

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Modules the application imports to answer a request.  Each is checked in its
# own interpreter, so one already-imported module cannot mask another.
SERVING_MODULES = [
    "app.main",
    "app.ml.occupancy_v3",
    "app.ml.localization",
    "app.ml.template_localizer",
    "app.ml.generalized_localizer",
    "app.ml.vehicle_detector",
    "app.services.scene_analysis",
    "app.services.video_service",
    "app.services.auto_calibration",
    "app.services.reliability_service",
]

# Modules that legitimately use torch.  They exist to train and export models
# and are imported by the training CLIs, never by the application.
TRAINING_MODULES = [
    "app.ml.occupancy_v3_training",
    "app.ml.localization_training",
    "app.ml.template_localizer_training",
]

PROBE = (
    "import importlib, sys; importlib.import_module({module!r}); "
    "print(sorted({{name.split('.')[0] for name in sys.modules "
    "if name.split('.')[0] in {{'torch', 'torchvision', 'ultralytics'}}}}))"
)


def frameworks_loaded_by(module: str) -> list[str]:
    """Import ``module`` in a fresh interpreter and report the frameworks it pulled in."""
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(module=module)],
        capture_output=True,
        text=True,
        cwd=BACKEND_ROOT,
        check=True,
    )
    return eval(result.stdout.strip())  # noqa: S307 - a list literal this test just printed


@pytest.mark.parametrize("module", SERVING_MODULES)
def test_serving_module_does_not_import_the_training_framework(module: str) -> None:
    assert frameworks_loaded_by(module) == [], (
        f"{module} pulls the training framework into the serving process; "
        "move the torch-dependent code into a *_training module"
    )


@pytest.mark.parametrize("module", TRAINING_MODULES)
def test_training_module_still_has_its_framework(module: str) -> None:
    """The split must not have broken retraining, only relocated it."""
    assert "torch" in frameworks_loaded_by(module)


def _images() -> list[Image.Image]:
    generator = np.random.default_rng(20260908)
    sizes = [(37, 41), (128, 128), (224, 224), (720, 1280), (601, 313)]
    return [
        Image.fromarray(generator.integers(0, 256, (height, width, 3), dtype=np.uint8))
        for height, width in sizes
    ]


@pytest.mark.parametrize("size", [128, 224])
def test_preprocessing_is_identical_to_torchvision(size: int) -> None:
    """Exact equality, not tolerance: the model must see what it saw before."""
    torch = pytest.importorskip("torch")
    from torchvision import transforms

    reference = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    for image in _images():
        expected = reference(image).numpy()
        actual = imagenet_chw(image, size)
        assert actual.shape == expected.shape
        assert actual.dtype == np.float32
        assert np.array_equal(actual, expected), (
            f"preprocessing drifted from torchvision at {size}px: "
            f"max |difference| = {float(np.abs(actual - expected).max())}"
        )
    assert torch is not None  # the import is the point of the check above


def test_batch_preprocessing_matches_the_single_image_path() -> None:
    images = _images()
    batch = imagenet_batch(images, 128)
    assert batch.shape == (len(images), 3, 128, 128)
    assert batch.dtype == np.float32
    for index, image in enumerate(images):
        assert np.array_equal(batch[index], imagenet_chw(image, 128))


def test_empty_batch_has_the_right_shape_for_a_session_run() -> None:
    """A scene with no bays must not reach the session with a malformed array."""
    batch = imagenet_batch([], 128)
    assert batch.shape == (0, 3, 128, 128)
    assert batch.dtype == np.float32
