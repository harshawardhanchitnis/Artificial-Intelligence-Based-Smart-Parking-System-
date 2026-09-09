"""Image preprocessing for the ONNX serving path, without the training stack.

Every model this product serves was trained on torchvision's ``Resize ->
ToTensor -> Normalize`` pipeline, so inference has to reproduce it exactly.  It
does not have to reproduce it *with torchvision*, and doing so was expensive in
a way that did not show up in any single measurement.

Torch and ONNX Runtime each start their own thread pool sized to the machine,
and in one process those pools fight for the same cores.  Measured here, the
vehicle detector runs at 123 ms in a process that has never imported torch and
2,103 ms in one that has -- a 17x penalty on every ONNX call.  A torchvision
transform in the request path is therefore not merely a slow function; it is
what pulls the competing runtime into the process in the first place.

These implementations are bit-identical to the transforms they replace, not
merely close.  ``transforms.Resize`` on a PIL image calls ``Image.resize`` with
the same bilinear filter used here; ``ToTensor`` is a divide by 255 and a
channel transpose; ``Normalize`` is the subtraction and division written out.
``tests/test_v35_serving_runtime.py`` asserts the equality against torchvision
itself, so a future change to either side is caught rather than assumed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

#: ImageNet statistics every model in this system was normalised against.
#: Held as plain arrays so the serving path does not need the training stack to
#: know them.
NORMALISE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORMALISE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def imagenet_chw(image: Image.Image, size: int) -> np.ndarray:
    """Resize to ``size`` squared, scale to 0-1 and normalise, returning CHW.

    Equivalent to ``transforms.Compose([Resize((size, size)), ToTensor(),
    Normalize(IMAGENET_MEAN, IMAGENET_STD)])`` applied to one image.
    """
    resized = image.convert("RGB").resize((size, size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return ((array - NORMALISE_MEAN) / NORMALISE_STD).transpose(2, 0, 1)


def imagenet_batch(images: list[Image.Image], size: int) -> np.ndarray:
    """One NCHW float32 batch from several images, ready for a session run."""
    if not images:
        return np.empty((0, 3, size, size), dtype=np.float32)
    return np.stack([imagenet_chw(image, size) for image in images]).astype(np.float32)
