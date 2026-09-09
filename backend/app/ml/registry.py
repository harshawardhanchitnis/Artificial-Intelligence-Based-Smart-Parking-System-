"""Process-lifetime cache for loaded model predictors and their metadata.

Loading a predictor checksums its ONNX file and constructs an
``onnxruntime.InferenceSession``.  Both are expensive and neither changes while
a model file is untouched, so every loader here is keyed on the observable
identity of the files it reads -- path, size and modification time.  A model
swapped on disk therefore invalidates its own cache entry without a restart,
while repeated requests reuse one session.

Status reporting is deliberately separated from predictor construction:
``status`` readers only parse metadata JSON, so readiness and status endpoints
never build an inference session.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for annotations
    import onnxruntime as ort

_LOCK = threading.RLock()
_CACHE: dict[str, tuple[tuple[Any, ...], Any]] = {}


def file_signature(*paths: Path) -> tuple[Any, ...]:
    """Identity of a set of files: path, size and mtime for each."""
    signature: list[Any] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            signature.append((str(path), None, None))
        else:
            signature.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def cached[T](key: str, signature: tuple[Any, ...], build: Callable[[], T]) -> T:
    """Return a cached value, rebuilding it when the file signature changes.

    ``build`` runs outside the lock is *not* attempted: model construction is
    short and holding the lock keeps two concurrent requests from building the
    same session twice, which is the situation this cache exists to avoid.
    """
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is not None and entry[0] == signature:
            return entry[1]  # type: ignore[no-any-return]
        value = build()
        _CACHE[key] = (signature, value)
        return value


def cached_failure(key: str, signature: tuple[Any, ...]) -> Exception | None:
    """Look up a previously cached load failure for this signature."""
    with _LOCK:
        entry = _CACHE.get(f"{key}::error")
        if entry is not None and entry[0] == signature:
            return entry[1]  # type: ignore[no-any-return]
    return None


def remember_failure(key: str, signature: tuple[Any, ...], error: Exception) -> None:
    """Cache a load failure so a missing model is not re-probed every request."""
    with _LOCK:
        _CACHE[f"{key}::error"] = (signature, error)


def read_metadata(path: Path) -> dict[str, object]:
    """Parse and cache a model metadata document without touching its weights."""

    def build() -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    return cached(f"metadata::{path}", file_signature(path), build)


def inference_providers() -> list[str]:
    """ONNX Runtime providers for this deployment, most preferred first.

    The CPU provider is always present and always last, so inference keeps
    working on a machine with no CUDA build installed.  A GPU provider is only
    offered when onnxruntime actually exposes it *and* ``PARKING_ONNX_PROVIDER``
    asks for it, because enabling CUDA changes startup cost and memory use and
    should be a deliberate deployment choice rather than an accident of which
    wheel happens to be installed.
    """
    import os

    import onnxruntime as ort

    requested = os.environ.get("PARKING_ONNX_PROVIDER", "cpu").strip().lower()
    if requested in {"", "cpu"}:
        return ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    preferred = {
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "directml": "DmlExecutionProvider",
    }.get(requested)
    if preferred and preferred in available:
        return [preferred, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


#: Upper bound on the threads any one inference session may spin up.
#:
#: ONNX Runtime sizes a session's thread pool to every logical processor it can
#: see.  That is a good default for a process serving one model; it is a bad one
#: here, because answering a single image request keeps four sessions alive --
#: layout classifier, space detector, vehicle detector, occupancy classifier --
#: and each builds its own pool.  On this 16-core hybrid part that is roughly 88
#: threads contending for 22 hardware threads, and they spend their time
#: descheduling one another rather than doing arithmetic.
#:
#: Measured over the full detect + classify pipeline, three passes with the
#: order reversed between them so machine drift could not favour one setting:
#:
#:     ONNX Runtime default   7,936 ms
#:     8 threads              3,818 ms
#:     6 threads              2,698 ms
#:     4 threads              1,539 ms
#:     3 threads              1,062 ms   <- shipped
#:     2 threads              1,172 ms
#:
#: Repeated later on a quieter machine, the same sweep put the floor at four
#: (930 ms), with three at 1,182 ms and the ONNX Runtime default at 2,890 ms.
#: The optimum therefore moves with machine load -- but the default is wrong
#: under both conditions, which is the finding that matters.  Three is kept
#: because it has the smaller worst case across the two measurements (27% off
#: the floor on an idle machine, against 45% for four on a busy one), and
#: because sharing a host is the situation a default has to survive.
#:
#: Nothing about any model changes: ONNX Runtime partitions the same operators
#: over fewer threads, and ``tests/test_v35_serving_runtime.py`` pins the
#: outputs to prove it.
DEFAULT_SESSION_THREADS = 3


def session_thread_limit() -> int:
    """Threads per inference session, overridable for other hardware.

    ``PARKING_ONNX_THREADS=0`` restores ONNX Runtime's own default, which is
    the right choice on a machine that serves one model per process.
    """
    import os

    raw = os.environ.get("PARKING_ONNX_THREADS", "").strip()
    if not raw:
        return DEFAULT_SESSION_THREADS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SESSION_THREADS
    return max(value, 0)


def session_options() -> ort.SessionOptions:
    """Session options shared by every model this application serves."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    threads = session_thread_limit()
    if threads:
        options.intra_op_num_threads = threads
        # One session runs at a time on the request path, so there is nothing
        # for a second operator-level pool to overlap with.
        options.inter_op_num_threads = 1
    return options


def clear() -> None:
    """Drop every cached entry.  Used by tests and after model installation."""
    with _LOCK:
        _CACHE.clear()


def entry_count() -> int:
    """Number of live cache entries, for diagnostics and tests."""
    with _LOCK:
        return len(_CACHE)
