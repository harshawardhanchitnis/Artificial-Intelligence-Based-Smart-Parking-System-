# Runtime artifacts outside Git

Under `PARKING_DATA_ROOT`, the refinement creates:

- `prepared/v2-protocol/`: source/crop manifests, rectified crops, exposure and quarantine reports
- `demo/`: 30+ image scenarios, QC overlays/contact sheets, review record, prepared videos
- `models/development/`: candidate reports, decision locks, temporary selected state
- `models/parking-occupancy-enhanced-v3.*`: enhanced occupancy ONNX and metadata
- `models/parking-slot-localizer-v1.*`: localizer ONNX and metadata
- `models/video-temporal-v1.json`: validation-selected temporal parameters
- `media/`: uploaded sources and generated overlays/video results
- SQLite database tables for media, jobs, layouts, slots, corrections, videos, and events

Expected peak local storage is driven by extracted sources and the bounded V2 crop cache. The
standard protocol should be budgeted at roughly 8–15 GB beyond the downloaded archives, plus user
media. Model files are expected to remain well below 100 MB each. A retention command removes
expired unreferenced uploads/results while preserving database references.

Training is GPU-recommended (8 GB VRAM is sufficient for the configured small batches) and may take
tens of minutes to several hours depending on disk and GPU. CPU deployment requires no CUDA and is
benchmarked during finalization. Video throughput depends on slot count, resolution, codec, and
sampling rate; the generated report records measured analysed FPS instead of making a fixed claim.
