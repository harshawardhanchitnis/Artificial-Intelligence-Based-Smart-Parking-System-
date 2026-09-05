# Prerecorded fixed-camera video

MP4 is the preferred input. AVI is accepted when the locally installed OpenCV/FFmpeg decoder can
open its codec. Uploads are streamed with byte limits, checked by extension and file signature,
decoded before persistence, deduplicated by hash, and stored outside Git.

The worker builds a median reference, detects parking-space geometry once, samples frames at the
configured rate, checks ORB/homography camera stability, classifies stable slots, and applies
validation-selected median/EMA smoothing, hysteresis, and persistence. Unstable frames are marked
uncertain. Jobs persist queued/running/progress/cancelled/failed/completed state and are recovered
safely after a process restart.

Results contain an annotated MP4, compact occupancy timeline, change events, processing duration,
analysed throughput, and database links. Long inputs are bounded and cancellable. Moving-camera and
live-video operation are not claimed.
