import { VideoAnalysis } from "@/components/video-analysis";

export default function VideoPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">Prerecorded video</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-950">Track parking occupancy over time</h1>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">Process fixed-camera MP4 or AVI footage with stable geometry reuse, confidence smoothing, change debouncing, and a persisted occupancy timeline.</p>
      <VideoAnalysis />
    </div>
  );
}
