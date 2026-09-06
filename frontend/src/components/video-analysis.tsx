"use client";

import { useEffect, useState } from "react";

import { Button, StatusBadge } from "@/components/ui";
import { apiErrorMessage, apiFetch, apiUrl } from "@/lib/api-client";

type Job = { id: number; status: string; phase: string; progress: number; result_analysis_id: number | null; error: { message: string } | null };
type TimelinePoint = { timestamp_seconds: number; status: "observed" | "uncertain_camera_motion"; occupied_spaces: number | null; vacant_spaces: number | null; average_confidence: number | null };
type Result = { analysis_id: number; processed_frames: number; dropped_frames: number; stability_confidence: number; processing_time_ms: number | null; analysed_frames_per_second: number | null; timeline: TimelinePoint[]; events: unknown[]; playback_url: string };
type PreparedVideo = { id: string; dataset: string; group_id: string; frame_count: number; duration_seconds: number; continuity: string };

export function VideoAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [prepared, setPrepared] = useState<PreparedVideo[]>([]);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [playbackDuration, setPlaybackDuration] = useState(0);

  useEffect(() => {
    apiFetch<{ videos: PreparedVideo[] }>("/datasets/videos")
      .then((payload) => setPrepared(payload.videos))
      .catch(() => setPrepared([]));
  }, []);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await apiFetch<Job>(`/video/jobs/${job.id}`);
        setJob(next);
        if (next.status === "completed" && next.result_analysis_id) setResult(await apiFetch<Result>(`/video/analyses/${next.result_analysis_id}`));
        if (next.status === "failed") setError(next.error?.message ?? "Video analysis failed");
      } catch (reason) { setError(apiErrorMessage(reason)); }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) return;
    setError(""); setResult(null);
    const data = new FormData(); data.append("file", file);
    try { setJob(await apiFetch<Job>("/video/analyse", { method: "POST", body: data, timeoutMs: 120_000 })); }
    catch (reason) { setError(apiErrorMessage(reason)); }
  }

  async function cancel() {
    if (!job) return;
    try { setJob(await apiFetch<Job>(`/video/jobs/${job.id}/cancel`, { method: "POST" })); }
    catch (reason) { setError(apiErrorMessage(reason)); }
  }

  async function analysePrepared(videoId: string) {
    setError(""); setResult(null);
    try { setJob(await apiFetch<Job>(`/video/prepared/${encodeURIComponent(videoId)}/analyse`, { method: "POST" })); }
    catch (reason) { setError(apiErrorMessage(reason)); }
  }

  const active = job && ["queued", "running"].includes(job.status);
  const observedTimeline = result?.timeline.filter((point) => point.status === "observed" && point.occupied_spaces !== null && point.vacant_spaces !== null) ?? [];
  const currentTimelineIndex =
    observedTimeline.length > 0 && playbackDuration > 0
      ? Math.min(
          observedTimeline.length - 1,
          Math.max(
            0,
            Math.floor(
              (playbackTime / playbackDuration) * observedTimeline.length,
            ),
          ),
        )
      : 0;

  const currentPoint =
    observedTimeline.length > 0
      ? observedTimeline[currentTimelineIndex]
      : undefined;
  return (
    <div className="mt-8 grid gap-6 xl:grid-cols-[360px_1fr]">
      <form className="card h-fit p-6" onSubmit={submit}>
        <p className="label">Fixed-camera workflow</p><h2 className="mt-2 text-xl font-black">Upload prerecorded video</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">The initial layout is detected once and reused only while camera-stability checks pass.</p>
        <label className="mt-6 block"><span className="mb-2 block text-sm font-bold">Video file</span><input required type="file" accept="video/mp4,video/x-msvideo,.mp4,.avi" onChange={(event) => setFile(event.target.files?.[0] ?? null)} className="block min-h-11 w-full rounded-xl border border-slate-300 bg-white p-2 text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-slate-900 file:px-3 file:py-2 file:font-bold file:text-white" /></label>
        <Button className="mt-5 w-full" disabled={!file || Boolean(active)}>Start fixed-camera analysis</Button>
        <p className="mt-3 text-xs leading-5 text-slate-600">MP4 or AVI · up to 500 MB · 5 minutes · 1920×1080</p>
        {active && <Button type="button" variant="danger" className="mt-3 w-full" onClick={cancel}>Cancel processing</Button>}
        {job && <div className="mt-5" aria-live="polite"><div className="flex justify-between text-sm font-bold"><span>{job.phase.replaceAll("_", " ")}</span><span>{Math.round(job.progress * 100)}%</span></div><div className="mt-2 h-3 overflow-hidden rounded-full bg-slate-200"><div className="h-full bg-amber-400 transition-all" style={{ width: `${job.progress * 100}%` }} /></div></div>}
        {error && <div role="alert" className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm font-semibold text-red-800">{error}</div>}
        {prepared.length > 0 && <div className="mt-6 border-t border-slate-200 pt-5"><p className="text-sm font-extrabold">Prepared time-lapse evidence</p><p className="mt-1 text-xs leading-5 text-slate-600">Same fixed camera and day, temporally ordered; these are not native continuous recordings.</p><div className="mt-3 space-y-2">{prepared.map((video) => <button type="button" disabled={Boolean(active)} onClick={() => analysePrepared(video.id)} key={video.id} className="min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-left text-xs font-bold text-slate-800 hover:bg-slate-100 focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2 disabled:bg-slate-200 disabled:text-slate-600"><span className="block">{video.dataset} · {video.group_id}</span><span className="mt-1 block font-medium text-slate-600">{video.frame_count} frames · {video.duration_seconds.toFixed(1)}s time-lapse</span></button>)}</div></div>}
      </form>
      <section className="card min-h-96 overflow-hidden">
        {!result && <div className="grid min-h-96 place-items-center p-8 text-center text-slate-600"><div><p className="font-bold text-slate-900">Processed playback and occupancy timeline</p><p className="mt-2 text-sm">Moving-camera footage is intentionally not claimed or accepted.</p></div></div>}
        {result && <><div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 p-5"><div><StatusBadge tone="success">Video analysis complete</StatusBadge><p className="mt-2 text-sm text-slate-600">{result.processed_frames} frames processed · {result.dropped_frames} unstable frames marked uncertain{result.analysed_frames_per_second === null ? "" : ` · ${result.analysed_frames_per_second.toFixed(2)} analysed FPS`}</p></div><p className="text-sm font-bold">Camera stability {(result.stability_confidence * 100).toFixed(1)}%</p></div><video
  key={result.analysis_id}
  controls
  preload="metadata"
  src={`${apiUrl(result.playback_url)}?analysis=${result.analysis_id}`}
  onLoadedMetadata={(event) => {
    setPlaybackDuration(event.currentTarget.duration || 0);
    setPlaybackTime(event.currentTarget.currentTime || 0);
  }}
  onTimeUpdate={(event) => {
    setPlaybackTime(event.currentTarget.currentTime);
  }}
  onSeeked={(event) => {
    setPlaybackTime(event.currentTarget.currentTime);
  }}
  onPlay={(event) => {
    setPlaybackDuration(event.currentTarget.duration || playbackDuration);
  }}
  className="aspect-video w-full bg-black"
>Your browser cannot play the processed video.</video>{currentPoint && <div className="grid grid-cols-3 divide-x divide-slate-200">{[["Time", `${currentPoint.timestamp_seconds.toFixed(1)}s`], ["Vacant", currentPoint.vacant_spaces], ["Occupied", currentPoint.occupied_spaces]].map(([label, value]) => <div key={label} className="p-5 text-center"><p className="text-2xl font-black">{value}</p><p className="mt-1 text-xs font-bold text-slate-600">{label}</p></div>)}</div>}<div className="border-t border-slate-200 p-5"><h3 className="font-bold">Occupancy timeline</h3><div className="mt-3 flex h-28 items-end gap-1 overflow-hidden" aria-label="Occupied spaces over time">{observedTimeline.map((point) => <div key={point.timestamp_seconds} title={`${point.timestamp_seconds}s: ${point.occupied_spaces} occupied`} className="min-w-1 flex-1 bg-amber-400" style={{ height: `${Math.max(5, (point.occupied_spaces ?? 0) / Math.max((point.occupied_spaces ?? 0) + (point.vacant_spaces ?? 0), 1) * 100)}%` }} />)}</div></div></>}
      </section>
    </div>
  );
}
