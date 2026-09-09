"use client";

/* eslint-disable @next/next/no-img-element */

import Link from "next/link";
import { useEffect, useState } from "react";

import { RetryPanel } from "@/components/retry-panel";
import { apiErrorMessage, apiFetch, apiUrl } from "@/lib/api-client";

type Prediction = { id: string; polygon: number[][]; predicted_occupied: boolean; ground_truth_occupied?: boolean; confidence: number; correct?: boolean; occupancy_state?: "vacant" | "occupied" | "uncertain"; vehicle_class?: "CAR" | "TWO_WHEELER" | "TRUCK" | null };
const VEHICLE_NAMES: Record<string, string> = { CAR: "Car", TWO_WHEELER: "Two-wheeler", TRUCK: "Truck" };
type Analysis = { id: number; dataset: string; scenario_id: string; display_name: string; source_type: string; total_spaces: number; occupied_spaces: number; vacant_spaces: number; processing_time_ms: number; model_name: string | null; average_confidence: number | null; ground_truth_agreement: number | null; created_at: string; predictions: Prediction[] };
type TimelinePoint = { timestamp_seconds: number; status: string; occupied_spaces: number | null; vacant_spaces: number | null };
type VideoResult = { analysis_id: number; processed_frames: number; dropped_frames: number; stability_confidence: number; analysed_frames_per_second: number | null; timeline: TimelinePoint[]; playback_url: string };

/** Video runs store a timeline and an annotated clip instead of per-slot rows. */
function VideoDetail({ analysis }: { analysis: Analysis }) {
  const [video, setVideo] = useState<VideoResult | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    apiFetch<VideoResult>(`/video/analyses/${analysis.id}`).then(setVideo).catch(() => setFailed(true));
  }, [analysis.id]);
  if (failed) return <div className="card mt-8 p-10 text-center"><p className="font-bold">Processed video unavailable</p><p className="mt-2 text-sm text-slate-600">This run&rsquo;s generated clip is no longer on disk. Analyse the video again to recreate it.</p></div>;
  if (!video) return <div className="card mt-8 p-8 text-sm text-slate-500">Loading processed video…</div>;
  const observed = video.timeline.filter((point) => point.status === "observed" && point.occupied_spaces !== null);
  const peak = Math.max(...observed.map((point) => (point.occupied_spaces ?? 0) + (point.vacant_spaces ?? 0)), 1);
  return (
    <div className="mt-8 space-y-6">
      <section className="card overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 p-5">
          <div><p className="label">Fixed-camera video</p><h2 className="mt-1 text-lg font-bold">{analysis.display_name}</h2></div>
          <p className="text-sm font-bold text-slate-700">Camera stability {(video.stability_confidence * 100).toFixed(1)}%</p>
        </div>
        <video controls preload="metadata" src={`${apiUrl(video.playback_url)}?analysis=${video.analysis_id}`} className="aspect-video w-full bg-black">Your browser cannot play the processed video.</video>
      </section>
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[["Frames analysed", String(video.processed_frames), "passed the stability check"],
          ["Frames omitted", String(video.dropped_frames), "camera moved too much to trust"],
          ["Processing rate", video.analysed_frames_per_second === null ? "—" : `${video.analysed_frames_per_second.toFixed(2)} FPS`, "analysis throughput, not the clip frame rate"],
          ["Spaces tracked", String(analysis.total_spaces), "from the verified layout"]].map(([label, value, note]) => (
          <article className="card p-5" key={label}><p className="label">{label}</p><p className="mt-3 text-2xl font-black">{value}</p><p className="mt-2 text-xs leading-5 text-slate-500">{note}</p></article>
        ))}
      </section>
      <section className="card p-6">
        <p className="label">Occupancy timeline</p>
        <h3 className="mt-1 text-lg font-bold">Occupied spaces over the recording</h3>
        <div className="mt-5 flex h-32 items-end gap-1 overflow-x-auto" aria-label="Occupied spaces over time">
          {observed.map((point) => (
            <div key={point.timestamp_seconds} title={`${point.timestamp_seconds.toFixed(1)}s · ${point.occupied_spaces} occupied · ${point.vacant_spaces} vacant`}
                 className="min-w-1.5 flex-1 rounded-t bg-amber-400" style={{ height: `${Math.max(4, ((point.occupied_spaces ?? 0) / peak) * 100)}%` }} />
          ))}
        </div>
        <p className="mt-3 text-xs text-slate-500">{observed.length} analysed frames spanning {observed.length ? observed[observed.length - 1].timestamp_seconds.toFixed(1) : "0"}s of source footage.</p>
      </section>
    </div>
  );
}
export function AnalysisDetail({ analysisId }: { analysisId: string }) {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [view, setView] = useState<"prediction" | "ground-truth">("prediction");
  const [errorsOnly, setErrorsOnly] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    apiFetch<Analysis>(`/analysis/history/${encodeURIComponent(analysisId)}`)
      .then(setAnalysis)
      .catch((reason: unknown) => setError(apiErrorMessage(reason)));
  }, [analysisId]);

  if (error) return <RetryPanel message={error} />;
  if (!analysis) return <div className="card mt-8 p-8 text-sm text-slate-500">Loading saved analysis…</div>;
  if (analysis.source_type === "video_upload") return <VideoDetail analysis={analysis} />;
  if (!analysis.predictions.length) return <div className="card mt-8 p-10 text-center"><p className="font-bold">No per-space detail was saved for this run</p><p className="mt-2 text-sm text-slate-600">This record predates per-space persistence. Analyse the source again to inspect every space.</p><Link href="/analyse" className="mt-5 inline-block rounded-xl bg-slate-900 px-5 py-3 text-sm font-bold text-white">Open Analyse</Link></div>;

  // A user upload has no verified labels, so `correct` is absent on every slot.
  // Treating that absence as "incorrect" reported every space on an upload as an
  // error and outlined all of them, which asserted an accuracy claim about an
  // image nobody had ever labelled.
  const scored = analysis.predictions.some((slot) => typeof slot.correct === "boolean");
  const visible = scored && errorsOnly ? analysis.predictions.filter((slot) => !slot.correct) : analysis.predictions;
  const incorrect = scored ? analysis.predictions.filter((slot) => !slot.correct).length : 0;
  const uncertain = analysis.predictions.filter((slot) => slot.occupancy_state === "uncertain").length;
  const vehicles = analysis.predictions.reduce<Record<string, number>>((totals, slot) => {
    if (slot.vehicle_class) totals[slot.vehicle_class] = (totals[slot.vehicle_class] ?? 0) + 1;
    return totals;
  }, {});
  // Benchmark scenarios render their catalogue frame; an upload renders the
  // overlay saved for that analysis, which is the only image it has.
  const imageSource = scored
    ? apiUrl(`/datasets/scenarios/${encodeURIComponent(analysis.scenario_id)}/image`)
    : apiUrl(`/media/analyses/${analysis.id}/image`);
  return (
    <div className="mt-8 grid gap-6 xl:grid-cols-[1.5fr_1fr]">
      <section className="card overflow-hidden"><div className="flex flex-wrap justify-between gap-3 border-b border-slate-100 p-5"><div><p className="font-bold">{scored ? `${analysis.dataset} prediction overlay` : "Product-mode scene result"}</p><p className="mt-1 max-w-xl truncate text-xs text-slate-500">{analysis.scenario_id}</p></div>{scored && <div className="flex gap-2" role="group" aria-label="Overlay view"><button aria-pressed={view === "prediction"} onClick={() => setView("prediction")} className={view === "prediction" ? "min-h-11 rounded-lg bg-slate-900 px-3 py-2 text-xs font-bold text-white focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2" : "min-h-11 rounded-lg bg-slate-100 px-3 py-2 text-xs font-bold text-slate-800 focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2"}>AI prediction</button><button aria-pressed={view === "ground-truth"} onClick={() => setView("ground-truth")} className={view === "ground-truth" ? "min-h-11 rounded-lg bg-slate-900 px-3 py-2 text-xs font-bold text-white focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2" : "min-h-11 rounded-lg bg-slate-100 px-3 py-2 text-xs font-bold text-slate-800 focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2"}>Ground truth</button></div>}</div><div className="relative bg-slate-900"><img className="block h-auto w-full" src={imageSource} alt={`${analysis.dataset} saved analysis`} /><svg className="absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Parking-space status overlay">{visible.map((slot) => { const occupied = view === "prediction" ? slot.predicted_occupied : slot.ground_truth_occupied; const wrong = scored && !slot.correct; const amber = slot.occupancy_state === "uncertain"; return <polygon key={slot.id} points={slot.polygon.map(([x, y]) => `${x},${y}`).join(" ")} fill={amber ? "rgba(217,119,6,.30)" : occupied ? "rgba(239,68,68,.30)" : "rgba(16,185,129,.30)"} stroke={wrong ? "#facc15" : amber ? "#d97706" : occupied ? "#ef4444" : "#10b981"} strokeWidth={wrong ? "0.007" : "0.004"} vectorEffect="non-scaling-stroke" />; })}</svg></div><div className="flex items-center justify-between border-t border-slate-100 p-4 text-xs">{scored ? <><span className="font-bold text-slate-500">Yellow border marks an incorrect prediction</span><label className="flex min-h-11 items-center gap-2 font-bold"><input type="checkbox" checked={errorsOnly} onChange={(event) => setErrorsOnly(event.target.checked)} /> Errors only ({incorrect})</label></> : <span className="font-bold text-slate-500">This image has no verified labels, so no accuracy is claimed for it. Amber marks a space the system left uncertain.</span>}</div></section>
      <aside className="card h-fit p-6"><p className="label">Analysis #{analysis.id}</p><h2 className="mt-1 text-xl font-black">Saved AI result</h2><div className="mt-5 grid grid-cols-3 gap-2">{[["Total", analysis.total_spaces, "text-slate-900"], ["Vacant", analysis.vacant_spaces, "text-emerald-600"], ["Occupied", analysis.occupied_spaces, "text-red-600"]].map(([label, value, color]) => <div className="rounded-xl bg-slate-50 p-3 text-center" key={label}><p className={`text-xl font-black ${color}`}>{value}</p><p className="mt-1 text-[10px] font-bold text-slate-400">{label}</p></div>)}</div><dl className="mt-6 space-y-3 text-sm"><Row label="Model" value={analysis.model_name ?? "Legacy"} /><Row label="Confidence" value={analysis.average_confidence === null ? "—" : `${(analysis.average_confidence * 100).toFixed(1)}%`} /><Row label="Agreement" value={analysis.ground_truth_agreement === null ? "Not applicable — no verified labels" : `${(analysis.ground_truth_agreement * 100).toFixed(1)}%`} />{scored ? <Row label="Incorrect slots" value={incorrect.toString()} /> : <Row label="Uncertain spaces" value={uncertain.toString()} />}{!scored && Object.keys(vehicles).length > 0 && <Row label="Vehicles in spaces" value={Object.entries(vehicles).map(([key, count]) => `${count} ${VEHICLE_NAMES[key] ?? key}`).join(", ")} />}<Row label="Processing" value={`${analysis.processing_time_ms.toFixed(1)} ms`} /><Row label="Completed" value={new Date(analysis.created_at).toLocaleString()} /></dl><a href={apiUrl(`/reports/analyses/${analysis.id}.json`)} className="mt-6 block rounded-xl bg-slate-900 px-4 py-3 text-center text-sm font-black text-white">Download detailed JSON</a></aside>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) { return <div className="flex justify-between gap-4 border-b border-slate-100 pb-3"><dt className="text-slate-500">{label}</dt><dd className="max-w-52 text-right font-bold">{value}</dd></div>; }
