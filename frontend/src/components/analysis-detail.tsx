"use client";

/* eslint-disable @next/next/no-img-element */

import Link from "next/link";
import { useEffect, useState } from "react";

import { RetryPanel } from "@/components/retry-panel";
import { apiErrorMessage, apiFetch, apiUrl } from "@/lib/api-client";

type Prediction = { id: string; polygon: number[][]; predicted_occupied: boolean; ground_truth_occupied: boolean; confidence: number; correct: boolean };
type Analysis = { id: number; dataset: string; scenario_id: string; total_spaces: number; occupied_spaces: number; vacant_spaces: number; processing_time_ms: number; model_name: string | null; average_confidence: number | null; ground_truth_agreement: number | null; created_at: string; predictions: Prediction[] };
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
  if (!analysis.predictions.length) return <div className="card mt-8 p-10 text-center"><p className="font-bold">Legacy analysis record</p><p className="mt-2 text-sm text-slate-500">This run predates detailed slot persistence. Run this scenario again to inspect its predictions.</p><Link href="/analyse" className="mt-5 inline-block rounded-xl bg-slate-900 px-5 py-3 text-sm font-bold text-white">Open Analyse</Link></div>;

  const visible = errorsOnly ? analysis.predictions.filter((slot) => !slot.correct) : analysis.predictions;
  const incorrect = analysis.predictions.filter((slot) => !slot.correct).length;
  return (
    <div className="mt-8 grid gap-6 xl:grid-cols-[1.5fr_1fr]">
      <section className="card overflow-hidden"><div className="flex flex-wrap justify-between gap-3 border-b border-slate-100 p-5"><div><p className="font-bold">{analysis.dataset} prediction overlay</p><p className="mt-1 max-w-xl truncate text-xs text-slate-500">{analysis.scenario_id}</p></div><div className="flex gap-2"><button onClick={() => setView("prediction")} className={view === "prediction" ? "rounded-lg bg-slate-900 px-3 py-2 text-xs font-bold text-white" : "rounded-lg bg-slate-100 px-3 py-2 text-xs font-bold"}>AI prediction</button><button onClick={() => setView("ground-truth")} className={view === "ground-truth" ? "rounded-lg bg-slate-900 px-3 py-2 text-xs font-bold text-white" : "rounded-lg bg-slate-100 px-3 py-2 text-xs font-bold"}>Ground truth</button></div></div><div className="relative bg-slate-900"><img className="block h-auto w-full" src={apiUrl(`/datasets/scenarios/${encodeURIComponent(analysis.scenario_id)}/image`)} alt={`${analysis.dataset} saved analysis`} /><svg className="absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none">{visible.map((slot) => { const occupied = view === "prediction" ? slot.predicted_occupied : slot.ground_truth_occupied; return <polygon key={slot.id} points={slot.polygon.map(([x, y]) => `${x},${y}`).join(" ")} fill={occupied ? "rgba(239,68,68,.30)" : "rgba(16,185,129,.30)"} stroke={!slot.correct ? "#facc15" : occupied ? "#ef4444" : "#10b981"} strokeWidth={!slot.correct ? "0.007" : "0.004"} vectorEffect="non-scaling-stroke" />; })}</svg></div><div className="flex items-center justify-between border-t border-slate-100 p-4 text-xs"><span className="font-bold text-slate-500">Yellow border marks an incorrect prediction</span><label className="flex items-center gap-2 font-bold"><input type="checkbox" checked={errorsOnly} onChange={(event) => setErrorsOnly(event.target.checked)} /> Errors only ({incorrect})</label></div></section>
      <aside className="card h-fit p-6"><p className="label">Analysis #{analysis.id}</p><h2 className="mt-1 text-xl font-black">Saved AI result</h2><div className="mt-5 grid grid-cols-3 gap-2">{[["Total", analysis.total_spaces, "text-slate-900"], ["Vacant", analysis.vacant_spaces, "text-emerald-600"], ["Occupied", analysis.occupied_spaces, "text-red-600"]].map(([label, value, color]) => <div className="rounded-xl bg-slate-50 p-3 text-center" key={label}><p className={`text-xl font-black ${color}`}>{value}</p><p className="mt-1 text-[10px] font-bold text-slate-400">{label}</p></div>)}</div><dl className="mt-6 space-y-3 text-sm"><Row label="Model" value={analysis.model_name ?? "Legacy"} /><Row label="Confidence" value={analysis.average_confidence === null ? "—" : `${(analysis.average_confidence * 100).toFixed(1)}%`} /><Row label="Agreement" value={analysis.ground_truth_agreement === null ? "—" : `${(analysis.ground_truth_agreement * 100).toFixed(1)}%`} /><Row label="Incorrect slots" value={incorrect.toString()} /><Row label="Processing" value={`${analysis.processing_time_ms.toFixed(1)} ms`} /><Row label="Completed" value={new Date(analysis.created_at).toLocaleString()} /></dl><a href={apiUrl(`/reports/analyses/${analysis.id}.json`)} className="mt-6 block rounded-xl bg-slate-900 px-4 py-3 text-center text-sm font-black text-white">Download detailed JSON</a></aside>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) { return <div className="flex justify-between gap-4 border-b border-slate-100 pb-3"><dt className="text-slate-500">{label}</dt><dd className="max-w-52 text-right font-bold">{value}</dd></div>; }
