"use client";

import { useEffect, useState } from "react";

type DatasetMetric = { dataset: string; runs: number; spaces_analysed: number; occupancy_rate: number; average_confidence: number | null; ground_truth_agreement: number | null };
type RecentRun = { id: number; dataset: string; occupancy_rate: number; created_at: string };
type Summary = { total_runs: number; total_spaces_analysed: number; average_processing_time_ms: number | null; average_confidence: number | null; ground_truth_agreement: number | null; dataset_breakdown: DatasetMetric[]; recent_runs: RecentRun[] };
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";
const percent = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;

export function AnalyticsDashboard() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    fetch(`${apiBase}/analytics/summary`)
      .then((response) => response.ok ? response.json() as Promise<Summary> : Promise.reject())
      .then(setSummary)
      .catch(() => setError(true));
  }, []);

  if (error) return <div className="card mt-8 p-6 text-sm text-red-600">Analytics API is unavailable.</div>;
  if (!summary) return <div className="card mt-8 p-6 text-sm text-slate-500">Loading operational analytics…</div>;

  const cards = [
    ["Analysis runs", summary.total_runs.toString()],
    ["Spaces analysed", summary.total_spaces_analysed.toString()],
    ["Average confidence", percent(summary.average_confidence)],
    ["Ground-truth agreement", percent(summary.ground_truth_agreement)],
  ];

  return (
    <>
      <section className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {cards.map(([label, value]) => <article className="card p-5" key={label}><p className="label">{label}</p><p className="mt-3 text-3xl font-black">{value}</p></article>)}
      </section>
      {!summary.total_runs ? (
        <div className="card mt-6 p-10 text-center"><p className="font-bold">No analysis data yet</p><p className="mt-2 text-sm text-slate-500">Run prepared scenarios to populate these charts.</p></div>
      ) : (
        <section className="mt-6 grid gap-6 xl:grid-cols-2">
          <article className="card p-6">
            <p className="label">Dataset comparison</p><h2 className="mt-1 text-lg font-bold">Observed occupancy rate</h2>
            <div className="mt-6 space-y-5">
              {summary.dataset_breakdown.map((row) => <div key={row.dataset}><div className="mb-2 flex justify-between text-sm"><span className="font-bold">{row.dataset}</span><span className="text-slate-500">{percent(row.occupancy_rate)} · {row.runs} run{row.runs === 1 ? "" : "s"}</span></div><div className="h-3 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-amber-500" style={{ width: `${row.occupancy_rate * 100}%` }} /></div></div>)}
            </div>
          </article>
          <article className="card p-6">
            <p className="label">Recent trend</p><h2 className="mt-1 text-lg font-bold">Occupancy by completed run</h2>
            <div className="mt-6 flex h-52 items-end gap-3 border-b border-slate-200 px-2">
              {summary.recent_runs.map((run) => <div className="flex min-w-0 flex-1 flex-col items-center gap-2" key={run.id}><span className="text-[10px] font-bold text-slate-400">{Math.round(run.occupancy_rate * 100)}%</span><div className="w-full max-w-10 rounded-t-md bg-indigo-500" style={{ height: `${Math.max(run.occupancy_rate * 160, 4)}px` }} title={`${run.dataset} run #${run.id}`} /><span className="text-[10px] font-bold text-slate-400">#{run.id}</span></div>)}
            </div>
            <p className="mt-4 text-xs text-slate-500">Average inference time: {summary.average_processing_time_ms?.toFixed(1) ?? "—"} ms</p>
          </article>
        </section>
      )}
    </>
  );
}
