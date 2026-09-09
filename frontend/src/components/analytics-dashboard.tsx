"use client";

import { useEffect, useState } from "react";

import { RetryPanel } from "@/components/retry-panel";
import { apiErrorMessage, apiFetch } from "@/lib/api-client";

type DatasetMetric = { dataset: string; runs: number; spaces_analysed: number; occupancy_rate: number; average_confidence: number | null; ground_truth_agreement: number | null };
type RecentRun = { id: number; dataset: string; occupancy_rate: number; created_at: string };
type Summary = { total_runs: number; total_spaces_analysed: number; average_image_analysis_ms: number | null; average_video_job_ms: number | null; image_runs: number; video_runs: number; average_confidence: number | null; ground_truth_agreement: number | null; ground_truth_runs: number; dataset_breakdown: DatasetMetric[]; user_run_breakdown: DatasetMetric[]; recent_runs: RecentRun[] };
const percent = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;

export function AnalyticsDashboard() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    apiFetch<Summary>("/analytics/summary")
      .then(setSummary)
      .catch((reason: unknown) => setError(apiErrorMessage(reason)));
  }, []);

  if (error) return <RetryPanel message={error} />;
  if (!summary) return <div className="card mt-8 p-6 text-sm text-slate-500">Loading operational analytics…</div>;

  const ms = (value: number | null) => (value === null ? "—" : `${Math.round(value).toLocaleString()} ms`);
  const cards: Array<[string, string, string]> = [
    ["Analysis runs", summary.total_runs.toString(), `${summary.image_runs} image · ${summary.video_runs} video`],
    ["Spaces analysed", summary.total_spaces_analysed.toLocaleString(), "across every saved run"],
    ["Average model confidence", percent(summary.average_confidence), "how sure the model was, not how often it was right"],
    ["Ground-truth agreement", percent(summary.ground_truth_agreement), `over ${summary.ground_truth_runs} run${summary.ground_truth_runs === 1 ? "" : "s"} with verified labels`],
  ];

  return (
    <>
      <section className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {cards.map(([label, value, note]) => <article className="card p-5" key={label}><p className="label">{label}</p><p className="mt-3 text-3xl font-black">{value}</p><p className="mt-2 text-xs leading-5 text-slate-500">{note}</p></article>)}
      </section>
      {!summary.total_runs ? (
        <div className="card mt-6 p-10 text-center"><p className="font-bold">No analysis data yet</p><p className="mt-2 text-sm text-slate-500">Run prepared scenarios to populate these charts.</p></div>
      ) : (
        <section className="mt-6 grid gap-6 xl:grid-cols-2">
          <article className="card p-6">
            <p className="label">Benchmark datasets</p><h2 className="mt-1 text-lg font-bold">Observed occupancy rate</h2><p className="mt-1 text-xs text-slate-500">Prepared scenarios with verified ground truth. Your own uploads are listed separately below.</p>
            <div className="mt-6 space-y-5">
              {summary.dataset_breakdown.map((row) => <div key={row.dataset}><div className="mb-2 flex justify-between text-sm"><span className="font-bold">{row.dataset}</span><span className="text-slate-500">{percent(row.occupancy_rate)} · {row.runs} run{row.runs === 1 ? "" : "s"}</span></div><div className="h-3 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-amber-500" style={{ width: `${row.occupancy_rate * 100}%` }} /></div></div>)}
            </div>
            {summary.user_run_breakdown.length > 0 && (
              <div className="mt-7 border-t border-slate-200 pt-5">
                <p className="label">Your runs</p>
                <p className="mt-1 text-xs text-slate-500">Uploaded images and videos. These have no verified labels, so occupancy here is a model output, not a measured rate.</p>
                <div className="mt-4 space-y-4">
                  {summary.user_run_breakdown.map((row) => <div key={row.dataset}><div className="mb-2 flex justify-between text-sm"><span className="font-bold">{row.dataset}</span><span className="text-slate-500">{percent(row.occupancy_rate)} · {row.runs} run{row.runs === 1 ? "" : "s"}</span></div><div className="h-3 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-slate-400" style={{ width: `${row.occupancy_rate * 100}%` }} /></div></div>)}
                </div>
              </div>
            )}
          </article>
          <article className="card p-6">
            <p className="label">Recent trend</p><h2 className="mt-1 text-lg font-bold">Occupancy by completed run</h2>
            <div className="mt-6 flex h-52 items-end gap-3 border-b border-slate-200 px-2">
              {summary.recent_runs.map((run) => <div className="flex min-w-0 flex-1 flex-col items-center gap-2" key={run.id}><span className="text-[10px] font-bold text-slate-400">{Math.round(run.occupancy_rate * 100)}%</span><div className="w-full max-w-10 rounded-t-md bg-indigo-500" style={{ height: `${Math.max(run.occupancy_rate * 160, 4)}px` }} title={`${run.dataset} run #${run.id}`} /><span className="text-[10px] font-bold text-slate-400">#{run.id}</span></div>)}
            </div>
            <dl className="mt-5 grid grid-cols-2 gap-4 border-t border-slate-200 pt-4 text-xs">
              <div><dt className="font-bold text-slate-600">Average image analysis</dt><dd className="mt-1 text-base font-black">{ms(summary.average_image_analysis_ms)}</dd><dd className="mt-0.5 text-slate-500">one image, end to end</dd></div>
              <div><dt className="font-bold text-slate-600">Average video job</dt><dd className="mt-1 text-base font-black">{ms(summary.average_video_job_ms)}</dd><dd className="mt-0.5 text-slate-500">whole clip, including encoding</dd></div>
            </dl>
          </article>
        </section>
      )}
    </>
  );
}
