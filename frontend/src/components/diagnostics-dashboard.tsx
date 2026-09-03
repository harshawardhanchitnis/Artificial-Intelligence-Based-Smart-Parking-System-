"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Metrics = { evaluated_slots: number; true_occupied: number; true_vacant: number; false_occupied: number; false_vacant: number; accuracy: number | null; precision: number | null; recall: number | null; specificity: number | null; f1_score: number | null };
type DatasetMetrics = Metrics & { dataset: string };
type SlotError = { analysis_id: number; dataset: string; scenario_id: string; slot_id: string; predicted_occupied: boolean; ground_truth_occupied: boolean; confidence: number | null };
type Diagnostics = { total_runs: number; eligible_runs: number; legacy_runs: number; overall: Metrics; dataset_breakdown: DatasetMetrics[]; recent_errors: SlotError[] };
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";
const percent = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;

export function DiagnosticsDashboard() {
  const [report, setReport] = useState<Diagnostics | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    fetch(`${apiBase}/diagnostics/summary`)
      .then((response) => response.ok ? response.json() as Promise<Diagnostics> : Promise.reject())
      .then(setReport)
      .catch(() => setError(true));
  }, []);

  if (error) return <div className="card mt-8 p-6 text-sm text-red-600">Diagnostics API is unavailable.</div>;
  if (!report) return <div className="card mt-8 p-6 text-sm text-slate-500">Calculating saved prediction diagnostics…</div>;
  if (!report.overall.evaluated_slots) return <div className="card mt-8 p-10 text-center"><p className="font-bold">No detailed predictions available</p><p className="mt-2 text-sm text-slate-500">Run a current AI analysis or the guided showcase to generate diagnostics.</p></div>;

  const quality = [["Accuracy", report.overall.accuracy], ["Precision", report.overall.precision], ["Occupied recall", report.overall.recall], ["Vacant specificity", report.overall.specificity], ["F1 score", report.overall.f1_score]] as const;
  const confusion = [["Correct occupied", report.overall.true_occupied, "bg-red-50 text-red-700"], ["False occupied", report.overall.false_occupied, "bg-amber-50 text-amber-700"], ["Missed occupied", report.overall.false_vacant, "bg-amber-50 text-amber-700"], ["Correct vacant", report.overall.true_vacant, "bg-emerald-50 text-emerald-700"]] as const;

  return (
    <div className="mt-8 space-y-6">
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">{quality.map(([label, value]) => <article className="card p-5" key={label}><p className="label">{label}</p><p className="mt-3 text-3xl font-black">{percent(value)}</p></article>)}</section>
      <section className="grid gap-6 xl:grid-cols-[1fr_1.4fr]">
        <article className="card p-6"><p className="label">Confusion matrix</p><h2 className="mt-1 text-lg font-bold">{report.overall.evaluated_slots} evaluated slots</h2><div className="mt-5 grid grid-cols-2 gap-3">{confusion.map(([label, value, color]) => <div className={`rounded-2xl p-5 ${color}`} key={label}><p className="text-3xl font-black">{value}</p><p className="mt-1 text-xs font-bold">{label}</p></div>)}</div><p className="mt-4 text-xs text-slate-500">{report.eligible_runs} prediction-enabled runs · {report.legacy_runs} legacy runs excluded</p></article>
        <article className="card overflow-hidden"><div className="border-b border-slate-100 p-5"><p className="label">Dataset diagnostics</p><h2 className="mt-1 text-lg font-bold">Quality by source</h2></div><div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="bg-slate-50 text-xs uppercase text-slate-400"><tr><th className="p-4">Dataset</th><th className="p-4">Slots</th><th className="p-4">Accuracy</th><th className="p-4">Precision</th><th className="p-4">Recall</th><th className="p-4">F1</th></tr></thead><tbody className="divide-y divide-slate-100">{report.dataset_breakdown.map((row) => <tr key={row.dataset}><td className="p-4 font-black">{row.dataset}</td><td className="p-4">{row.evaluated_slots}</td><td className="p-4 font-bold">{percent(row.accuracy)}</td><td className="p-4">{percent(row.precision)}</td><td className="p-4">{percent(row.recall)}</td><td className="p-4">{percent(row.f1_score)}</td></tr>)}</tbody></table></div></article>
      </section>
      <article className="card overflow-hidden"><div className="border-b border-slate-100 p-5"><p className="label">Inspection queue</p><h2 className="mt-1 text-lg font-bold">Recent incorrect slot predictions</h2></div>{report.recent_errors.length ? <div className="divide-y divide-slate-100">{report.recent_errors.map((row, index) => <div className="flex flex-col justify-between gap-3 p-4 md:flex-row md:items-center" key={`${row.analysis_id}-${row.slot_id}-${index}`}><div><p className="text-sm font-bold">{row.dataset} · slot {row.slot_id}</p><p className="mt-1 text-xs text-slate-500">Predicted {row.predicted_occupied ? "occupied" : "vacant"}; ground truth {row.ground_truth_occupied ? "occupied" : "vacant"}{row.confidence === null ? "" : ` · ${(row.confidence * 100).toFixed(1)}% confidence`}</p></div><Link href={`/history/${row.analysis_id}`} className="rounded-lg bg-slate-900 px-4 py-2 text-center text-xs font-bold text-white">Inspect analysis</Link></div>)}</div> : <div className="p-8 text-center text-sm font-semibold text-emerald-700">No incorrect predictions are present in the stored detailed runs.</div>}</article>
    </div>
  );
}
