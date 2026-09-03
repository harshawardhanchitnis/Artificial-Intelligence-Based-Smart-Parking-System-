"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Matrix = { true_occupied: number; true_vacant: number; false_occupied: number; false_vacant: number };
type RunMetrics = Matrix & { evaluated_slots: number; accuracy: number | null; balanced_accuracy: number | null; precision: number | null; recall: number | null; specificity: number | null; f1_score: number | null };
type BenchmarkMetrics = { accuracy: number; balanced_accuracy: number; precision_occupied: number; recall_occupied: number; f1_occupied: number; specificity_vacant: number; confusion_matrix: Matrix; unique_samples: number; unique_sources: number; unique_groups: number; majority_baseline_accuracy: number };
type DatasetBenchmark = BenchmarkMetrics & { dataset: string };
type SlotError = { analysis_id: number; dataset: string; scenario_id: string; slot_id: string; predicted_occupied: boolean; ground_truth_occupied: boolean; confidence: number | null };
type ApplicationRuns = { semantic_label: string; total_runs: number; prediction_enabled_runs: number; legacy_runs: number; stored_slot_predictions: number; unique_scenarios: number; unique_scenario_slots: number; repeated_slot_predictions: number; includes_repeated_executions: boolean; context_note: string; overall: RunMetrics; dataset_breakdown: (RunMetrics & { dataset: string })[]; recent_errors: SlotError[] };
type Benchmark = { validation: BenchmarkMetrics; unseen_test: BenchmarkMetrics; unseen_test_by_dataset: DatasetBenchmark[] };
type Diagnostics = { application_runs: ApplicationRuns; independent_benchmark: Benchmark | null; benchmark_available: boolean };

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";
const percent = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;

function MetricCards({ metrics }: { metrics: BenchmarkMetrics }) {
  const cards = [
    ["Accuracy", metrics.accuracy],
    ["Balanced accuracy", metrics.balanced_accuracy],
    ["Occupied precision", metrics.precision_occupied],
    ["Occupied recall", metrics.recall_occupied],
    ["Occupied F1", metrics.f1_occupied],
    ["Vacant specificity", metrics.specificity_vacant],
  ] as const;
  return <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-6">{cards.map(([label, value]) => <article className="card p-5" key={label}><p className="label">{label}</p><p className="mt-3 text-3xl font-black">{percent(value)}</p></article>)}</div>;
}

function MatrixPanel({ title, metrics }: { title: string; metrics: BenchmarkMetrics }) {
  const matrix = metrics.confusion_matrix;
  const cells = [
    ["Correct occupied", matrix.true_occupied, "bg-red-50 text-red-700"],
    ["False occupied", matrix.false_occupied, "bg-amber-50 text-amber-700"],
    ["Missed occupied", matrix.false_vacant, "bg-amber-50 text-amber-700"],
    ["Correct vacant", matrix.true_vacant, "bg-emerald-50 text-emerald-700"],
  ] as const;
  return <article className="card p-6"><p className="label">{title}</p><h2 className="mt-1 text-lg font-bold">{metrics.unique_samples.toLocaleString()} unique samples</h2><div className="mt-5 grid grid-cols-2 gap-3">{cells.map(([label, value, color]) => <div className={`rounded-2xl p-5 ${color}`} key={label}><p className="text-3xl font-black">{value}</p><p className="mt-1 text-xs font-bold">{label}</p></div>)}</div><p className="mt-4 text-xs text-slate-500">{metrics.unique_sources.toLocaleString()} source images · {metrics.unique_groups} non-overlapping groups · majority baseline {percent(metrics.majority_baseline_accuracy)}</p></article>;
}

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
  if (!report) return <div className="card mt-8 p-6 text-sm text-slate-500">Loading model benchmark and stored-run diagnostics…</div>;
  const runs = report.application_runs;
  const benchmark = report.independent_benchmark;

  return <div className="mt-8 space-y-8">
    <section className="space-y-5">
      <div><p className="label">Independent model benchmark</p><h2 className="mt-1 text-2xl font-black">Unseen test performance</h2><p className="mt-2 max-w-4xl text-sm text-slate-500">These samples were excluded from model fitting, threshold selection, and feature decisions. Re-running an application scenario does not alter these results.</p></div>
      {benchmark ? <>
        <MetricCards metrics={benchmark.unseen_test} />
        <div className="grid gap-6 xl:grid-cols-[1fr_1.5fr]">
          <MatrixPanel title="Unseen test confusion matrix" metrics={benchmark.unseen_test} />
          <article className="card overflow-hidden"><div className="border-b border-slate-100 p-5"><p className="label">Generalization by source</p><h3 className="mt-1 text-lg font-bold">Held-out dataset results</h3></div><div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="bg-slate-50 text-xs uppercase text-slate-400"><tr><th className="p-4">Partition</th><th className="p-4">Samples</th><th className="p-4">Accuracy</th><th className="p-4">Balanced</th><th className="p-4">Precision</th><th className="p-4">Recall</th><th className="p-4">F1</th></tr></thead><tbody className="divide-y divide-slate-100"><tr><td className="p-4 font-black">Validation</td><td className="p-4">{benchmark.validation.unique_samples}</td><td className="p-4">{percent(benchmark.validation.accuracy)}</td><td className="p-4">{percent(benchmark.validation.balanced_accuracy)}</td><td className="p-4">{percent(benchmark.validation.precision_occupied)}</td><td className="p-4">{percent(benchmark.validation.recall_occupied)}</td><td className="p-4">{percent(benchmark.validation.f1_occupied)}</td></tr>{benchmark.unseen_test_by_dataset.map((row) => <tr key={row.dataset}><td className="p-4 font-black">Test · {row.dataset}</td><td className="p-4">{row.unique_samples}</td><td className="p-4">{percent(row.accuracy)}</td><td className="p-4">{percent(row.balanced_accuracy)}</td><td className="p-4">{percent(row.precision_occupied)}</td><td className="p-4">{percent(row.recall_occupied)}</td><td className="p-4">{percent(row.f1_occupied)}</td></tr>)}</tbody></table></div></article>
        </div>
      </> : <div className="card border-amber-200 bg-amber-50 p-6 text-sm text-amber-800">Independent benchmark unavailable. Prepare the Benchmark profile and retrain the local model.</div>}
    </section>

    <section className="space-y-5 border-t border-slate-200 pt-8">
      <div><p className="label">Application run diagnostics</p><h2 className="mt-1 text-2xl font-black">{runs.semantic_label}</h2><p className="mt-2 max-w-4xl text-sm text-amber-700">{runs.context_note}</p></div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {[["Agreement", runs.overall.accuracy], ["Occupied precision", runs.overall.precision], ["Occupied recall", runs.overall.recall], ["Vacant specificity", runs.overall.specificity], ["F1 score", runs.overall.f1_score]].map(([label, value]) => <article className="card p-5" key={String(label)}><p className="label">{label}</p><p className="mt-3 text-3xl font-black">{percent(value as number | null)}</p></article>)}
      </div>
      <article className="card p-6"><div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5"><div><p className="label">Prediction-enabled runs</p><p className="mt-2 text-2xl font-black">{runs.prediction_enabled_runs}</p></div><div><p className="label">Stored predictions</p><p className="mt-2 text-2xl font-black">{runs.stored_slot_predictions}</p></div><div><p className="label">Unique scenarios</p><p className="mt-2 text-2xl font-black">{runs.unique_scenarios}</p></div><div><p className="label">Unique scenario slots</p><p className="mt-2 text-2xl font-black">{runs.unique_scenario_slots}</p></div><div><p className="label">Repeated predictions</p><p className="mt-2 text-2xl font-black text-amber-700">{runs.repeated_slot_predictions}</p></div></div><p className="mt-5 text-xs text-slate-500">{runs.legacy_runs} legacy runs excluded from slot agreement.</p></article>
      <article className="card overflow-hidden"><div className="border-b border-slate-100 p-5"><p className="label">Inspection queue</p><h3 className="mt-1 text-lg font-bold">Recent incorrect stored predictions</h3></div>{runs.recent_errors.length ? <div className="divide-y divide-slate-100">{runs.recent_errors.map((row, index) => <div className="flex flex-col justify-between gap-3 p-4 md:flex-row md:items-center" key={`${row.analysis_id}-${row.slot_id}-${index}`}><div><p className="text-sm font-bold">{row.dataset} · slot {row.slot_id}</p><p className="mt-1 text-xs text-slate-500">Predicted {row.predicted_occupied ? "occupied" : "vacant"}; ground truth {row.ground_truth_occupied ? "occupied" : "vacant"}{row.confidence === null ? "" : ` · ${(row.confidence * 100).toFixed(1)}% confidence`}</p></div><Link href={`/history/${row.analysis_id}`} className="rounded-lg bg-slate-900 px-4 py-2 text-center text-xs font-bold text-white">Inspect analysis</Link></div>)}</div> : <div className="p-8 text-center text-sm font-semibold text-emerald-700">No incorrect predictions are present in the stored detailed runs.</div>}</article>
    </section>
  </div>;
}
