"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { RetryPanel } from "@/components/retry-panel";
import { apiErrorMessage, apiFetch } from "@/lib/api-client";

type AnalysisRecord = {
  id: number;
  dataset: string;
  scenario_id: string;
  total_spaces: number;
  occupied_spaces: number;
  vacant_spaces: number;
  processing_time_ms: number;
  average_confidence: number | null;
  ground_truth_agreement: number | null;
  source_type: string;
  result_status?: string;
  display_name: string;
  created_at: string;
};
type HistoryResponse = { count: number; analyses: AnalysisRecord[] };

/**
 * What kind of run this row is.
 *
 * History previously showed a diagnostic correction and an ordinary product
 * upload identically, so a hand-adjusted layout was indistinguishable from an
 * automatic one in the audit trail -- exactly the confusion the diagnostic tool
 * is meant to stay clear of.
 */
function runKind(record: { source_type: string; result_status?: string }): { label: string; tone: string } {
  if (record.source_type === "video_upload") return { label: "Video analysis", tone: "bg-indigo-100 text-indigo-800" };
  if (record.source_type === "scenario") return { label: "Benchmark run", tone: "bg-slate-200 text-slate-800" };
  if (record.result_status === "user_verified") return { label: "Diagnostic correction", tone: "bg-amber-100 text-amber-900" };
  return { label: "Product upload", tone: "bg-emerald-100 text-emerald-900" };
}

export function AnalysisHistory() {
  const [records, setRecords] = useState<AnalysisRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    apiFetch<HistoryResponse>("/analysis/history?limit=50")
      .then((payload) => setRecords(payload.analyses))
      .catch((reason: unknown) => setError(apiErrorMessage(reason)))
      .finally(() => setLoading(false));
  }, []);
  if (loading) return <div className="card mt-8 p-6 text-sm text-slate-500">Loading analysis history…</div>;
  if (error) return <RetryPanel message={error} />;
  if (!records.length) return <div className="card mt-8 p-10 text-center"><p className="font-bold">No analyses recorded yet</p><p className="mt-2 text-sm text-slate-500">Run a local AI analysis to create the first record.</p></div>;
  return (
    <div className="card mt-8 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-400"><tr><th className="p-4">Run</th><th className="p-4">Dataset / scenario</th><th className="p-4">Result</th><th className="p-4">Model confidence</th><th className="p-4">Ground-truth agreement</th><th className="p-4">Time</th><th className="p-4">Completed</th><th className="p-4">Detail</th></tr></thead>
          <tbody className="divide-y divide-slate-100">
            {records.map((record) => (
              <tr key={record.id}>
                <td className="p-4 font-black">#{record.id}</td>
                <td className="p-4"><p className="font-bold">{record.dataset}</p><p className="mt-1 max-w-xs truncate text-xs text-slate-500" title={record.scenario_id}>{record.display_name}</p><span className={`mt-2 inline-block rounded-full px-2 py-0.5 text-[10px] font-black uppercase tracking-wider ${runKind(record).tone}`}>{runKind(record).label}</span></td>
                <td className="p-4"><span className="font-bold text-emerald-600">{record.vacant_spaces} vacant</span><span className="mx-2 text-slate-300">/</span><span className="font-bold text-red-600">{record.occupied_spaces} occupied</span></td>
                <td className="p-4 text-xs"><p className="font-bold">{record.average_confidence === null ? "—" : `${(record.average_confidence * 100).toFixed(1)}%`}</p><p className="mt-1 text-slate-500">how sure the model was</p></td>
                <td className="p-4 text-xs">{record.ground_truth_agreement === null ? <><p className="font-bold text-slate-500">Not applicable</p><p className="mt-1 text-slate-500">no verified labels for this source</p></> : <><p className="font-bold">{(record.ground_truth_agreement * 100).toFixed(1)}%</p><p className="mt-1 text-slate-500">vs dataset ground truth</p></>}</td>
                <td className="p-4 font-semibold">{record.processing_time_ms.toFixed(1)} ms</td>
                <td className="p-4 text-slate-500">{new Date(record.created_at).toLocaleString()}</td>
                <td className="p-4"><Link href={`/history/${record.id}`} className="inline-flex min-h-11 items-center justify-center rounded-lg bg-slate-900 px-3 py-2 text-xs font-bold text-white">Inspect</Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
