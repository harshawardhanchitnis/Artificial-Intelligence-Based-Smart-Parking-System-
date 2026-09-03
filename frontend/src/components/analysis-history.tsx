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
  created_at: string;
};
type HistoryResponse = { count: number; analyses: AnalysisRecord[] };
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
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-400"><tr><th className="p-4">Run</th><th className="p-4">Dataset / scenario</th><th className="p-4">Result</th><th className="p-4">AI quality</th><th className="p-4">Time</th><th className="p-4">Completed</th><th className="p-4">Detail</th></tr></thead>
          <tbody className="divide-y divide-slate-100">
            {records.map((record) => (
              <tr key={record.id}>
                <td className="p-4 font-black">#{record.id}</td>
                <td className="p-4"><p className="font-bold">{record.dataset}</p><p className="mt-1 max-w-xs truncate text-xs text-slate-400">{record.scenario_id}</p></td>
                <td className="p-4"><span className="font-bold text-emerald-600">{record.vacant_spaces} vacant</span><span className="mx-2 text-slate-300">/</span><span className="font-bold text-red-600">{record.occupied_spaces} occupied</span></td>
                <td className="p-4 text-xs"><p className="font-bold">{record.average_confidence === null ? "Legacy run" : `${(record.average_confidence * 100).toFixed(1)}% confidence`}</p><p className="mt-1 text-slate-400">{record.ground_truth_agreement === null ? "No saved metric" : `${(record.ground_truth_agreement * 100).toFixed(1)}% agreement`}</p></td>
                <td className="p-4 font-semibold">{record.processing_time_ms.toFixed(1)} ms</td>
                <td className="p-4 text-slate-500">{new Date(record.created_at).toLocaleString()}</td>
                <td className="p-4"><Link href={`/history/${record.id}`} className="rounded-lg bg-slate-900 px-3 py-2 text-xs font-bold text-white">Inspect</Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
