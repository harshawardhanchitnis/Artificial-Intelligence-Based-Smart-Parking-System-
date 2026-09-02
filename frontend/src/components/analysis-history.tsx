"use client";

import { useEffect, useState } from "react";

type AnalysisRecord = {
  id: number;
  dataset: string;
  scenario_id: string;
  total_spaces: number;
  occupied_spaces: number;
  vacant_spaces: number;
  processing_time_ms: number;
  created_at: string;
};
type HistoryResponse = { count: number; analyses: AnalysisRecord[] };
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

export function AnalysisHistory() {
  const [records, setRecords] = useState<AnalysisRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  useEffect(() => {
    fetch(`${apiBase}/analysis/history?limit=50`)
      .then((response) => response.ok ? response.json() as Promise<HistoryResponse> : Promise.reject())
      .then((payload) => setRecords(payload.analyses))
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);
  if (loading) return <div className="card mt-8 p-6 text-sm text-slate-500">Loading analysis history…</div>;
  if (error) return <div className="card mt-8 p-6 text-sm text-red-600">History API is unavailable.</div>;
  if (!records.length) return <div className="card mt-8 p-10 text-center"><p className="font-bold">No analyses recorded yet</p><p className="mt-2 text-sm text-slate-500">Run a local AI analysis to create the first record.</p></div>;
  return (
    <div className="card mt-8 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-400"><tr><th className="p-4">Run</th><th className="p-4">Dataset / scenario</th><th className="p-4">Result</th><th className="p-4">Time</th><th className="p-4">Completed</th></tr></thead>
          <tbody className="divide-y divide-slate-100">
            {records.map((record) => (
              <tr key={record.id}>
                <td className="p-4 font-black">#{record.id}</td>
                <td className="p-4"><p className="font-bold">{record.dataset}</p><p className="mt-1 max-w-xs truncate text-xs text-slate-400">{record.scenario_id}</p></td>
                <td className="p-4"><span className="font-bold text-emerald-600">{record.vacant_spaces} vacant</span><span className="mx-2 text-slate-300">/</span><span className="font-bold text-red-600">{record.occupied_spaces} occupied</span></td>
                <td className="p-4 font-semibold">{record.processing_time_ms.toFixed(1)} ms</td>
                <td className="p-4 text-slate-500">{new Date(record.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
