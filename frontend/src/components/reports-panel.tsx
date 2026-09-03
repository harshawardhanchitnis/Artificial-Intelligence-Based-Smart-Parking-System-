"use client";

import { useEffect, useState } from "react";

import { RetryPanel } from "@/components/retry-panel";
import { apiErrorMessage, apiFetch, apiUrl } from "@/lib/api-client";

type Record = { id: number; dataset: string; scenario_id: string; total_spaces: number; occupied_spaces: number; vacant_spaces: number; created_at: string };
type History = { count: number; analyses: Record[] };
export function ReportsPanel() {
  const [records, setRecords] = useState<Record[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    apiFetch<History>("/analysis/history?limit=50")
      .then((payload) => setRecords(payload.analyses))
      .catch((reason: unknown) => setError(apiErrorMessage(reason)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="card mt-8 p-6 text-sm text-slate-500">Loading report catalogue…</div>;
  if (error) return <RetryPanel message={error} />;

  return (
    <div className="mt-8 space-y-6">
      <article className="card flex flex-col items-start justify-between gap-5 bg-[#111a2e] p-6 text-white md:flex-row md:items-center">
        <div><p className="font-bold">Complete analysis history</p><p className="mt-2 text-sm text-slate-400">Download a spreadsheet-ready CSV generated from the current SQLite records.</p></div>
        <a href={apiUrl("/reports/analyses.csv")} className="rounded-xl bg-amber-500 px-5 py-3 text-sm font-black text-slate-950">Download history CSV</a>
      </article>
      <article className="card overflow-hidden">
        <div className="border-b border-slate-100 p-5"><p className="label">Individual runs</p><h2 className="mt-1 font-bold">Detailed inference exports</h2></div>
        {!records.length ? <div className="p-10 text-center text-sm text-slate-500">Run an analysis to make individual exports available.</div> : (
          <div className="divide-y divide-slate-100">
            {records.map((record) => <div className="flex flex-col justify-between gap-4 p-5 md:flex-row md:items-center" key={record.id}><div><p className="font-bold">Run #{record.id} · {record.dataset}</p><p className="mt-1 max-w-xl truncate text-xs text-slate-500">{record.scenario_id} · {record.vacant_spaces} vacant / {record.occupied_spaces} occupied</p></div><div className="flex gap-2"><a className="rounded-lg border border-slate-200 px-4 py-2 text-xs font-bold hover:bg-slate-50" href={apiUrl(`/reports/analyses/${record.id}.csv`)}>CSV</a><a className="rounded-lg bg-slate-900 px-4 py-2 text-xs font-bold text-white" href={apiUrl(`/reports/analyses/${record.id}.json`)}>JSON + slots</a></div></div>)}
          </div>
        )}
      </article>
      <p className="text-xs text-slate-500">Exports are generated on demand. No report files or dataset images are copied into Git.</p>
    </div>
  );
}
