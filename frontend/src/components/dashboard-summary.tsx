"use client";

import { useEffect, useState } from "react";

type Summary = {
  prepared: boolean;
  scenario_count: number;
  datasets: unknown[];
  analysis_count: number;
  model: { ready: boolean; model_name: string };
  latest_analysis: { dataset: string; vacant_spaces: number; occupied_spaces: number } | null;
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

export function DashboardSummary() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/dashboard/summary`)
      .then((response) => response.ok ? response.json() as Promise<Summary> : Promise.reject())
      .then(setSummary)
      .catch(() => setError(true));
  }, []);

  if (error) {
    return <div className="card mt-8 p-5 text-sm font-semibold text-red-600">Dashboard API is unavailable. Start the backend and refresh.</div>;
  }

  const metrics = [
    { label: "Prepared scenarios", value: summary?.scenario_count ?? "—", note: summary?.prepared ? "Demo catalogue ready" : "Preparing catalogue" },
    { label: "Local AI model", value: summary?.model.ready ? "Ready" : summary ? "Not ready" : "—", note: summary?.model.model_name ?? "Checking model" },
    { label: "Analyses saved", value: summary?.analysis_count ?? "—", note: "Persistent SQLite audit trail" },
    { label: "Latest result", value: summary?.latest_analysis ? `${summary.latest_analysis.vacant_spaces} free` : "—", note: summary?.latest_analysis ? `${summary.latest_analysis.dataset} · ${summary.latest_analysis.occupied_spaces} occupied` : "Run the first analysis" },
  ];

  return (
    <section className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label="Live system summary">
      {metrics.map((metric) => (
        <article className="card p-5" key={metric.label}>
          <p className="label">{metric.label}</p>
          <p className="mt-3 text-3xl font-black tracking-tight text-slate-900">{metric.value}</p>
          <p className="mt-2 truncate text-xs text-slate-500">{metric.note}</p>
        </article>
      ))}
    </section>
  );
}
