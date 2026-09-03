"use client";

import { useEffect, useState } from "react";

import { RetryPanel } from "@/components/retry-panel";
import { apiErrorMessage, apiFetch } from "@/lib/api-client";

type Summary = {
  prepared: boolean;
  scenario_count: number;
  datasets: unknown[];
  analysis_count: number;
  model: { ready: boolean; model_name: string };
  latest_analysis: { dataset: string; vacant_spaces: number; occupied_spaces: number } | null;
};

export function DashboardSummary() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    apiFetch<Summary>("/dashboard/summary")
      .then(setSummary)
      .catch((reason: unknown) => setError(apiErrorMessage(reason)));
  }, []);

  if (error) {
    return <RetryPanel message={error} />;
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
