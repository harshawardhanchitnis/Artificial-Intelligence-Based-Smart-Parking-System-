"use client";

import { useEffect, useState } from "react";

import { apiErrorMessage, apiFetch } from "@/lib/api-client";

type Check = { key: string; label: string; ready: boolean; detail: string };
type Readiness = {
  ready: boolean;
  status: string;
  timestamp: string;
  checks: Check[];
  summary: { passed: number; total: number };
};
type ModelPart = { ready: boolean; model_name: string; architecture?: string; reason?: string };
type ModelStatus = {
  active_model: ModelPart;
  enhanced_occupancy: ModelPart;
  automatic_localizer: ModelPart;
  reproducible_baseline: ModelPart;
};

export function SystemStatus() {
  const [report, setReport] = useState<Readiness | null>(null);
  const [models, setModels] = useState<ModelStatus | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    Promise.all([
      apiFetch<Readiness>("/health/ready", {
        timeoutMs: 5_000,
        acceptedStatuses: [503],
      }),
      apiFetch<ModelStatus>("/model/status", { timeoutMs: 5_000 }),
    ])
      .then(([readinessPayload, modelPayload]) => {
        if (active) {
          setReport(readinessPayload);
          setModels(modelPayload);
          setError("");
        }
      })
      .catch((reason: unknown) => {
        if (active) setError(apiErrorMessage(reason));
      });
    return () => {
      active = false;
    };
  }, [attempt]);

  if (error) {
    return (
      <div className="rounded-2xl border border-red-200 bg-red-50 p-5" role="alert">
        <p className="text-sm font-semibold text-red-700">{error}</p>
        <button onClick={() => setAttempt((value) => value + 1)} className="mt-3 min-h-11 rounded-lg bg-slate-900 px-4 py-2 text-xs font-bold text-white focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2">Retry checks</button>
      </div>
    );
  }
  if (!report || !models) return <div className="rounded-2xl border border-slate-200 p-5 text-sm text-slate-500">Running system and readiness checks…</div>;

  return (
    <div className="space-y-5">
      <div className="rounded-2xl bg-slate-900 p-5 text-white">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.16em] text-amber-300">Vision system</p>
            <p className="mt-1 text-2xl font-black">AI model readiness</p>
            <p className="mt-2 text-sm text-slate-300">
              Active occupancy model: {models.active_model.model_name}
            </p>
          </div>
          <span className="rounded-full border border-emerald-400/40 bg-emerald-400/10 px-3 py-1 text-xs font-black uppercase tracking-wider text-emerald-300">
            {report.ready ? "Ready" : "Attention"}
          </span>
        </div>
        <div className="mt-4 grid gap-3 border-t border-white/10 pt-4 sm:grid-cols-3">
          <p className="text-xs text-slate-400"><span className="block font-black text-white">{models.enhanced_occupancy.ready ? "Ready" : "Not installed"}</span>Enhanced occupancy</p>
          <p className="text-xs text-slate-400"><span className="block font-black text-white">{models.automatic_localizer.ready ? "Ready" : "Not installed"}</span>Automatic space detection</p>
          <p className="text-xs text-slate-400"><span className="block font-black text-white">{models.reproducible_baseline.ready ? "Available" : "Unavailable"}</span>Reproducible baseline</p>
        </div>
      </div>
      <div className={report.ready ? "rounded-2xl border border-emerald-200 bg-emerald-50 p-5" : "rounded-2xl border border-amber-200 bg-amber-50 p-5"}>
        <p className={report.ready ? "font-black text-emerald-700" : "font-black text-amber-700"}>{report.ready ? "Presentation system ready" : "System needs attention"}</p>
        <p className="mt-1 text-xs text-slate-600">{report.summary.passed}/{report.summary.total} reliability checks passed · checked {new Date(report.timestamp).toLocaleTimeString()}</p>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {report.checks.map((check) => (
          <div className="flex items-start justify-between gap-4 rounded-2xl border border-slate-200 p-4" key={check.key}>
            <div><p className="text-sm font-semibold text-slate-700">{check.label}</p><p className="mt-1 text-xs text-slate-400">{check.detail}</p></div>
            <span className={check.ready ? "text-xs font-black text-emerald-600" : "text-xs font-black text-amber-600"}>{check.ready ? "READY" : "ATTENTION"}</span>
          </div>
        ))}
      </div>
      <button onClick={() => setAttempt((value) => value + 1)} className="min-h-11 rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-bold text-slate-800 focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2">Run checks again</button>
    </div>
  );
}
