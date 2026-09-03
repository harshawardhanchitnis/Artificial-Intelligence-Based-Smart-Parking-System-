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

export function SystemStatus() {
  const [report, setReport] = useState<Readiness | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    apiFetch<Readiness>("/health/ready", { timeoutMs: 5_000, acceptedStatuses: [503] })
      .then((payload) => {
        if (active) {
          setReport(payload);
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
        <button onClick={() => setAttempt((value) => value + 1)} className="mt-3 rounded-lg bg-slate-900 px-4 py-2 text-xs font-bold text-white">Retry checks</button>
      </div>
    );
  }
  if (!report) return <div className="rounded-2xl border border-slate-200 p-5 text-sm text-slate-500">Running deep readiness checks…</div>;

  return (
    <div className="space-y-5">
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
      <button onClick={() => setAttempt((value) => value + 1)} className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-xs font-bold text-slate-700">Run checks again</button>
    </div>
  );
}
