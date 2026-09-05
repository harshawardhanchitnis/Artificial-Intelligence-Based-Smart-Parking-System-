"use client";

import { useEffect, useState } from "react";

import { apiFetch } from "@/lib/api-client";

type Readiness = { ready: boolean; summary: { passed: number; total: number } };
type State = "checking" | "ready" | "degraded" | "offline";

export function ConnectionStatus() {
  const [state, setState] = useState<State>("checking");
  const [summary, setSummary] = useState("");

  useEffect(() => {
    let active = true;
    async function check() {
      try {
        const report = await apiFetch<Readiness>("/health/ready", {
          timeoutMs: 4_000,
          acceptedStatuses: [503],
        });
        if (active) {
          setState(report.ready ? "ready" : "degraded");
          setSummary(`${report.summary.passed}/${report.summary.total} checks`);
        }
      } catch {
        if (!active) return;
        setState("offline");
        setSummary("");
      }
    }
    void check();
    const interval = window.setInterval(check, 15_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  const styles = {
    checking: "border-slate-200 bg-slate-50 text-slate-500",
    ready: "border-emerald-200 bg-emerald-50 text-emerald-700",
    degraded: "border-amber-200 bg-amber-50 text-amber-700",
    offline: "border-red-200 bg-red-50 text-red-700",
  }[state];
  const label = {
    checking: "Checking local system",
    ready: "Local system ready",
    degraded: "Local system needs attention",
    offline: "Local backend unavailable",
  }[state];

  return (
    <div className={`rounded-full border px-3 py-1.5 text-xs font-bold ${styles}`} title={summary} aria-live="polite">
      {label}
    </div>
  );
}
