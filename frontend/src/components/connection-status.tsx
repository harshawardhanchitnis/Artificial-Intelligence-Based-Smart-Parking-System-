"use client";

import { useEffect, useState } from "react";

import { apiFetch } from "@/lib/api-client";

type Liveness = {
  status: string;
  timestamp: string;
  version: string;
};

type State = "checking" | "ready" | "offline";

export function ConnectionStatus() {
  const [state, setState] = useState<State>("checking");

  useEffect(() => {
    let active = true;

    async function check() {
      try {
        const report = await apiFetch<Liveness>("/health/live", {
          timeoutMs: 3_000,
        });

        if (!active) return;

        setState(report.status === "alive" ? "ready" : "offline");
      } catch {
        if (!active) return;
        setState("offline");
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
    offline: "border-red-200 bg-red-50 text-red-700",
  }[state];

  const label = {
    checking: "Checking local system",
    ready: "Local backend connected",
    offline: "Local backend unavailable",
  }[state];

  return (
    <div
      className={`rounded-full border px-3 py-1.5 text-xs font-bold ${styles}`}
      aria-live="polite"
    >
      {label}
    </div>
  );
}