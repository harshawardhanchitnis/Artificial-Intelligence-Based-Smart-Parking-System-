"use client";

import { useEffect, useState } from "react";

type Health = {
  status: string;
  database: string;
  dataset_root_exists: boolean;
  demo_catalogue_exists: boolean;
};

const apiBase =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

export function SystemStatus() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/health`)
      .then((response) => {
        if (!response.ok) throw new Error("Backend health request failed");
        return response.json() as Promise<Health>;
      })
      .then((payload) => {
        setHealth(payload);
        setError(false);
      })
      .catch(() => setError(true));
  }, []);

  const checks: Array<[string, string, boolean]> = [
    ["Frontend", "Ready", true],
    [
      "Backend API",
      health?.status ?? (error ? "Not running" : "Checking"),
      health?.status === "healthy",
    ],
    ["SQLite", health?.database ?? "Checking", health?.database === "connected"],
    [
      "External datasets",
      health ? (health.dataset_root_exists ? "Folder found" : "Folder missing") : "Checking",
      health?.dataset_root_exists === true,
    ],
    [
      "Demo catalogue",
      health ? (health.demo_catalogue_exists ? "Prepared" : "Not prepared") : "Checking",
      health?.demo_catalogue_exists === true,
    ],
  ];

  return (
    <div className="grid gap-3 md:grid-cols-2">
      {checks.map(([label, value, good]) => (
        <div
          className="flex items-center justify-between rounded-2xl border border-slate-200 p-4"
          key={label}
        >
          <span className="text-sm font-semibold text-slate-600">{label}</span>
          <span
            className={
              good
                ? "text-xs font-black text-emerald-600"
                : "text-xs font-black text-slate-400"
            }
          >
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}
