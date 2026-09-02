"use client";

import { useEffect, useMemo, useState } from "react";

type Slot = { id: string; occupied: boolean; polygon: number[][] };
type Scenario = {
  id: string; dataset: string; lot: string; condition: string; captured_at: string | null;
  total_spaces: number; occupied_spaces: number; vacant_spaces: number; slots: Slot[];
};
type ScenarioResponse = { count: number; scenarios: Scenario[] };
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

export function ScenarioExplorer() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [dataset, setDataset] = useState("All datasets");
  const [scenarioId, setScenarioId] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${apiBase}/datasets/scenarios?limit=500`)
      .then((response) => {
        if (!response.ok) throw new Error("Could not load prepared scenarios");
        return response.json() as Promise<ScenarioResponse>;
      })
      .then((payload) => {
        setScenarios(payload.scenarios);
        setScenarioId(payload.scenarios[0]?.id ?? "");
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }, []);

  const datasets = useMemo(
    () => ["All datasets", ...Array.from(new Set(scenarios.map((row) => row.dataset)))],
    [scenarios],
  );
  const visible = useMemo(
    () => scenarios.filter((row) => dataset === "All datasets" || row.dataset === dataset),
    [dataset, scenarios],
  );
  const selected = scenarios.find((row) => row.id === scenarioId) ?? visible[0];

  function changeDataset(value: string) {
    setDataset(value);
    const next = scenarios.find((row) => value === "All datasets" || row.dataset === value);
    setScenarioId(next?.id ?? "");
  }

  if (loading) return <div className="card mt-8 p-8 text-sm text-slate-500">Loading catalogue…</div>;
  if (error) return <div className="card mt-8 p-8 text-sm text-red-600">{error}. Is the backend running?</div>;
  if (!selected) return (
    <div className="card mt-8 p-8">
      <h2 className="font-bold text-slate-900">No prepared scenarios yet</h2>
      <p className="mt-2 text-sm text-slate-500">Run <code>scripts\prepare-datasets.ps1 -Profile Demo</code>, then reload this page.</p>
    </div>
  );

  return (
    <div className="mt-8 grid gap-6 xl:grid-cols-[360px_1fr]">
      <aside className="card h-fit p-6">
        <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-black text-amber-700">Dataset ground truth</span>
        <div className="mt-5 space-y-5">
          <label className="block">
            <span className="mb-2 block text-xs font-bold text-slate-500">Dataset</span>
            <select className="w-full rounded-xl border border-slate-200 bg-white px-3 py-3 text-sm" value={dataset} onChange={(event) => changeDataset(event.target.value)}>
              {datasets.map((name) => <option key={name}>{name}</option>)}
            </select>
          </label>
          <label className="block">
            <span className="mb-2 block text-xs font-bold text-slate-500">Scenario</span>
            <select className="w-full rounded-xl border border-slate-200 bg-white px-3 py-3 text-sm" value={selected.id} onChange={(event) => setScenarioId(event.target.value)}>
              {visible.map((row) => <option key={row.id} value={row.id}>{row.lot} · {row.condition}</option>)}
            </select>
          </label>
        </div>
        <dl className="mt-6 space-y-3 border-t border-slate-100 pt-5 text-sm">
          <div className="flex justify-between"><dt className="text-slate-500">Dataset</dt><dd className="font-bold">{selected.dataset}</dd></div>
          <div className="flex justify-between"><dt className="text-slate-500">Parking lot</dt><dd className="font-bold">{selected.lot}</dd></div>
          <div className="flex justify-between"><dt className="text-slate-500">Condition</dt><dd className="font-bold">{selected.condition}</dd></div>
        </dl>
      </aside>
      <section className="card overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-6 py-5">
          <div><p className="font-bold">Parking-lot preview</p><p className="mt-1 text-xs text-slate-500">Normalized annotations over the source image</p></div>
          <div className="flex gap-4 text-xs font-bold"><span className="text-emerald-600">● Vacant</span><span className="text-red-600">● Occupied</span></div>
        </div>
        <div className="relative bg-slate-900">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="block h-auto w-full" src={`${apiBase}/datasets/scenarios/${encodeURIComponent(selected.id)}/image`} alt={`${selected.dataset} parking scenario`} />
          <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Parking-space overlay">
            {selected.slots.map((slot) => <polygon key={slot.id} points={slot.polygon.map(([x, y]) => `${x},${y}`).join(" ")} fill={slot.occupied ? "rgba(239,68,68,.30)" : "rgba(16,185,129,.30)"} stroke={slot.occupied ? "#ef4444" : "#10b981"} strokeWidth="0.004" vectorEffect="non-scaling-stroke" />)}
          </svg>
        </div>
        <div className="grid grid-cols-3 divide-x divide-slate-100">
          {[["Total", selected.total_spaces], ["Vacant", selected.vacant_spaces], ["Occupied", selected.occupied_spaces]].map(([label, value]) => <div className="p-5 text-center" key={label}><p className="text-2xl font-black">{value}</p><p className="mt-1 text-xs font-bold text-slate-400">{label}</p></div>)}
        </div>
      </section>
    </div>
  );
}
