"use client";

import { useEffect, useState } from "react";

type Dataset = { name: string; scenario_count: number; lots: string[]; conditions: string[] };
type Catalogue = { prepared: boolean; scenario_count: number; datasets: Dataset[] };
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

export function DatasetCatalogue() {
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    fetch(`${apiBase}/datasets/catalogue`)
      .then((response) => response.ok ? response.json() as Promise<Catalogue> : Promise.reject())
      .then(setCatalogue)
      .catch(() => setError(true));
  }, []);
  if (error) return <div className="card mt-8 p-6 text-sm text-red-600">Catalogue API is unavailable.</div>;
  if (!catalogue) return <div className="card mt-8 p-6 text-sm text-slate-500">Loading catalogue…</div>;
  if (!catalogue.prepared) return <div className="card mt-8 p-6 text-sm text-slate-500">Prepare the Demo profile to populate this page.</div>;
  return (
    <div className="mt-8 grid gap-4 md:grid-cols-3">
      {catalogue.datasets.map((dataset, index) => (
        <article className="card p-5" key={dataset.name}>
          <div className="flex items-center justify-between"><span className="text-xs font-black text-amber-600">0{index + 1}</span><span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold">{dataset.scenario_count} scenarios</span></div>
          <h2 className="mt-4 text-lg font-black">{dataset.name}</h2>
          <p className="mt-3 text-xs font-bold uppercase tracking-wide text-slate-400">Lots / cameras</p>
          <p className="mt-1 text-sm text-slate-600">{dataset.lots.join(", ")}</p>
          <p className="mt-3 text-xs font-bold uppercase tracking-wide text-slate-400">Conditions</p>
          <p className="mt-1 text-sm text-slate-600">{dataset.conditions.join(", ")}</p>
        </article>
      ))}
    </div>
  );
}
