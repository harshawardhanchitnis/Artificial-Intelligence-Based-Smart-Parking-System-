"use client";

import { useEffect, useMemo, useState } from "react";

import { RetryPanel } from "@/components/retry-panel";
import { apiErrorMessage, apiFetch, apiUrl } from "@/lib/api-client";

type Slot = { id: string; occupied: boolean; polygon: number[][] };
type Scenario = {
  id: string;
  dataset: string;
  lot: string;
  condition: string;
  total_spaces: number;
  occupied_spaces: number;
  vacant_spaces: number;
  slots: Slot[];
};
type Prediction = {
  id: string;
  polygon: number[][];
  predicted_occupied: boolean;
  occupied_probability: number;
  confidence: number;
  correct: boolean;
};
type Analysis = {
  analysis_id: number;
  model_name: string;
  total_spaces: number;
  occupied_spaces: number;
  vacant_spaces: number;
  ground_truth_agreement: number;
  average_confidence: number;
  processing_time_ms: number;
  predictions: Prediction[];
};
type ScenarioResponse = { count: number; scenarios: Scenario[] };

export function ScenarioExplorer() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [dataset, setDataset] = useState("All datasets");
  const [scenarioId, setScenarioId] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [view, setView] = useState<"ground-truth" | "prediction">("ground-truth");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    apiFetch<ScenarioResponse>("/datasets/scenarios?limit=500")
      .then((payload) => {
        setScenarios(payload.scenarios);
        setScenarioId(payload.scenarios[0]?.id ?? "");
      })
      .catch((reason: unknown) => setError(apiErrorMessage(reason)))
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

  function chooseScenario(nextId: string) {
    setScenarioId(nextId);
    setAnalysis(null);
    setView("ground-truth");
    setError("");
  }

  function changeDataset(value: string) {
    setDataset(value);
    const next = scenarios.find((row) => value === "All datasets" || row.dataset === value);
    chooseScenario(next?.id ?? "");
  }

  async function runAnalysis() {
    if (!selected) return;
    setRunning(true);
    setError("");
    try {
      const payload = await apiFetch<Analysis>(
        `/analysis/scenarios/${encodeURIComponent(selected.id)}`,
        { method: "POST", timeoutMs: 120_000 },
      );
      setAnalysis(payload);
      setView("prediction");
    } catch (reason) {
      setError(apiErrorMessage(reason));
    } finally {
      setRunning(false);
    }
  }

  if (loading) return <div className="card mt-8 p-8 text-sm text-slate-500">Loading catalogue…</div>;
  if (error && !scenarios.length) return <RetryPanel message={error} />;
  if (!selected) return <div className="card mt-8 p-8 text-sm text-slate-500">Prepare the Demo profile before training the model.</div>;

  const predictionView = view === "prediction" && analysis;
  const overlaySlots = predictionView
    ? analysis.predictions.map((slot) => ({
        id: slot.id,
        polygon: slot.polygon,
        occupied: slot.predicted_occupied,
      }))
    : selected.slots;
  const totals = predictionView ? analysis : selected;

  return (
    <div className="mt-8 grid gap-6 xl:grid-cols-[360px_1fr]">
      <aside className="card h-fit p-6">
        <span className={predictionView ? "rounded-full bg-indigo-100 px-3 py-1 text-xs font-black text-indigo-700" : "rounded-full bg-amber-100 px-3 py-1 text-xs font-black text-amber-700"}>
          {predictionView ? "Local AI prediction" : "Dataset ground truth"}
        </span>
        <div className="mt-5 space-y-5">
          <label className="block">
            <span className="mb-2 block text-xs font-bold text-slate-500">Dataset</span>
            <select className="w-full rounded-xl border border-slate-200 bg-white px-3 py-3 text-sm" value={dataset} onChange={(event) => changeDataset(event.target.value)}>
              {datasets.map((name) => <option key={name}>{name}</option>)}
            </select>
          </label>
          <label className="block">
            <span className="mb-2 block text-xs font-bold text-slate-500">Scenario</span>
            <select className="w-full rounded-xl border border-slate-200 bg-white px-3 py-3 text-sm" value={selected.id} onChange={(event) => chooseScenario(event.target.value)}>
              {visible.map((row) => <option key={row.id} value={row.id}>{row.lot} · {row.condition}</option>)}
            </select>
          </label>
        </div>
        <dl className="mt-6 space-y-3 border-t border-slate-100 pt-5 text-sm">
          <div className="flex justify-between"><dt className="text-slate-500">Dataset</dt><dd className="font-bold">{selected.dataset}</dd></div>
          <div className="flex justify-between"><dt className="text-slate-500">Parking lot</dt><dd className="font-bold">{selected.lot}</dd></div>
          <div className="flex justify-between"><dt className="text-slate-500">Condition</dt><dd className="font-bold">{selected.condition}</dd></div>
        </dl>
        <button onClick={runAnalysis} disabled={running} className="mt-6 w-full rounded-xl bg-slate-900 px-4 py-3 text-sm font-black text-white disabled:cursor-wait disabled:opacity-60">
          {running ? "Running local inference…" : "Run local AI analysis"}
        </button>
        {error && <p className="mt-3 rounded-xl bg-red-50 p-3 text-xs font-semibold text-red-700">{error}</p>}
        {analysis && (
          <div className="mt-4 grid grid-cols-2 gap-2 rounded-xl bg-slate-50 p-2">
            <button onClick={() => setView("ground-truth")} className={view === "ground-truth" ? "rounded-lg bg-white px-2 py-2 text-xs font-black shadow-sm" : "px-2 py-2 text-xs font-bold text-slate-500"}>Ground truth</button>
            <button onClick={() => setView("prediction")} className={view === "prediction" ? "rounded-lg bg-white px-2 py-2 text-xs font-black shadow-sm" : "px-2 py-2 text-xs font-bold text-slate-500"}>AI prediction</button>
          </div>
        )}
      </aside>

      <section className="card overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-6 py-5">
          <div><p className="font-bold">Parking-lot analysis</p><p className="mt-1 text-xs text-slate-500">{predictionView ? analysis.model_name : "Verified dataset annotations"}</p></div>
          <div className="flex gap-4 text-xs font-bold"><span className="text-emerald-600">● Vacant</span><span className="text-red-600">● Occupied</span></div>
        </div>
        <div className="relative bg-slate-900">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="block h-auto w-full" src={apiUrl(`/datasets/scenarios/${encodeURIComponent(selected.id)}/image`)} alt={`${selected.dataset} parking scenario`} />
          <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Parking-space overlay">
            {overlaySlots.map((slot) => <polygon key={slot.id} points={slot.polygon.map(([x, y]) => `${x},${y}`).join(" ")} fill={slot.occupied ? "rgba(239,68,68,.30)" : "rgba(16,185,129,.30)"} stroke={slot.occupied ? "#ef4444" : "#10b981"} strokeWidth="0.004" vectorEffect="non-scaling-stroke" />)}
          </svg>
        </div>
        <div className="grid grid-cols-3 divide-x divide-slate-100">
          {[["Total", totals.total_spaces], ["Vacant", totals.vacant_spaces], ["Occupied", totals.occupied_spaces]].map(([label, value]) => <div className="p-5 text-center" key={label}><p className="text-2xl font-black">{value}</p><p className="mt-1 text-xs font-bold text-slate-400">{label}</p></div>)}
        </div>
        {analysis && (
          <div className="grid gap-3 border-t border-slate-100 bg-slate-50 p-5 sm:grid-cols-3">
            <Metric label="Ground-truth agreement" value={`${(analysis.ground_truth_agreement * 100).toFixed(1)}%`} />
            <Metric label="Average confidence" value={`${(analysis.average_confidence * 100).toFixed(1)}%`} />
            <Metric label="Processing time" value={`${analysis.processing_time_ms.toFixed(1)} ms`} />
          </div>
        )}
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl bg-white p-3 text-center"><p className="text-lg font-black text-slate-900">{value}</p><p className="mt-1 text-xs font-bold text-slate-400">{label}</p></div>;
}
