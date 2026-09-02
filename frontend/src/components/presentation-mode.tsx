"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useMemo, useState } from "react";

type Check = { key: string; label: string; ready: boolean; detail: string };
type Readiness = { ready: boolean; mode: string; checks: Check[]; showcase_count: number };
type Slot = { id: string; occupied: boolean; polygon: number[][] };
type Scenario = { id: string; dataset: string; lot: string; condition: string; total_spaces: number; occupied_spaces: number; vacant_spaces: number; slots: Slot[] };
type Analysis = { analysis_id: number; scenario_id: string; dataset: string; model_name: string; total_spaces: number; occupied_spaces: number; vacant_spaces: number; ground_truth_agreement: number; average_confidence: number; processing_time_ms: number; predictions: Array<{ id: string; polygon: number[][]; predicted_occupied: boolean }> };
type Showcase = { count: number; scenarios: Scenario[] };

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

export function PresentationMode() {
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [results, setResults] = useState<Analysis[]>([]);
  const [activeId, setActiveId] = useState("");
  const [runningId, setRunningId] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(`${apiBase}/demo/readiness`).then((response) => response.ok ? response.json() as Promise<Readiness> : Promise.reject()),
      fetch(`${apiBase}/demo/showcase`).then((response) => response.ok ? response.json() as Promise<Showcase> : Promise.reject()),
    ])
      .then(([nextReadiness, showcase]) => {
        setReadiness(nextReadiness);
        setScenarios(showcase.scenarios);
      })
      .catch(() => setError("The local demo APIs are unavailable. Start the backend and refresh."));
  }, []);

  const activeResult = results.find((result) => result.scenario_id === activeId) ?? results.at(-1);
  const activeScenario = useMemo(
    () => scenarios.find((scenario) => scenario.id === activeResult?.scenario_id),
    [activeResult, scenarios],
  );

  async function runShowcase() {
    if (!readiness?.ready || runningId) return;
    setResults([]);
    setActiveId("");
    setError("");
    for (const scenario of scenarios) {
      setRunningId(scenario.id);
      try {
        const response = await fetch(`${apiBase}/analysis/scenarios/${encodeURIComponent(scenario.id)}`, { method: "POST" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail ?? "Local inference failed");
        const result = payload as Analysis;
        setResults((current) => [...current, result]);
        setActiveId(result.scenario_id);
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Local inference failed");
        break;
      }
    }
    setRunningId("");
  }

  function reset() {
    setResults([]);
    setActiveId("");
    setError("");
  }

  async function enterFullscreen() {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
    else await document.exitFullscreen();
  }

  if (error && !readiness) return <div className="card p-8 text-sm font-semibold text-red-600">{error}</div>;
  if (!readiness) return <div className="card p-8 text-sm text-slate-500">Checking offline demo readiness…</div>;

  return (
    <div className="space-y-6">
      <section className="overflow-hidden rounded-3xl bg-[#111a2e] text-white shadow-xl">
        <div className="flex flex-col justify-between gap-6 p-7 lg:flex-row lg:items-center">
          <div><p className="text-xs font-black uppercase tracking-[0.18em] text-amber-400">Guided offline showcase</p><h2 className="mt-2 text-3xl font-black">Three datasets. One local AI workflow.</h2><p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">Run one deterministic, preloaded scenario from PKLot, CNRPark+EXT, and ACPDS. Every result is stored in normal analysis history.</p></div>
          <div className="flex flex-wrap gap-3"><button onClick={enterFullscreen} className="rounded-xl border border-white/20 px-4 py-3 text-sm font-bold hover:bg-white/10">Toggle full screen</button>{results.length ? <button onClick={reset} className="rounded-xl bg-white/10 px-4 py-3 text-sm font-bold">Reset showcase</button> : <button onClick={runShowcase} disabled={!readiness.ready || Boolean(runningId)} className="rounded-xl bg-amber-500 px-5 py-3 text-sm font-black text-slate-950 disabled:cursor-not-allowed disabled:opacity-50">{runningId ? `Analysing ${scenarios.find((row) => row.id === runningId)?.dataset}…` : "Run guided showcase"}</button>}</div>
        </div>
        <div className="grid border-t border-white/10 sm:grid-cols-2 lg:grid-cols-5">
          {readiness.checks.map((check) => <div className="border-white/10 p-4 lg:border-r" key={check.key}><p className={check.ready ? "text-xs font-black text-emerald-400" : "text-xs font-black text-red-400"}>{check.ready ? "● READY" : "● ATTENTION"}</p><p className="mt-1 text-sm font-bold">{check.label}</p><p className="mt-1 text-xs text-slate-400">{check.detail}</p></div>)}
        </div>
      </section>

      {error && <p className="rounded-2xl bg-red-50 p-4 text-sm font-semibold text-red-700">{error}</p>}

      {!results.length ? (
        <section className="grid gap-5 lg:grid-cols-3">
          {scenarios.map((scenario, index) => <article className="card overflow-hidden" key={scenario.id}><div className="relative aspect-[16/9] bg-slate-900"><img className="h-full w-full object-cover" src={`${apiBase}/datasets/scenarios/${encodeURIComponent(scenario.id)}/image`} alt={`${scenario.dataset} showcase scenario`} /><span className="absolute left-4 top-4 rounded-full bg-slate-950/80 px-3 py-1 text-xs font-black text-white">{index + 1} · {scenario.dataset}</span></div><div className="p-5"><p className="font-bold">{scenario.lot}</p><p className="mt-1 text-xs text-slate-500">{scenario.condition} · {scenario.total_spaces} annotated spaces</p><div className="mt-4 flex gap-4 text-xs font-bold"><span className="text-emerald-600">{scenario.vacant_spaces} vacant truth</span><span className="text-red-600">{scenario.occupied_spaces} occupied truth</span></div></div></article>)}
        </section>
      ) : activeResult && activeScenario ? (
        <section className="card overflow-hidden">
          <div className="flex flex-col justify-between gap-4 border-b border-slate-100 p-5 md:flex-row md:items-center"><div><p className="label">AI showcase result</p><h2 className="mt-1 text-xl font-black">{activeResult.dataset} · {activeScenario.lot}</h2></div><div className="flex flex-wrap gap-2">{results.map((result) => <button key={result.scenario_id} onClick={() => setActiveId(result.scenario_id)} className={result.scenario_id === activeResult.scenario_id ? "rounded-lg bg-slate-900 px-4 py-2 text-xs font-black text-white" : "rounded-lg bg-slate-100 px-4 py-2 text-xs font-bold text-slate-600"}>{result.dataset}</button>)}</div></div>
          <div className="grid xl:grid-cols-[1.55fr_1fr]">
            <div className="relative bg-slate-900"><img className="block h-auto w-full" src={`${apiBase}/datasets/scenarios/${encodeURIComponent(activeScenario.id)}/image`} alt={`${activeScenario.dataset} AI result`} /><svg className="absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="AI prediction overlay">{activeResult.predictions.map((slot) => <polygon key={slot.id} points={slot.polygon.map(([x, y]) => `${x},${y}`).join(" ")} fill={slot.predicted_occupied ? "rgba(239,68,68,.30)" : "rgba(16,185,129,.30)"} stroke={slot.predicted_occupied ? "#ef4444" : "#10b981"} strokeWidth="0.004" vectorEffect="non-scaling-stroke" />)}</svg></div>
            <div className="p-7"><span className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-black text-indigo-700">LOCAL AI PREDICTION</span><div className="mt-6 grid grid-cols-3 gap-3">{[["Total", activeResult.total_spaces, "text-slate-900"], ["Vacant", activeResult.vacant_spaces, "text-emerald-600"], ["Occupied", activeResult.occupied_spaces, "text-red-600"]].map(([label, value, color]) => <div className="rounded-2xl bg-slate-50 p-4 text-center" key={label}><p className={`text-2xl font-black ${color}`}>{value}</p><p className="mt-1 text-xs font-bold text-slate-400">{label}</p></div>)}</div><dl className="mt-6 space-y-4 text-sm"><ResultRow label="Average confidence" value={`${(activeResult.average_confidence * 100).toFixed(1)}%`} /><ResultRow label="Ground-truth agreement" value={`${(activeResult.ground_truth_agreement * 100).toFixed(1)}%`} /><ResultRow label="Processing time" value={`${activeResult.processing_time_ms.toFixed(1)} ms`} /><ResultRow label="Saved analysis" value={`#${activeResult.analysis_id}`} /></dl><a href={`${apiBase}/reports/analyses/${activeResult.analysis_id}.json`} className="mt-6 block rounded-xl bg-slate-900 px-4 py-3 text-center text-sm font-black text-white">Download detailed result</a></div>
          </div>
          <div className="flex flex-col justify-between gap-3 border-t border-slate-100 bg-slate-50 p-5 text-sm md:flex-row md:items-center"><p className="font-semibold text-slate-600">Completed {results.length}/{scenarios.length} showcase analyses</p>{runningId && <p className="font-black text-amber-600">Local inference in progress…</p>}{results.length === scenarios.length && <p className="font-black text-emerald-600">✓ Showcase complete — open Analytics for comparison</p>}</div>
        </section>
      ) : null}
    </div>
  );
}

function ResultRow({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between border-b border-slate-100 pb-3"><dt className="text-slate-500">{label}</dt><dd className="font-black">{value}</dd></div>;
}
