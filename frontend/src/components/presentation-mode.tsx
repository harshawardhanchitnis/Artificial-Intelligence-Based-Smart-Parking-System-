"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useMemo, useState } from "react";

import { apiErrorMessage, apiFetch, apiUrl } from "@/lib/api-client";
import { GEOMETRY_CANDIDATES, GEOMETRY_SOURCE, HONEST_LIMITS, PIPELINE, VERIFIED_FIGURES } from "@/lib/architecture";

type Check = { key: string; label: string; ready: boolean; detail: string };
type Readiness = { ready: boolean; mode: string; checks: Check[]; showcase_count: number };
type Slot = { id: string; occupied: boolean; polygon: number[][] };
type Scenario = { id: string; dataset: string; lot: string; condition: string; total_spaces: number; occupied_spaces: number; vacant_spaces: number; slots: Slot[] };
type Analysis = { analysis_id: number; scenario_id: string; dataset: string; lot?: string; condition?: string; model_name: string; decision_threshold?: number; total_spaces: number; occupied_spaces: number; vacant_spaces: number; uncertain_spaces?: number; ground_truth_agreement: number; ground_truth_occupied_spaces?: number; ground_truth_vacant_spaces?: number; disagreeing_spaces?: number; average_confidence: number; processing_time_ms: number; predictions: Array<{ id: string; polygon: number[][]; predicted_occupied: boolean }> };
type Showcase = { count: number; scenarios: Scenario[] };


function ArchitectureSections({ modelNames }: { modelNames: string[] }) {
  return (
    <>
      <section className="card p-7">
        <p className="label">The current pipeline</p>
        <h2 className="mt-1 text-2xl font-black text-slate-950">From a frame to an auditable occupancy record</h2>
        <ol className="mt-6 space-y-4">
          {PIPELINE.map((stage, index) => (
            <li key={stage.step} className="flex gap-4">
              <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-slate-900 text-xs font-black text-white">{index + 1}</span>
              <div><p className="font-bold text-slate-900">{stage.step}</p><p className="mt-1 text-sm leading-6 text-slate-600">{stage.detail}</p></div>
            </li>
          ))}
        </ol>
        {modelNames.length > 0 && <p className="mt-6 border-t border-slate-100 pt-4 text-xs font-semibold text-slate-500">Models reported ready by this running instance: {modelNames.join(" · ")}</p>}
      </section>

      <section className="card p-7">
        <p className="label">Parking-space geometry</p>
        <h2 className="mt-1 text-2xl font-black text-slate-950">Four independent corners, chosen against three alternatives</h2>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">A rotated rectangle cannot describe a bay seen through perspective. Each bay is therefore predicted as four independently regressed corners, and the representation was chosen by measurement rather than assumption.</p>
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[46rem] border-collapse text-sm">
            <thead><tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wider text-slate-500">
              <th className="py-2 pr-4 font-bold">Candidate</th><th className="py-2 pr-4 font-bold">Family</th>
              <th className="py-2 pr-4 text-right font-bold">ACPDS r@50</th><th className="py-2 pr-4 text-right font-bold">ACPDS r@75</th>
              <th className="py-2 pr-4 text-right font-bold">Corner error</th><th className="py-2 font-bold">Deployment cost</th>
            </tr></thead>
            <tbody>
              {GEOMETRY_CANDIDATES.map((row) => (
                <tr key={row.name} className={row.selected ? "border-b border-slate-100 bg-amber-50 font-bold" : "border-b border-slate-100"}>
                  <td className="py-2 pr-4">{row.name}{row.selected && <span className="ml-2 rounded-full bg-amber-500 px-2 py-0.5 text-[10px] font-black text-slate-950">SHIPPED</span>}</td>
                  <td className="py-2 pr-4 text-slate-600">{row.family}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{row.acpdsRecall50}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{row.acpdsRecall75}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{row.cornerError}</td>
                  <td className="py-2 text-slate-600">{row.cost}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-xs leading-5 text-slate-500">{GEOMETRY_SOURCE}</p>
      </section>

      <section className="card p-7">
        <p className="label">Vehicle intelligence</p>
        <h2 className="mt-1 text-2xl font-black text-slate-950">Exactly three public classes</h2>
        <div className="mt-5 flex flex-wrap gap-3">
          {["CAR", "TWO_WHEELER", "TRUCK"].map((name) => <span key={name} className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-black text-white">{name}</span>)}
        </div>
        <p className="mt-5 max-w-3xl text-sm leading-6 text-slate-600">The whole frame is searched for vehicles independently of the bay map. A source label outside these three is never renamed into them — a bus is not a truck — but a vehicle-shaped object carrying such a label is kept as internal occupancy evidence, because it still fills the bay it stands in. It is drawn on the overlay without a class name and never reaches a public count.</p>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">Vehicle detection does not depend on the geometry succeeding. When no layout can be established the system still reports what it saw, and says plainly that spatial categorisation is unknown rather than showing zeros.</p>
      </section>

      <section className="card p-7">
        <p className="label">Measured, with sources</p>
        <h2 className="mt-1 text-2xl font-black text-slate-950">Every figure traceable to its artifact</h2>
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          {VERIFIED_FIGURES.map((figure) => (
            <div key={figure.label} className="rounded-2xl border border-slate-200 p-5">
              <p className="text-2xl font-black text-slate-950">{figure.value}</p>
              <p className="mt-1 text-sm font-bold text-slate-700">{figure.label}</p>
              <p className="mt-2 text-xs leading-5 text-slate-500">{figure.source}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="card border-amber-200 bg-amber-50 p-7">
        <p className="label text-amber-800">Stated plainly</p>
        <h2 className="mt-1 text-2xl font-black text-slate-950">What this system does not do well</h2>
        <ul className="mt-5 space-y-4">
          {HONEST_LIMITS.map((limit) => <li key={limit.slice(0, 40)} className="text-sm leading-6 text-slate-800">{limit}</li>)}
        </ul>
      </section>
    </>
  );
}

export function PresentationMode() {
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [results, setResults] = useState<Analysis[]>([]);
  const [activeId, setActiveId] = useState("");
  const [runningId, setRunningId] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      apiFetch<Readiness>("/demo/readiness"),
      apiFetch<Showcase>("/demo/showcase"),
    ])
      .then(([nextReadiness, showcase]) => {
        setReadiness(nextReadiness);
        setScenarios(showcase.scenarios);
      })
      .catch((reason: unknown) => setError(apiErrorMessage(reason)));
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
        const result = await apiFetch<Analysis>(`/analysis/scenarios/${encodeURIComponent(scenario.id)}`, { method: "POST", timeoutMs: 120_000 });
        setResults((current) => [...current, result]);
        setActiveId(result.scenario_id);
      } catch (reason) {
        setError(apiErrorMessage(reason));
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
  if (!readiness) return <div className="card p-8 text-sm text-slate-600">Checking presentation readiness…</div>;

  return (
    <div className="space-y-6">
      <section className="overflow-hidden rounded-3xl bg-[#111a2e] text-white shadow-xl">
        <div className="flex flex-col justify-between gap-6 p-7 lg:flex-row lg:items-center">
          <div><p className="text-xs font-black uppercase tracking-[0.18em] text-amber-300">Guided product showcase</p><h2 className="mt-2 text-3xl font-black">Three datasets. One AI workflow.</h2><p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">Run one deterministic, preloaded scenario from PKLot, CNRPark+EXT, and ACPDS. Every result is stored in normal analysis history.</p></div>
          <div className="flex flex-wrap gap-3"><button onClick={enterFullscreen} className="min-h-11 rounded-xl border border-white/20 px-4 py-3 text-sm font-bold hover:bg-white/10 focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2">Toggle full screen</button>{results.length ? <button onClick={reset} className="min-h-11 rounded-xl bg-white/10 px-4 py-3 text-sm font-bold focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2">Reset showcase</button> : <button onClick={runShowcase} disabled={!readiness.ready || Boolean(runningId)} className="min-h-11 rounded-xl bg-amber-500 px-5 py-3 text-sm font-black text-slate-950 focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-700">{runningId ? `Analysing ${scenarios.find((row) => row.id === runningId)?.dataset}…` : "Run guided showcase"}</button>}</div>
        </div>
        <div className="grid border-t border-white/10 sm:grid-cols-2 lg:grid-cols-5">
          {readiness.checks.map((check) => <div className="border-white/10 p-4 lg:border-r" key={check.key}><p className={check.ready ? "text-xs font-black text-emerald-400" : "text-xs font-black text-red-400"}>{check.ready ? "● READY" : "● ATTENTION"}</p><p className="mt-1 text-sm font-bold">{check.label}</p><p className="mt-1 text-xs text-slate-400">{check.detail}</p></div>)}
        </div>
      </section>

      {error && <p className="rounded-2xl bg-red-50 p-4 text-sm font-semibold text-red-700">{error}</p>}

      {!results.length ? (
        <section className="grid gap-5 lg:grid-cols-3">
          {scenarios.map((scenario, index) => <article className="card overflow-hidden" key={scenario.id}><div className="relative aspect-[16/9] bg-slate-900"><img className="h-full w-full object-cover" src={apiUrl(`/datasets/scenarios/${encodeURIComponent(scenario.id)}/image`)} alt={`${scenario.dataset} showcase scenario`} /><span className="absolute left-4 top-4 rounded-full bg-slate-950/80 px-3 py-1 text-xs font-black text-white">{index + 1} · {scenario.dataset}</span></div><div className="p-5"><p className="font-bold">{scenario.lot}</p><p className="mt-1 text-xs text-slate-500">{scenario.condition} · {scenario.total_spaces} annotated spaces</p><div className="mt-4 flex gap-4 text-xs font-bold"><span className="text-emerald-600">{scenario.vacant_spaces} vacant truth</span><span className="text-red-600">{scenario.occupied_spaces} occupied truth</span></div></div></article>)}
        </section>
      ) : activeResult && activeScenario ? (
        <section className="card overflow-hidden">
          <div className="flex flex-col justify-between gap-4 border-b border-slate-100 p-5 md:flex-row md:items-center"><div><p className="label">AI showcase result</p><h2 className="mt-1 text-xl font-black">{activeResult.dataset} · {activeScenario.lot} · {activeScenario.condition}</h2><p className="mt-1 font-mono text-xs text-slate-500" title="Run this exact identifier under Analyse images to reproduce these numbers">{activeResult.scenario_id}</p></div><div className="flex flex-wrap gap-2" role="tablist" aria-label="Showcase results">{results.map((result) => <button role="tab" aria-selected={result.scenario_id === activeResult.scenario_id} key={result.scenario_id} onClick={() => setActiveId(result.scenario_id)} className={result.scenario_id === activeResult.scenario_id ? "min-h-11 rounded-lg bg-slate-900 px-4 py-2 text-xs font-black text-white focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2" : "min-h-11 rounded-lg bg-slate-100 px-4 py-2 text-xs font-bold text-slate-700 focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2"}>{result.dataset}</button>)}</div></div>
          <div className="grid xl:grid-cols-[1.55fr_1fr]">
            <div className="relative bg-slate-900"><img className="block h-auto w-full" src={apiUrl(`/datasets/scenarios/${encodeURIComponent(activeScenario.id)}/image`)} alt={`${activeScenario.dataset} AI result`} /><svg className="absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="AI prediction overlay">{activeResult.predictions.map((slot) => <polygon key={slot.id} points={slot.polygon.map(([x, y]) => `${x},${y}`).join(" ")} fill={slot.predicted_occupied ? "rgba(239,68,68,.30)" : "rgba(16,185,129,.30)"} stroke={slot.predicted_occupied ? "#ef4444" : "#10b981"} strokeWidth="0.004" vectorEffect="non-scaling-stroke" />)}</svg></div>
            <div className="p-7"><span className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-black text-indigo-700">LOCAL AI PREDICTION</span><div className="mt-6 grid grid-cols-3 gap-3">{[["Total", activeResult.total_spaces, "text-slate-900"], ["Vacant", activeResult.vacant_spaces, "text-emerald-600"], ["Occupied", activeResult.occupied_spaces, "text-red-600"]].map(([label, value, color]) => <div className="rounded-2xl bg-slate-50 p-4 text-center" key={label}><p className={`text-2xl font-black ${color}`}>{value}</p><p className="mt-1 text-xs font-bold text-slate-400">{label}</p></div>)}</div><dl className="mt-6 space-y-4 text-sm">{typeof activeResult.ground_truth_vacant_spaces === "number" && <ResultRow label="Verified labels" value={`${activeResult.ground_truth_vacant_spaces} vacant · ${activeResult.ground_truth_occupied_spaces} occupied`} />}<ResultRow label="Average confidence" value={`${(activeResult.average_confidence * 100).toFixed(1)}%`} /><ResultRow label="Ground-truth agreement" value={`${(activeResult.ground_truth_agreement * 100).toFixed(1)}%${typeof activeResult.disagreeing_spaces === "number" ? ` — ${activeResult.disagreeing_spaces} disagreeing` : ""}`} /><ResultRow label="Model" value={activeResult.model_name} /><ResultRow label="Processing time" value={`${activeResult.processing_time_ms.toFixed(1)} ms`} /><ResultRow label="Saved analysis" value={`#${activeResult.analysis_id}`} /></dl><a href={apiUrl(`/reports/analyses/${activeResult.analysis_id}.json`)} className="mt-6 block rounded-xl bg-slate-900 px-4 py-3 text-center text-sm font-black text-white">Download detailed result</a></div>
          </div>
          <div className="flex flex-col justify-between gap-3 border-t border-slate-100 bg-slate-50 p-5 text-sm md:flex-row md:items-center"><p className="font-semibold text-slate-600">Completed {results.length}/{scenarios.length} showcase analyses</p>{runningId && <p className="font-black text-amber-600">Local inference in progress…</p>}{results.length === scenarios.length && <p className="font-black text-emerald-600">✓ Showcase complete — open Analytics for comparison</p>}</div>
        </section>
      ) : null}

      <ArchitectureSections modelNames={readiness.checks.filter((check) => check.ready && /model|detector|classifier|fusion/i.test(check.label)).map((check) => check.label)} />
    </div>
  );
}

function ResultRow({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between border-b border-slate-100 pb-3"><dt className="text-slate-500">{label}</dt><dd className="font-black">{value}</dd></div>;
}
