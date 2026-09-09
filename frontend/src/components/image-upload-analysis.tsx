"use client";

/* eslint-disable @next/next/no-img-element */

import { useState } from "react";

import { Button, LinkButton, StatusBadge } from "@/components/ui";
import { apiErrorMessage, apiFetch, apiUrl } from "@/lib/api-client";

type UploadResult = {
  status: string;
  verification_state: string;
  /** The vocabulary the product speaks; internal states never reach the screen. */
  product_state?: "LAYOUT_ESTABLISHED" | "AUTOMATIC_LAYOUT_UNRESOLVED" | "INSUFFICIENT_VISUAL_EVIDENCE";
  geometry_source: string;
  message?: string;
  analysis_id?: number;
  layout_id: number;
  total_spaces?: number;
  occupied_spaces?: number;
  vacant_spaces?: number;
  uncertain_spaces?: number;
  detected_spaces?: number;
  localization_confidence: number;
  average_confidence?: number;
  processing_time_ms?: number;
  result_image_url?: string;
  analysis_mode?: "product" | "benchmark";
  vehicle_counts?: { CAR: number; TWO_WHEELER: number; TRUCK: number };
  unmapped_vehicles?: number;
  vehicle_detection_available?: boolean;
  unclassified_objects?: number;
  vehicles_outside_parking_area?: number;
  parking_area_known?: boolean;
  candidate_spaces?: number;
  /** Why a vehicle count is what it is; absent when the detector did not run. */
  scene_diagnostics?: {
    confident_objects: number;
    vehicle_shaped_objects: number;
    top_non_vehicle_labels: string[];
  } | null;
};

/** The only vehicle names the interface may show. */
const VEHICLE_LABELS: Array<[keyof NonNullable<UploadResult["vehicle_counts"]>, string]> = [
  ["CAR", "Cars"],
  ["TWO_WHEELER", "Two-wheelers"],
  ["TRUCK", "Trucks"],
];

/** How each product state is presented. Internal names never appear here. */
const LAYOUT_STATES: Record<string, { tone: "success" | "warning" | "neutral"; label: string; note: string }> = {
  LAYOUT_ESTABLISHED: { tone: "success", label: "Layout established", note: "The system established this camera's parking geometry itself and measured occupancy against it." },
  AUTOMATIC_LAYOUT_UNRESOLVED: { tone: "warning", label: "Layout not established", note: "The parking geometry could not be read reliably from this scene, so no occupancy is asserted for it." },
  INSUFFICIENT_VISUAL_EVIDENCE: { tone: "warning", label: "No parking spaces found", note: "Nothing in this image resembled a parking layout." },
};

function stateOf(result: UploadResult) {
  return LAYOUT_STATES[result.product_state ?? ""] ?? LAYOUT_STATES.AUTOMATIC_LAYOUT_UNRESOLVED;
}


/**
 * Explains the overlay, because a reader cannot judge a detection product from
 * totals alone. Bays carry a verdict colour; vehicles are unfilled boxes with a
 * class chip, so the two layers stay tellable apart.
 */
function OverlayLegend({ candidates = false }: { candidates?: boolean }) {
  const keys: Array<[string, string]> = candidates
    ? [["#64748b", "Candidate space — proposed, no verdict"], ["#38bdf8", "Vehicle, with class and confidence"], ["#94a3b8", "Vehicle-shaped object, no supported class"]]
    : [["#16a34a", "Vacant space"], ["#dc2626", "Occupied space"], ["#d97706", "Uncertain space"], ["#38bdf8", "Vehicle in a detected space"], ["#a855f7", "Vehicle in the facility, unmapped"], ["#94a3b8", "Vehicle-shaped object, no supported class"]];
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-2 border-t border-slate-200 px-5 py-3 text-xs font-semibold text-slate-600">
      {keys.map(([colour, label]) => (
        <li key={label} className="flex items-center gap-2">
          <span aria-hidden="true" className="inline-block h-3 w-3 rounded-sm border-2" style={{ borderColor: colour }} />
          {label}
        </li>
      ))}
    </ul>
  );
}

/** A measured reason for a zero vehicle count, never a guess. */
function VehicleEvidenceNote({ result }: { result: UploadResult }) {
  const counts = result.vehicle_counts;
  const seen = (counts?.CAR ?? 0) + (counts?.TWO_WHEELER ?? 0) + (counts?.TRUCK ?? 0);
  const diagnostics = result.scene_diagnostics;
  if (seen > 0) {
    return <p className="mt-3 text-xs leading-5 text-slate-600">Vehicle detection runs on the whole frame and does not depend on the parking-space map, so these {seen} vehicles were found even though the bays were not.</p>;
  }
  if (diagnostics && diagnostics.confident_objects > 0 && diagnostics.vehicle_shaped_objects === 0) {
    return <p className="mt-3 text-xs leading-5 text-slate-600">The full-frame detector was confident about {diagnostics.confident_objects} object{diagnostics.confident_objects === 1 ? "" : "s"} here, and classified none of them as a vehicle{diagnostics.top_non_vehicle_labels.length > 0 ? ` (its strongest labels were ${diagnostics.top_non_vehicle_labels.slice(0, 2).join(" and ")})` : ""}. That pattern is characteristic of a near-vertical aerial viewpoint, which this detector reads poorly — so treat this as “the detector could not read the vehicles”, not as “the car park is empty”.</p>;
  }
  return <p className="mt-3 text-xs leading-5 text-slate-600">The full-frame detector found no vehicle-shaped object in this scene.</p>;
}

export function ImageUploadAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) return;
    setRunning(true);
    setError("");
    setResult(null);
    const data = new FormData();
    data.append("file", file);
    try {
      setResult(await apiFetch<UploadResult>("/media/images/analyse", { method: "POST", body: data, timeoutMs: 180_000 }));
    } catch (reason) {
      setError(apiErrorMessage(reason));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="mt-8 grid gap-6 xl:grid-cols-[360px_1fr]">
      <form className="card h-fit p-6" onSubmit={submit}>
        <p className="label">Automatic workflow</p>
        <h2 className="mt-2 text-xl font-black text-slate-950">Upload a parking-lot image</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">The AI detects parking-space geometry before determining occupancy. No slot coordinates are required.</p>
        <label className="mt-6 block">
          <span className="mb-2 block text-sm font-bold text-slate-800">Image file</span>
          <input type="file" accept="image/jpeg,image/png,image/webp" required onChange={(event) => setFile(event.target.files?.[0] ?? null)} className="block min-h-11 w-full rounded-xl border border-slate-300 bg-white p-2 text-sm text-slate-800 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-900 file:px-3 file:py-2 file:font-bold file:text-white" />
        </label>
        <Button className="mt-5 w-full" disabled={!file || running} type="submit">{running ? "Detecting spaces and occupancy…" : "Analyse image automatically"}</Button>
        <p className="mt-3 text-xs leading-5 text-slate-600">JPEG, PNG, or WebP · up to 25 MB · parking-lot domain only</p>
        {error && <div role="alert" className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm font-semibold text-red-800">{error}</div>}
      </form>

      <section className="card min-h-96 overflow-hidden" aria-live="polite">
        {!result && <div className="grid min-h-96 place-items-center p-8 text-center"><div><p className="text-lg font-bold text-slate-800">Your analysed image will appear here</p><p className="mt-2 text-sm text-slate-600">Automatic localisation confidence is checked before occupancy is reported.</p></div></div>}
        {result && result.status !== "success" && (() => {
          const state = stateOf(result);
          return <>
            <div className="border-b border-slate-200 p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <StatusBadge tone="warning">{state.label}</StatusBadge>
                {typeof result.candidate_spaces === "number" && <p className="text-sm font-bold text-slate-700">{result.candidate_spaces} candidate space{result.candidate_spaces === 1 ? "" : "s"}</p>}
              </div>
              <h3 className="mt-3 text-xl font-black text-slate-950">The parking layout could not be determined reliably</h3>
              <p className="mt-2 text-sm leading-6 text-slate-600">{result.message}</p>
            </div>
            {result.result_image_url && <img src={apiUrl(result.result_image_url)} alt="Scene with candidate parking spaces and any detected vehicles annotated" className="block h-auto w-full" />}
            {result.result_image_url && <OverlayLegend candidates />}
            {result.vehicle_detection_available && <div className="border-t border-slate-200 p-5">
              <p className="text-xs font-extrabold uppercase tracking-[.18em] text-slate-500">Vehicles found in the whole scene</p>
              <div className="mt-3 flex flex-wrap gap-x-8 gap-y-3">
                {VEHICLE_LABELS.map(([key, label]) => <div key={key}><p className="text-xl font-black text-slate-950">{result.vehicle_counts?.[key] ?? 0}</p><p className="text-xs font-bold text-slate-600">{label}</p></div>)}
              </div>
              <VehicleEvidenceNote result={result} />
              <p className="mt-2 text-xs leading-5 text-slate-500">Because no parking layout was established, the parking area itself is unknown: these vehicles cannot be sorted into in-space, in-area or off-site, and none is described as such.</p>
              {Boolean(result.unclassified_objects) && <p className="mt-2 text-xs leading-5 text-slate-500">{result.unclassified_objects} vehicle-shaped object{result.unclassified_objects === 1 ? "" : "s"} could not be placed in a supported class. {result.unclassified_objects === 1 ? "It is" : "They are"} kept as occupancy evidence but never reported as a vehicle type.</p>}
            </div>}
            <div className="border-t border-slate-200 bg-slate-50 p-5">
              <p className="text-sm font-bold text-slate-700">No occupancy figures are shown, so nothing here can be mistaken for a measured result.</p>
              <LinkButton href={`/analyse/correct/${result.layout_id}`} variant="outline" className="mt-4">Advanced: inspect the geometry by hand</LinkButton>
            </div>
          </>;
        })()}
        {result?.analysis_id && (() => { const state = stateOf(result); return <><div className="border-b border-slate-200 p-5"><div className="flex flex-wrap items-center justify-between gap-3"><StatusBadge tone={state.tone}>{state.label}</StatusBadge><p className="text-sm font-bold text-slate-700">Saved analysis #{result.analysis_id}</p></div><p className="mt-2 text-sm leading-6 text-slate-600">{state.note}</p><dl className="mt-4 flex flex-wrap gap-x-8 gap-y-2 text-xs"><div><dt className="font-bold text-slate-600">Layout confidence</dt><dd className="mt-0.5 text-slate-500">{(result.localization_confidence * 100).toFixed(1)}% — certainty about the geometry</dd></div><div><dt className="font-bold text-slate-600">Occupancy confidence</dt><dd className="mt-0.5 text-slate-500">{((result.average_confidence ?? 0) * 100).toFixed(1)}% — mean per-space certainty, not accuracy</dd></div></dl></div><img src={apiUrl(result.result_image_url ?? "")} alt="Analysed parking lot with numbered occupancy overlays and labelled vehicle detections" className="block h-auto w-full" /><OverlayLegend /><div className={`grid divide-x divide-slate-200 ${result.uncertain_spaces ? "grid-cols-4" : "grid-cols-3"}`}>{([["Total", result.total_spaces], ["Vacant", result.vacant_spaces], ["Occupied", result.occupied_spaces], ...(result.uncertain_spaces ? [["Uncertain", result.uncertain_spaces] as [string, number]] : [])] as Array<[string, number | undefined]>).map(([label, value]) => <div key={label} className="p-5 text-center"><p className="text-2xl font-black text-slate-950">{value}</p><p className="mt-1 text-xs font-bold text-slate-600">{label}</p></div>)}</div>{Boolean(result.uncertain_spaces) && <p className="border-t border-slate-200 bg-amber-50 px-5 py-3 text-xs font-semibold text-amber-900">{result.uncertain_spaces} space{result.uncertain_spaces === 1 ? "" : "s"} fell inside the calibrated ambiguous range, so no vacant or occupied verdict is asserted for {result.uncertain_spaces === 1 ? "it" : "them"}.</p>}{result.vehicle_detection_available && result.vehicle_counts && <div className="border-t border-slate-200 p-5"><p className="text-xs font-extrabold uppercase tracking-[.18em] text-slate-500">Vehicles found in the whole scene</p><div className="mt-3 flex flex-wrap gap-x-8 gap-y-3">{VEHICLE_LABELS.map(([key, label]) => <div key={key}><p className="text-xl font-black text-slate-950">{result.vehicle_counts?.[key] ?? 0}</p><p className="text-xs font-bold text-slate-600">{label}</p></div>)}<div><p className="text-xl font-black text-slate-950">{result.unmapped_vehicles ?? 0}</p><p className="text-xs font-bold text-slate-600">Unmapped</p></div></div><p className="mt-3 text-xs leading-5 text-slate-600">These counts describe the parking facility. Vehicles are found across the entire image, independently of the parking-space map; an <span className="font-bold">unmapped</span> vehicle is inside the facility but in no detected space — parked in an aisle, or in a space that was not detected.</p>{typeof result.vehicles_outside_parking_area === "number" && <p className="mt-2 text-xs leading-5 text-slate-500">{result.vehicles_outside_parking_area} further vehicle{result.vehicles_outside_parking_area === 1 ? " was" : "s were"} seen outside the inferred parking area — passing traffic — and {result.vehicles_outside_parking_area === 1 ? "is" : "are"} excluded from these figures.</p>}{Boolean(result.unclassified_objects) && <p className="mt-2 text-xs leading-5 text-slate-500">{result.unclassified_objects} object{result.unclassified_objects === 1 ? "" : "s"} could not be placed in a supported vehicle class. {result.unclassified_objects === 1 ? "It counts" : "They count"} towards occupancy but {result.unclassified_objects === 1 ? "is" : "are"} not reported as a vehicle type.</p>}</div>}</>; })()}
      </section>
    </div>
  );
}
