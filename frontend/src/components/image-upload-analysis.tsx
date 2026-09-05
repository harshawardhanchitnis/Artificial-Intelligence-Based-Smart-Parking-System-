"use client";

/* eslint-disable @next/next/no-img-element */

import { useState } from "react";

import { Button, LinkButton, StatusBadge } from "@/components/ui";
import { apiErrorMessage, apiFetch, apiUrl } from "@/lib/api-client";

type UploadResult = {
  status: "success" | "success_with_warnings" | "unsupported_layout";
  message?: string;
  analysis_id?: number;
  layout_id: number;
  total_spaces?: number;
  occupied_spaces?: number;
  vacant_spaces?: number;
  localization_confidence: number;
  average_confidence?: number;
  processing_time_ms?: number;
  result_image_url?: string;
};

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
        {result?.status === "unsupported_layout" && <div className="grid min-h-96 place-items-center p-8 text-center"><div><StatusBadge tone="warning">Insufficient localisation confidence</StatusBadge><h3 className="mt-4 text-2xl font-black text-slate-950">A trustworthy layout was not found</h3><p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-slate-600">{result.message} Try a clear fixed-camera parking-lot view. Advanced layout correction is available as an exceptional fallback.</p><p className="mt-4 font-bold">Localisation confidence {(result.localization_confidence * 100).toFixed(1)}%</p><LinkButton href={`/analyse/correct/${result.layout_id}`} variant="outline" className="mt-5">Open advanced calibration</LinkButton></div></div>}
        {result?.analysis_id && <><div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 p-5"><div><StatusBadge tone={result.status === "success" ? "success" : "warning"}>{result.status === "success" ? "Automatic analysis complete" : "Complete with localisation warning"}</StatusBadge><p className="mt-2 text-sm text-slate-600">Layout confidence {(result.localization_confidence * 100).toFixed(1)}% · Occupancy confidence {((result.average_confidence ?? 0) * 100).toFixed(1)}%</p></div><p className="text-sm font-bold text-slate-700">Saved analysis #{result.analysis_id}</p></div><img src={apiUrl(result.result_image_url ?? "")} alt="Analysed parking lot with numbered red occupied and green vacant spaces" className="block h-auto w-full" /><div className="grid grid-cols-3 divide-x divide-slate-200">{[["Total", result.total_spaces], ["Vacant", result.vacant_spaces], ["Occupied", result.occupied_spaces]].map(([label, value]) => <div key={label} className="p-5 text-center"><p className="text-2xl font-black text-slate-950">{value}</p><p className="mt-1 text-xs font-bold text-slate-600">{label}</p></div>)}</div></>}
      </section>
    </div>
  );
}
