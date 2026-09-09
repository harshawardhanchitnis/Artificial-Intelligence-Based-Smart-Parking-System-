"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useRef, useState } from "react";

import { Button, LinkButton, StatusBadge } from "@/components/ui";
import { apiErrorMessage, apiFetch, apiUrl } from "@/lib/api-client";

type Point = [number, number];
type Slot = { slot_index: number; polygon: Point[] };
type Layout = { id: number; media_asset_id: number; confidence: number; slots: Slot[] };
type CorrectionResult = { analysis_id: number; result_image_url: string; total_spaces: number };

export function LayoutCorrectionEditor({ layoutId }: { layoutId: string }) {
  const [layout, setLayout] = useState<Layout | null>(null);
  const [original, setOriginal] = useState<Point[][]>([]);
  const [polygons, setPolygons] = useState<Point[][]>([]);
  const [selected, setSelected] = useState(0);
  const [dragging, setDragging] = useState<{ slot: number; corner: number } | null>(null);
  const [result, setResult] = useState<CorrectionResult | null>(null);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    apiFetch<Layout>(`/media/layouts/${layoutId}`)
      .then((value) => {
        const points = value.slots.map((slot) => slot.polygon);
        setLayout(value); setOriginal(points); setPolygons(points);
      })
      .catch((reason: unknown) => setError(apiErrorMessage(reason)));
  }, [layoutId]);

  function pointerPoint(event: React.PointerEvent<SVGSVGElement>): Point {
    const bounds = event.currentTarget.getBoundingClientRect();
    return [
      Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
      Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
    ];
  }

  function move(event: React.PointerEvent<SVGSVGElement>) {
    if (!dragging) return;
    const point = pointerPoint(event);
    setPolygons((current) => current.map((polygon, slot) => slot === dragging.slot ? polygon.map((value, corner) => corner === dragging.corner ? point : value) : polygon));
  }

  function addSlot() {
    setPolygons((current) => [...current, [[0.4, 0.4], [0.55, 0.4], [0.55, 0.58], [0.4, 0.58]]]);
    setSelected(polygons.length);
  }

  function removeSlot() {
    if (!polygons.length) return;
    setPolygons((current) => current.filter((_, index) => index !== selected));
    setSelected((value) => Math.max(0, value - 1));
  }

  async function save() {
    setSaving(true); setError(""); setResult(null);
    try {
      setResult(await apiFetch<CorrectionResult>(`/media/layouts/${layoutId}/corrections`, {
        method: "POST",
        body: JSON.stringify({ slots: polygons.map((polygon) => ({ polygon })), note: "Operator-reviewed advanced layout correction" }),
      }));
    } catch (reason) { setError(apiErrorMessage(reason)); }
    finally { setSaving(false); }
  }

  if (error && !layout) return <div role="alert" className="card mt-8 border-red-200 bg-red-50 p-6 text-red-800">{error}</div>;
  if (!layout) return <div className="card mt-8 p-6 text-slate-600">Loading detected layout…</div>;
  return <div className="mt-8 grid gap-6 xl:grid-cols-[1fr_320px]">
    <section className="card overflow-hidden">
      <div className="relative bg-slate-950">
        <img src={apiUrl(`/media/assets/${layout.media_asset_id}`)} alt="Parking-lot source for advanced layout correction" className="block h-auto w-full" />
        <svg ref={svgRef} viewBox="0 0 1 1" preserveAspectRatio="none" className="absolute inset-0 h-full w-full touch-none" onPointerMove={move} onPointerUp={() => setDragging(null)} onPointerCancel={() => setDragging(null)} aria-label="Editable parking-space polygons">
          {polygons.map((polygon, slotIndex) => <g key={slotIndex} onPointerDown={() => setSelected(slotIndex)}>
            <polygon points={polygon.map(([x, y]) => `${x},${y}`).join(" ")} fill={slotIndex === selected ? "rgba(245,158,11,.30)" : "rgba(59,130,246,.18)"} stroke={slotIndex === selected ? "#fbbf24" : "#60a5fa"} strokeWidth="0.004" vectorEffect="non-scaling-stroke" />
            {slotIndex === selected && polygon.map(([x, y], cornerIndex) => <circle key={cornerIndex} cx={x} cy={y} r="0.012" fill="#fff" stroke="#f59e0b" strokeWidth="0.004" vectorEffect="non-scaling-stroke" onPointerDown={(event) => { event.stopPropagation(); event.currentTarget.setPointerCapture(event.pointerId); setDragging({ slot: slotIndex, corner: cornerIndex }); }} />)}
          </g>)}
        </svg>
      </div>
      <p className="border-t border-slate-200 p-4 text-xs leading-5 text-slate-600">Drag the four visible handles on the selected space. This advanced correction is saved separately and is explicitly tagged as corrected.</p>
    </section>
    <aside className="card h-fit p-6"><StatusBadge tone="warning">Diagnostic tool</StatusBadge><h2 className="mt-4 text-xl font-black">Inspect and adjust detected geometry</h2><p className="mt-2 text-sm leading-6 text-slate-600">This page is <span className="font-bold">not part of the normal workflow</span>. The product establishes parking geometry itself, and reports that it could not rather than asking anyone to draw it. This exists for research, diagnostics and benchmark investigation.</p><p className="mt-5 text-sm font-bold">{polygons.length === 0 ? "No spaces yet — add the first one to begin" : `${polygons.length} spaces · editing #${selected + 1}`}</p><div className="mt-4 grid gap-2"><Button type="button" variant="outline" onClick={addSlot}>Add missed space</Button><Button type="button" variant="danger" onClick={removeSlot} disabled={!polygons.length}>Remove selected space</Button><Button type="button" variant="outline" onClick={() => { setPolygons(original); setSelected(0); }}>Restore automatic layout</Button><Button type="button" onClick={save} disabled={saving || !polygons.length}>{saving ? "Saving corrected analysis…" : "Save corrected analysis"}</Button></div>{error && <p role="alert" className="mt-4 text-sm font-bold text-red-700">{error}</p>}{result && <div className="mt-5 rounded-xl border border-emerald-200 bg-emerald-50 p-4"><p className="font-bold text-emerald-800">Corrected analysis #{result.analysis_id} saved</p><p className="mt-1 text-xs text-emerald-800">{result.total_spaces} reviewed spaces</p><LinkButton href={`/history/${result.analysis_id}`} className="mt-3 w-full">Inspect result</LinkButton></div>}</aside>
  </div>;
}
