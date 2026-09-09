import layout from "@/lib/demo-layout.json";

type Slot = { p: number[][]; o: boolean };

/**
 * A static illustration of a real analysed parking layout.
 *
 * The polygons and vacant/occupied states are genuine output of the local
 * pipeline on a prepared PKLot scenario, exported at build time -- not a mock
 * grid, but not live either. It is drawn as a diagram rather than over the
 * source photograph so the page carries no redistributed dataset imagery, and
 * its caption says "static illustration" so a marketing graphic on the landing
 * page can never be mistaken for a measured result. Every figure the product
 * *reports* comes from a live analysis; this is the only static visual, and it
 * is outside Presentation Mode.
 */
export function LayoutVisual({ className = "" }: { className?: string }) {
  const slots = layout.slots as Slot[];
  return (
    <figure className={`m-0 ${className}`}>
      <div className="overflow-hidden rounded-2xl border border-white/15 bg-[#0d1524]">
        <svg
          viewBox="0 0 100 56"
          className="block h-auto w-full"
          role="img"
          aria-label={`Analysed parking layout: ${layout.vacant} vacant and ${layout.occupied} occupied spaces across ${layout.total} detected spaces`}
        >
          <defs>
            <linearGradient id="lot-surface" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#1b2740" />
              <stop offset="100%" stopColor="#0d1524" />
            </linearGradient>
          </defs>
          <rect width="100" height="56" fill="url(#lot-surface)" />
          {[14, 28, 42].map((y) => (
            <line key={y} x1="4" y1={y} x2="96" y2={y} stroke="#ffffff" strokeOpacity="0.05" strokeWidth="0.3" />
          ))}
          {slots.map((slot, index) => (
            <polygon
              key={index}
              points={slot.p.map(([x, y]) => `${x * 100},${y * 56}`).join(" ")}
              fill={slot.o ? "rgba(239,68,68,0.34)" : "rgba(16,185,129,0.30)"}
              stroke={slot.o ? "#f87171" : "#34d399"}
              strokeWidth="0.22"
            />
          ))}
        </svg>
      </div>
      <figcaption className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm font-bold">
        <span className="flex items-center gap-2 text-emerald-300">
          <span className="h-2.5 w-2.5 rounded-sm bg-emerald-400" aria-hidden="true" />
          {layout.vacant} vacant
        </span>
        <span className="flex items-center gap-2 text-red-300">
          <span className="h-2.5 w-2.5 rounded-sm bg-red-400" aria-hidden="true" />
          {layout.occupied} occupied
        </span>
        <span className="text-slate-400">{layout.total} spaces detected</span>
        <span className="ml-auto text-xs font-semibold text-slate-500">
          Static illustration · exported from a real run on {layout.dataset} {layout.lot} · {(layout.agreement * 100).toFixed(0)}% agreement with verified labels
        </span>
      </figcaption>
    </figure>
  );
}
