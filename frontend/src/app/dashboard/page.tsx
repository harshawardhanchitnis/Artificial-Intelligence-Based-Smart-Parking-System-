import { DashboardSummary } from "@/components/dashboard-summary";
import { LinkButton, StatusBadge } from "@/components/ui";

const datasets = [
  ["PKLot", "Three sites with sunny, cloudy, and rainy full-lot scenes"],
  ["CNRPark+EXT", "Nine cameras with full images and verified occupancy labels"],
  ["ACPDS", "Quadrilateral parking-space annotations including difficult lighting"],
];

export default function DashboardPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
        <div>
          <p className="label">Operational overview</p>
          <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-950">Parking intelligence dashboard</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">Analyse prepared evidence, upload a parking-lot image, or process fixed-camera video with automatic space localisation.</p>
        </div>
        <div className="flex flex-wrap gap-3">
          <LinkButton href="/analyse">Analyse an image</LinkButton>
          <LinkButton href="/presentation" variant="primary">Start presentation</LinkButton>
        </div>
      </div>
      <DashboardSummary />
      <section className="mt-6 grid gap-6 xl:grid-cols-[1.3fr_1fr]">
        <article className="card p-6">
          <div className="flex items-center justify-between gap-4">
            <div><p className="label">Dataset evidence</p><h2 className="mt-1 text-lg font-bold">Approved parking sources</h2></div>
            <StatusBadge tone="success">3 verified</StatusBadge>
          </div>
          <div className="mt-5 space-y-3">
            {datasets.map(([name, detail]) => (
              <div key={name} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <p className="font-bold text-slate-900">{name}</p><p className="mt-1 text-sm text-slate-600">{detail}</p>
              </div>
            ))}
          </div>
        </article>
        <article className="card overflow-hidden">
          <div className="bg-slate-900 p-6 text-white"><p className="text-xs font-bold uppercase tracking-[.16em] text-amber-300">How to begin</p><h2 className="mt-2 text-xl font-bold">Choose a workflow</h2></div>
          <div className="space-y-3 p-6 text-sm text-slate-700">
            <p><strong>Prepared analysis:</strong> repeatable, ground-truth-backed scenarios.</p>
            <p><strong>Image upload:</strong> automatic layout and occupancy prediction.</p>
            <p><strong>Video:</strong> fixed-camera tracking with a saved occupancy timeline.</p>
          </div>
        </article>
      </section>
    </div>
  );
}
