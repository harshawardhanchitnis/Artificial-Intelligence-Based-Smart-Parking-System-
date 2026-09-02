import Link from "next/link";
import { DashboardSummary } from "@/components/dashboard-summary";

const datasets = [
  { name: "PKLot", detail: "Full-lot views, XML slot annotations", color: "bg-amber-500" },
  {
    name: "CNRPark+EXT",
    detail: "Multiple cameras, full images and patches",
    color: "bg-indigo-500",
  },
  { name: "ACPDS", detail: "Quadrilateral ROI annotations", color: "bg-emerald-500" },
];

export default function DashboardPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
        <div>
          <p className="label">Product overview</p>
          <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">
            Parking intelligence dashboard
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            Select a prepared scenario, run local AI analysis, and present vacant and
            occupied parking spaces with clear visual overlays.
          </p>
        </div>
        <Link
          href="/presentation"
          className="rounded-xl bg-amber-500 px-5 py-3 text-center text-sm font-extrabold text-slate-950 shadow-lg shadow-amber-500/20 transition hover:bg-amber-400"
        >
          Start guided presentation →
        </Link>
      </div>

      <DashboardSummary />

      <section className="mt-6 grid gap-6 xl:grid-cols-[1.4fr_1fr]">
        <article className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="label">Dataset catalogue</p>
              <h2 className="mt-1 text-lg font-bold">Approved sources</h2>
            </div>
            <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-bold text-emerald-700">
              3 configured
            </span>
          </div>
          <div className="mt-5 space-y-3">
            {datasets.map((dataset) => (
              <div
                key={dataset.name}
                className="flex items-center gap-4 rounded-2xl border border-slate-100 bg-slate-50/70 p-4"
              >
                <span className={`h-10 w-1.5 rounded-full ${dataset.color}`} />
                <div className="min-w-0 flex-1">
                  <p className="font-bold text-slate-800">{dataset.name}</p>
                  <p className="mt-1 text-xs text-slate-500">{dataset.detail}</p>
                </div>
                <span className="text-xs font-bold text-slate-400">ARCHIVED</span>
              </div>
            ))}
          </div>
        </article>

        <article className="card overflow-hidden">
          <div className="bg-[#111a2e] p-6 text-white">
            <p className="text-xs font-bold uppercase tracking-[0.16em] text-amber-400">
              Presentation flow
            </p>
            <h2 className="mt-2 text-xl font-bold">Designed for a reliable live demo</h2>
          </div>
          <ol className="space-y-4 p-6">
            {[
              "Choose a dataset and prepared parking lot",
              "Select a preloaded scenario image",
              "Run the local occupancy model",
              "Present green vacant and red occupied spaces",
              "Review totals and saved analysis history",
            ].map((step, index) => (
              <li className="flex gap-3 text-sm text-slate-600" key={step}>
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-amber-100 text-xs font-black text-amber-700">
                  {index + 1}
                </span>
                <span className="pt-0.5">{step}</span>
              </li>
            ))}
          </ol>
        </article>
      </section>
    </div>
  );
}
