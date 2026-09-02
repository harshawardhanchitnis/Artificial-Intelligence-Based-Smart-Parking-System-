import { ReportsPanel } from "@/components/reports-panel";

export default function ReportsPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">Exports</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">Presentation reports</h1>
      <p className="mt-2 text-sm text-slate-500">Export reproducible local inference summaries as CSV or detailed JSON.</p>
      <ReportsPanel />
    </div>
  );
}
