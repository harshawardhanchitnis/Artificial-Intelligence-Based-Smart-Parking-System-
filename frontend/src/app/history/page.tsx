import { AnalysisHistory } from "@/components/analysis-history";

export default function HistoryPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">Audit trail</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">Analysis history</h1>
      <p className="mt-2 text-sm text-slate-500">Review local AI occupancy analyses stored in SQLite.</p>
      <AnalysisHistory />
    </div>
  );
}
