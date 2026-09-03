import { DiagnosticsDashboard } from "@/components/diagnostics-dashboard";

export default function DiagnosticsPage() {
  return <div className="mx-auto max-w-7xl"><p className="label">Model operations</p><h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">AI quality diagnostics</h1><p className="mt-2 text-sm text-slate-500">Separate leakage-safe model evaluation from agreement observed in saved application runs.</p><DiagnosticsDashboard /></div>;
}
