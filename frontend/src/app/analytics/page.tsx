import { AnalyticsDashboard } from "@/components/analytics-dashboard";

export default function AnalyticsPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">Insights</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">Occupancy analytics</h1>
      <p className="mt-2 text-sm text-slate-500">Compare real saved inference results across the prepared datasets.</p>
      <AnalyticsDashboard />
    </div>
  );
}
