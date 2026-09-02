import { SectionPlaceholder } from "@/components/section-placeholder";

export default function AnalyticsPage() {
  return (
    <SectionPlaceholder
      eyebrow="Insights"
      title="Occupancy analytics"
      description="Compare occupancy rates, dataset conditions, and model processing performance."
    >
      <div className="grid h-64 place-items-center rounded-2xl bg-slate-50">
        <div className="flex h-40 items-end gap-3">
          {[42, 68, 54, 82, 63, 74, 48].map((height, index) => (
            <span
              className="w-8 rounded-t-lg bg-amber-400/70"
              key={index}
              style={{ height: `${height}%` }}
            />
          ))}
        </div>
      </div>
      <p className="mt-3 text-center text-xs font-semibold text-slate-400">
        Illustrative chart — real values will be generated from analysis history.
      </p>
    </SectionPlaceholder>
  );
}
