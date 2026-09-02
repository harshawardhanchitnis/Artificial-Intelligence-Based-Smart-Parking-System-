import { SectionPlaceholder } from "@/components/section-placeholder";

export default function ReportsPage() {
  return (
    <SectionPlaceholder
      eyebrow="Exports"
      title="Presentation reports"
      description="Generate concise local reports containing annotated images and occupancy summaries."
    >
      <div className="flex flex-col items-start justify-between gap-5 rounded-2xl bg-[#111a2e] p-6 text-white md:flex-row md:items-center">
        <div>
          <p className="font-bold">Analysis summary report</p>
          <p className="mt-2 text-sm text-slate-400">
            PDF export becomes available after analysis history is implemented.
          </p>
        </div>
        <button
          disabled
          className="cursor-not-allowed rounded-xl bg-white/10 px-5 py-3 text-sm font-bold text-slate-400"
        >
          Export unavailable
        </button>
      </div>
    </SectionPlaceholder>
  );
}
