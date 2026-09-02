import { SectionPlaceholder } from "@/components/section-placeholder";

export default function HistoryPage() {
  return (
    <SectionPlaceholder
      eyebrow="Audit trail"
      title="Analysis history"
      description="Every completed scenario analysis will be saved locally for repeatable demonstrations."
    >
      <div className="rounded-2xl border border-dashed border-slate-300 py-16 text-center">
        <p className="font-bold text-slate-700">No analyses recorded yet</p>
        <p className="mt-2 text-sm text-slate-500">
          Results will appear here after the inference milestone.
        </p>
      </div>
    </SectionPlaceholder>
  );
}
