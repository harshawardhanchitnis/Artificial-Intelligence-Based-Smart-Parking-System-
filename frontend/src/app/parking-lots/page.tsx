import { SectionPlaceholder } from "@/components/section-placeholder";

export default function ParkingLotsPage() {
  return (
    <SectionPlaceholder
      eyebrow="Catalogue"
      title="Prepared parking lots"
      description="Browse the lots and camera views imported from the three approved datasets."
    >
      <div className="grid gap-4 md:grid-cols-3">
        {["PKLot", "CNRPark+EXT", "ACPDS"].map((dataset, index) => (
          <article className="rounded-2xl border border-slate-200 p-5" key={dataset}>
            <span className="text-xs font-black text-amber-600">0{index + 1}</span>
            <h2 className="mt-3 font-bold">{dataset}</h2>
            <p className="mt-2 text-sm text-slate-500">
              Catalogue entries will appear after dataset preparation.
            </p>
          </article>
        ))}
      </div>
    </SectionPlaceholder>
  );
}
