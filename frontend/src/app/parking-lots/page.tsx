import { DatasetCatalogue } from "@/components/dataset-catalogue";

export default function ParkingLotsPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">Catalogue</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">Prepared parking lots</h1>
      <p className="mt-2 text-sm text-slate-500">Browse offline demo scenarios from the three approved datasets.</p>
      <DatasetCatalogue />
    </div>
  );
}
