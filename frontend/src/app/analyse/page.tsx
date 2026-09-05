import { ImageUploadAnalysis } from "@/components/image-upload-analysis";
import { ScenarioExplorer } from "@/components/scenario-explorer";

export default function AnalysePage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">Analysis workspace</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">Analyse parking-lot images</h1>
      <p className="mt-2 max-w-3xl text-sm text-slate-600">
        Upload a parking-lot image for fully automatic space localisation and occupancy analysis.
      </p>
      <ImageUploadAnalysis />
      <section className="mt-14 border-t border-slate-300 pt-10">
        <p className="label">Prepared evidence</p>
        <h2 className="mt-2 text-2xl font-black text-slate-950">Analyse a ground-truth scenario</h2>
        <p className="mt-2 max-w-3xl text-sm text-slate-600">Use the curated catalogue for repeatable evaluation against verified dataset labels.</p>
        <ScenarioExplorer />
      </section>
    </div>
  );
}
