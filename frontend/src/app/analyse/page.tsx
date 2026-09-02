import { ScenarioExplorer } from "@/components/scenario-explorer";

export default function AnalysePage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">Analysis workspace</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">Analyse a prepared parking scenario</h1>
      <p className="mt-2 max-w-3xl text-sm text-slate-500">
        Select an offline sample, run the trained local classifier, and compare its
        occupancy predictions with the verified dataset labels.
      </p>
      <ScenarioExplorer />
    </div>
  );
}
