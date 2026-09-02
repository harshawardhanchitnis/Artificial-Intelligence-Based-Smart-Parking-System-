import { PresentationMode } from "@/components/presentation-mode";

export default function PresentationPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">Presentation mode</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">Live product showcase</h1>
      <p className="mb-8 mt-2 text-sm text-slate-500">Preflight the offline system, run a curated scenario from every approved dataset, and present the AI results from one screen.</p>
      <PresentationMode />
    </div>
  );
}
