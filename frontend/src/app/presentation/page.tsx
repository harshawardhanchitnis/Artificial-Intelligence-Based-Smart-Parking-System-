import { PresentationMode } from "@/components/presentation-mode";
import { LinkButton } from "@/components/ui";

export default function PresentationPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">Presentation mode</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">Live product showcase</h1>
      <p className="mb-8 mt-2 text-sm text-slate-600">Preflight the application, run a curated scenario from every approved dataset, and present the AI results from one screen.</p>
      <PresentationMode />
      <section className="mt-8 grid gap-5 md:grid-cols-2" aria-label="Extended product demonstrations">
        <article className="card p-6"><p className="label">Automatic layout localisation</p><h2 className="mt-2 text-xl font-black">Analyse a supported parking-lot image</h2><p className="mt-2 text-sm leading-6 text-slate-600">Demonstrate the validated fixed-camera workflow: automatic space polygons, perspective-correct occupancy, confidence checks, overlays, and persistence.</p><LinkButton href="/analyse" className="mt-5">Open image analysis</LinkButton></article>
        <article className="card p-6"><p className="label">Temporal occupancy intelligence</p><h2 className="mt-2 text-xl font-black">Run a prepared or uploaded video</h2><p className="mt-2 text-sm leading-6 text-slate-600">Use a scientifically labelled fixed-camera time-lapse or an MP4/AVI upload to show layout reuse, stability checks, debouncing, playback, and a timeline.</p><LinkButton href="/video" className="mt-5">Open video analysis</LinkButton></article>
      </section>
    </div>
  );
}
