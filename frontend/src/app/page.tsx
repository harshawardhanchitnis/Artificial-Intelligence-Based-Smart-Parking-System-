import Link from "next/link";

import { LinkButton } from "@/components/ui";
import { PRODUCT_NAME } from "@/lib/product";

const capabilities = [
  ["Automatic space detection", "For supported fixed-camera views, the vision pipeline identifies the layout and returns verified parking-space polygons."],
  ["Occupancy intelligence", "A calibrated per-space classifier reports vacant and occupied states with confidence."],
  ["Fixed-camera video", "Stable layouts are reused while temporal smoothing suppresses one-frame state changes."],
  ["Traceable operations", "History, analytics, diagnostics, reports, and source-aware results remain connected."],
];

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <a className="skip-link" href="#landing-content">Skip to main content</a>
      <header className="border-b border-white/10">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-5 px-5 py-5 lg:px-8">
          <Link href="/" className="flex items-center gap-3 font-extrabold">
            <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-amber-400 text-slate-950">P</span>
            <span className="hidden max-w-xl leading-5 sm:block">{PRODUCT_NAME}</span>
          </Link>
          <nav className="flex items-center gap-3" aria-label="Landing page actions">
            <LinkButton href="/dashboard" variant="outline">Open dashboard</LinkButton>
            <LinkButton href="/presentation" variant="primary" className="hidden sm:inline-flex">Start presentation</LinkButton>
          </nav>
        </div>
      </header>

      <div id="landing-content">
        <section className="relative overflow-hidden px-5 py-20 lg:px-8 lg:py-28">
          <div aria-hidden="true" className="absolute -right-40 top-4 h-96 w-96 rounded-full bg-amber-400/20 blur-3xl" />
          <div className="mx-auto grid max-w-7xl gap-14 lg:grid-cols-[1.1fr_.9fr] lg:items-center">
            <div className="relative">
              <p className="text-sm font-extrabold uppercase tracking-[.2em] text-amber-300">AI-powered parking visibility</p>
              <h1 className="mt-5 max-w-4xl text-4xl font-black leading-tight tracking-tight sm:text-5xl lg:text-6xl">{PRODUCT_NAME}</h1>
              <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">Turn parking-lot imagery into an explainable map of free and occupied spaces—without sensors or manual slot marking in the normal workflow.</p>
              <div className="mt-9 flex flex-wrap gap-3">
                <LinkButton href="/dashboard" variant="primary">Open dashboard</LinkButton>
                <LinkButton href="/presentation" variant="outline">Start presentation</LinkButton>
              </div>
            </div>
            <div className="relative rounded-[2rem] border border-white/15 bg-white/5 p-5 shadow-2xl">
              <div className="aspect-[4/3] rounded-2xl bg-[linear-gradient(145deg,#334155,#0f172a)] p-5">
                <div className="grid h-full grid-cols-4 gap-3" aria-label="Illustration of parking-space status">
                  {Array.from({ length: 20 }, (_, index) => (
                    <div key={index} className={`rounded-lg border-2 ${[1,2,5,8,9,14,17].includes(index) ? "border-red-400 bg-red-500/30" : "border-emerald-400 bg-emerald-500/20"}`} />
                  ))}
                </div>
              </div>
              <div className="mt-4 flex items-center justify-between text-sm font-bold"><span className="text-emerald-300">13 vacant</span><span className="text-red-300">7 occupied</span><span>20 total</span></div>
            </div>
          </div>
        </section>

        <section className="bg-white px-5 py-20 text-slate-950 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <div className="grid gap-10 lg:grid-cols-2">
              <div><p className="label">The parking problem</p><h2 className="mt-3 text-3xl font-black">Manual monitoring is slow, inconsistent, and difficult to scale.</h2></div>
              <p className="text-lg leading-8 text-slate-600">Drivers circle for spaces while operators rely on observation or expensive dedicated hardware. This product analyses existing image and prerecorded-video evidence, automatically localises spaces, and presents useful occupancy information with a visible confidence trail.</p>
            </div>
            <div className="mt-14 grid gap-5 md:grid-cols-2 xl:grid-cols-4">
              {capabilities.map(([title, detail]) => <article key={title} className="rounded-2xl border border-slate-200 bg-slate-50 p-6"><h3 className="font-extrabold">{title}</h3><p className="mt-3 text-sm leading-6 text-slate-600">{detail}</p></article>)}
            </div>
          </div>
        </section>

        <section className="px-5 py-20 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <p className="text-sm font-extrabold uppercase tracking-[.18em] text-amber-300">AI workflow</p>
            <h2 className="mt-3 text-3xl font-black">From pixels to operational evidence</h2>
            <ol className="mt-10 grid gap-5 md:grid-cols-4">
              {["Upload an image or fixed-camera video", "Automatically localise parking spaces", "Classify and stabilize occupancy states", "Save overlays, timelines, analytics, and reports"].map((step, index) => <li key={step} className="rounded-2xl border border-white/15 bg-white/5 p-6"><span className="text-3xl font-black text-amber-300">0{index + 1}</span><p className="mt-4 font-bold leading-6">{step}</p></li>)}
            </ol>
          </div>
        </section>

        <section className="bg-slate-100 px-5 py-20 text-slate-950 lg:px-8">
          <div className="mx-auto grid max-w-7xl gap-8 lg:grid-cols-3">
            <article className="rounded-2xl bg-white p-7"><p className="label">Dataset evidence</p><p className="mt-3 text-2xl font-black">PKLot · CNRPark+EXT · ACPDS</p><p className="mt-3 text-sm leading-6 text-slate-600">At least ten validated, non-duplicate prepared image scenarios per dataset, with full-image geometry and ground truth.</p></article>
            <article className="rounded-2xl bg-white p-7"><p className="label">Model evidence</p><p className="mt-3 text-2xl font-black">Leakage-safe evaluation</p><p className="mt-3 text-sm leading-6 text-slate-600">The reproducible 90.7% baseline remains available while enhanced models use group-separated train, validation, and single-use holdout partitions.</p></article>
            <article className="rounded-2xl bg-white p-7"><p className="label">Privacy and control</p><p className="mt-3 text-2xl font-black">Local processing</p><p className="mt-3 text-sm leading-6 text-slate-600">Uploads, model artifacts, history, and generated results remain on the operator’s own computer.</p></article>
          </div>
        </section>
      </div>

      <footer className="border-t border-white/10 px-5 py-8 text-sm text-slate-400"><div className="mx-auto max-w-7xl">{PRODUCT_NAME}</div></footer>
    </main>
  );
}
