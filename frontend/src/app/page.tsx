import Link from "next/link";

import { LayoutVisual } from "@/components/layout-visual";
import { LinkButton } from "@/components/ui";
import { deployment } from "@/lib/deployment";
import { PRODUCT_NAME } from "@/lib/product";

export const metadata = {
  title: `${PRODUCT_NAME} — automatic parking-space detection and occupancy`,
  description: deployment.summary,
};

const valueProps = [
  {
    title: "Find the spaces, not just the cars",
    body: "Parking-space geometry is predicted directly from the image as four independent corners, so bays follow the real boundaries through perspective and angled parking instead of being squared off into rectangles.",
  },
  {
    title: "Vacant, occupied — or honestly uncertain",
    body: "A calibrated per-space classifier reports state with confidence, and borderline spaces are shown as uncertain rather than forced into a guess.",
  },
  {
    title: "Fixed-camera video intelligence",
    body: "A new camera calibrates itself across several frames, keeping only the geometry that recurs, then reuses that layout while stability checks pass. Temporal smoothing suppresses single-frame flicker.",
  },
  {
    title: deployment.privacyTitle,
    body: deployment.privacyBody,
  },
];

const steps = [
  {
    title: "Bring a parking image or fixed-camera clip",
    body: "Upload a JPEG, PNG or WebP still, an MP4 or AVI recording, or pick one of the thirty prepared ground-truth scenarios.",
  },
  {
    title: "The system establishes the parking-space geometry",
    body: "A camera it already knows resolves instantly to its stored layout. An unfamiliar one calibrates automatically from several frames, and a scene that cannot support a reliable layout is reported as unresolved rather than guessed at.",
  },
  {
    title: "Read the whole scene, then decide",
    body: "Every space is perspective-rectified and scored, and the full frame is searched separately for cars, two-wheelers and trucks. The two lines of evidence are fused, so a visible vehicle is not missed because its bay was.",
  },
  {
    title: "Keep the evidence",
    body: deployment.auditBody,
  },
];

const features = [
  {
    label: "Onboarding",
    title: "A parking lot the system has never seen",
    body: "The system establishes the layout itself, stores it against that camera and reuses it on every later image from the same view. When the evidence will not support a layout it says so rather than inventing one.",
    href: "/analyse",
    cta: "Analyse an image",
  },
  {
    label: "Image intelligence",
    title: "One still, a complete occupancy map",
    body: "Numbered green and red overlays, per-space confidence, totals, and — for prepared scenarios — direct agreement against verified dataset labels.",
    href: "/analyse",
    cta: "Open image analysis",
  },
  {
    label: "Video intelligence",
    title: "Occupancy across a whole recording",
    body: "Annotated H.264 playback with metrics that track the frame on screen, an occupancy timeline, camera-stability scoring and per-space change events.",
    href: "/video",
    cta: "Open video analysis",
  },
  {
    label: "Operations",
    title: "Analytics, history and reports",
    body: "Every run is stored and inspectable: occupancy trends, dataset comparisons, per-space error review and CSV or JSON export.",
    href: "/analytics",
    cta: "Open analytics",
  },
];

const useCases = [
  ["University campuses", "Understand how faculty and student lots fill through the day from existing camera footage."],
  ["Commercial parking", "Report occupancy per level or zone without installing per-bay sensors."],
  ["Office parks", "Measure how much of a leased allocation is actually used before renewing it."],
  ["Residential societies", "Audit resident and visitor bay usage from a single fixed camera."],
  ["Shopping centres", "Quantify peak-hour pressure and identify which zones saturate first."],
  ["Smart-city pilots", "Evaluate a computer-vision approach against recorded footage before committing to hardware."],
];

const distinctions = [
  ["Layout confidence", "How certain the system is that it has found the right parking-space geometry."],
  ["Occupancy confidence", "How certain the classifier is about one space, which is not the same as being correct."],
  ["Ground-truth agreement", "How a saved run compared against verified dataset labels, where those labels exist."],
  ["Benchmark accuracy", "Measured on a protected holdout that is opened once, after the model is frozen."],
];

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-white text-slate-950">
      <a className="skip-link" href="#landing-content">Skip to main content</a>

      <header className="sticky top-0 z-40 border-b border-white/10 bg-[#111a2e]/95 text-white backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-5 px-5 py-4 lg:px-8">
          <Link href="/" className="flex items-center gap-3 font-extrabold">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-amber-400 text-lg text-slate-950">P</span>
            <span className="hidden text-sm leading-5 sm:block sm:max-w-56 lg:max-w-none">{PRODUCT_NAME}</span>
          </Link>
          <nav className="flex items-center gap-2 sm:gap-3" aria-label="Primary">
            <Link href="/diagnostics" className="hidden min-h-11 items-center px-3 text-sm font-bold text-slate-300 hover:text-white lg:inline-flex">Model evidence</Link>
            <LinkButton href="/analyse" variant="primary">Analyse an image</LinkButton>
            <LinkButton href="/dashboard" variant="outline" className="hidden sm:inline-flex">Dashboard</LinkButton>
          </nav>
        </div>
      </header>

      <div id="landing-content">
        {/* ---------------- hero ---------------- */}
        <section className="relative overflow-hidden bg-[#111a2e] px-5 pb-20 pt-16 text-white lg:px-8 lg:pb-28 lg:pt-24">
          <div aria-hidden="true" className="pointer-events-none absolute -right-52 -top-28 h-[34rem] w-[34rem] rounded-full bg-amber-400/15 blur-3xl" />
          <div aria-hidden="true" className="pointer-events-none absolute -left-40 bottom-0 h-96 w-96 rounded-full bg-emerald-400/10 blur-3xl" />
          <div className="relative mx-auto grid max-w-7xl items-center gap-14 lg:grid-cols-[1.05fr_.95fr]">
            <div>
              <p className="text-xs font-extrabold uppercase tracking-[.22em] text-amber-300">{deployment.eyebrow}</p>
              <h1 className="mt-5 text-4xl font-black leading-[1.08] tracking-tight sm:text-5xl lg:text-[3.4rem]">
                Turn parking-lot footage into a live map of free spaces
              </h1>
              <p className="mt-6 max-w-xl text-lg leading-8 text-slate-300">
                Upload an image or a fixed-camera recording. The system finds the parking spaces, detects every vehicle in the frame, decides which spaces are taken, and shows you exactly how confident it is — {deployment.heroTail}.
              </p>
              <div className="mt-9 flex flex-wrap gap-3">
                <LinkButton href="/analyse" variant="primary">Analyse a parking image</LinkButton>
                <LinkButton href="/video" variant="outline">Analyse parking video</LinkButton>
                <Link href="/dashboard" className="inline-flex min-h-11 items-center px-2 text-sm font-extrabold text-slate-300 underline-offset-4 hover:text-white hover:underline">
                  View the dashboard
                </Link>
              </div>
              <dl className="mt-11 grid max-w-lg grid-cols-3 gap-6 border-t border-white/10 pt-7">
                {[["3", "benchmark datasets"], ["30", "prepared scenarios"], deployment.privacyStat].map(([value, label]) => (
                  <div key={label}>
                    <dt className="text-2xl font-black text-amber-300">{value}</dt>
                    <dd className="mt-1 text-xs font-semibold leading-5 text-slate-400">{label}</dd>
                  </div>
                ))}
              </dl>
            </div>
            <LayoutVisual />
          </div>
        </section>

        {/* ---------------- problem ---------------- */}
        <section className="border-b border-slate-200 px-5 py-20 lg:px-8">
          <div className="mx-auto grid max-w-7xl gap-10 lg:grid-cols-[.85fr_1.15fr]">
            <div>
              <p className="label">The problem</p>
              <h2 className="mt-3 text-3xl font-black leading-tight sm:text-4xl">Counting parking spaces by hand does not scale.</h2>
            </div>
            <div className="space-y-5 text-lg leading-8 text-slate-600">
              <p>
                Drivers circle for a bay while operators rely on someone walking the lot, or on per-bay sensors that cost more than the problem. Both approaches stop working the moment you add another level or another site.
              </p>
              <p>
                This system works from the footage you already have. It reads existing images and prerecorded fixed-camera video, works out where the parking spaces are, and reports which are free — with the reasoning visible rather than hidden behind a number.
              </p>
            </div>
          </div>
        </section>

        {/* ---------------- value props ---------------- */}
        <section className="bg-slate-50 px-5 py-20 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <p className="label">What it does</p>
            <h2 className="mt-3 max-w-2xl text-3xl font-black leading-tight sm:text-4xl">Four things this product is built to get right</h2>
            <div className="mt-12 grid gap-6 md:grid-cols-2 xl:grid-cols-4">
              {valueProps.map((item) => (
                <article className="card flex flex-col p-7" key={item.title}>
                  <h3 className="text-lg font-black leading-snug">{item.title}</h3>
                  <p className="mt-3 text-sm leading-6 text-slate-600">{item.body}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* ---------------- how it works ---------------- */}
        <section className="px-5 py-20 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <p className="label">How it works</p>
            <h2 className="mt-3 max-w-2xl text-3xl font-black leading-tight sm:text-4xl">From a photograph to an auditable occupancy record</h2>
            <ol className="mt-12 grid gap-x-8 gap-y-10 md:grid-cols-2 xl:grid-cols-4">
              {steps.map((step, index) => (
                <li key={step.title} className="relative border-t-2 border-slate-900 pt-6">
                  <span className="absolute -top-4 left-0 grid h-8 w-8 place-items-center rounded-full bg-slate-900 text-xs font-black text-white">
                    {index + 1}
                  </span>
                  <h3 className="text-lg font-black leading-snug">{step.title}</h3>
                  <p className="mt-3 text-sm leading-6 text-slate-600">{step.body}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* ---------------- features ---------------- */}
        <section className="bg-slate-50 px-5 py-20 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <p className="label">Capabilities</p>
            <h2 className="mt-3 max-w-2xl text-3xl font-black leading-tight sm:text-4xl">Built around the whole workflow, not just the model</h2>
            <div className="mt-12 grid gap-6 lg:grid-cols-2">
              {features.map((feature) => (
                <article className="card flex flex-col p-8" key={feature.title}>
                  <p className="label">{feature.label}</p>
                  <h3 className="mt-3 text-xl font-black leading-snug">{feature.title}</h3>
                  <p className="mt-3 flex-1 text-sm leading-6 text-slate-600">{feature.body}</p>
                  <LinkButton href={feature.href} variant="outline" className="mt-6 self-start">{feature.cta}</LinkButton>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* ---------------- model evidence ---------------- */}
        <section className="bg-[#111a2e] px-5 py-20 text-white lg:px-8">
          <div className="mx-auto max-w-7xl">
            <div className="grid gap-10 lg:grid-cols-[.9fr_1.1fr] lg:items-end">
              <div>
                <p className="text-xs font-extrabold uppercase tracking-[.22em] text-amber-300">Under the hood</p>
                <h2 className="mt-3 text-3xl font-black leading-tight sm:text-4xl">A two-stage pipeline you can inspect</h2>
              </div>
              <p className="text-lg leading-8 text-slate-300">
                Geometry and occupancy are separate stages, so a mistake in one is visible rather than absorbed by the other. Every figure below comes from the project&rsquo;s own evaluation protocol, with cameras and capture dates kept apart across training, validation and the protected holdout.
              </p>
            </div>

            <div className="mt-12 grid gap-6 lg:grid-cols-3">
              <article className="rounded-[18px] border border-white/15 bg-white/5 p-7">
                <p className="text-xs font-extrabold uppercase tracking-[.14em] text-slate-400">Stage one</p>
                <h3 className="mt-3 text-xl font-black">Parking-space localisation</h3>
                <p className="mt-3 text-sm leading-6 text-slate-300">
                  A registered camera resolves to its stored polygons immediately. An unfamiliar view goes to a generalized four-corner detector that predicts each bay as four independently regressed corners, and a fixed-camera sequence calibrates itself across several frames. Where the evidence will not support a layout the system abstains and says so, rather than reporting a guess as fact.
                </p>
              </article>
              <article className="rounded-[18px] border border-white/15 bg-white/5 p-7">
                <p className="text-xs font-extrabold uppercase tracking-[.14em] text-slate-400">Stage two</p>
                <h3 className="mt-3 text-xl font-black">Occupancy classification</h3>
                <p className="mt-3 text-sm leading-6 text-slate-300">
                  Each space is perspective-rectified to a canonical patch and scored by a MobileNetV3 classifier with temperature-calibrated probabilities, so a reported confidence means what it says.
                </p>
              </article>
              <article className="rounded-[18px] border border-white/15 bg-white/5 p-7">
                <p className="text-xs font-extrabold uppercase tracking-[.14em] text-slate-400">For video</p>
                <h3 className="mt-3 text-xl font-black">Temporal confirmation</h3>
                <p className="mt-3 text-sm leading-6 text-slate-300">
                  Camera stability is scored per frame. Frames where the view has moved are marked uncertain rather than analysed, and a space must hold its new state across frames before the change is reported.
                </p>
              </article>
            </div>

            <div className="mt-10 rounded-[18px] border border-white/15 bg-white/5 p-8">
              <div className="flex flex-wrap items-baseline justify-between gap-4">
                <h3 className="text-lg font-black">Occupancy model, measured on the protected holdout</h3>
                <Link href="/diagnostics" className="text-sm font-extrabold text-amber-300 underline-offset-4 hover:underline">
                  See the full evaluation
                </Link>
              </div>
              <dl className="mt-7 grid gap-6 sm:grid-cols-2 xl:grid-cols-4">
                {[
                  ["97.9%", "Balanced accuracy", "25,113 held-out samples"],
                  ["99.3%", "Occupied recall", "occupied spaces correctly flagged"],
                  ["0.75%", "False-vacant rate", "occupied spaces wrongly shown free"],
                  ["0.015", "Calibration error", "expected calibration error"],
                ].map(([value, label, note]) => (
                  <div key={label}>
                    <dt className="text-3xl font-black text-amber-300">{value}</dt>
                    <dd className="mt-2 text-sm font-bold">{label}</dd>
                    <dd className="mt-1 text-xs leading-5 text-slate-400">{note}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-7 border-t border-white/10 pt-5 text-xs leading-6 text-slate-400">
                Measured on the frozen protected holdout after the architecture, threshold and calibration were locked. Cameras and dates do not cross partitions, and near-duplicate frames are quarantined. Generalisation to a camera outside the benchmark datasets is not claimed by these numbers — that is what the verification step exists for.
              </p>
            </div>
          </div>
        </section>

        {/* ---------------- trust ---------------- */}
        <section className="px-5 py-20 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <div className="grid gap-10 lg:grid-cols-[.85fr_1.15fr]">
              <div>
                <p className="label">Trust</p>
                <h2 className="mt-3 text-3xl font-black leading-tight sm:text-4xl">A confident answer is not the same as a correct one</h2>
                <p className="mt-5 text-base leading-7 text-slate-600">
                  The interface keeps these ideas separate everywhere they appear, because collapsing them is how a computer-vision product ends up quietly misleading the person using it.
                </p>
              </div>
              <dl className="grid gap-x-8 gap-y-6 sm:grid-cols-2">
                {distinctions.map(([term, meaning]) => (
                  <div key={term} className="border-l-2 border-amber-400 pl-5">
                    <dt className="font-black">{term}</dt>
                    <dd className="mt-2 text-sm leading-6 text-slate-600">{meaning}</dd>
                  </div>
                ))}
              </dl>
            </div>
            <div className="card mt-12 grid gap-8 p-8 lg:grid-cols-3">
              {[
                ["Uncertain is a real answer", "Spaces whose probability falls in the ambiguous band are reported as uncertain instead of being pushed to vacant or occupied."],
                ["Weak geometry is abstained on, not guessed", "If the evidence will not support a layout, the run reports that and shows the scene it could still read — including every vehicle it detected — rather than producing a confident-looking overlay of the wrong lot."],
                ["Every run is reproducible", "Prepared scenarios can be re-run and compared against verified labels, and every saved run keeps its model name and decision threshold."],
              ].map(([title, body]) => (
                <div key={title}>
                  <h3 className="font-black">{title}</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ---------------- use cases ---------------- */}
        <section className="bg-slate-50 px-5 py-20 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <p className="label">Where it fits</p>
            <h2 className="mt-3 max-w-2xl text-3xl font-black leading-tight sm:text-4xl">Any fixed camera already pointed at a car park</h2>
            <div className="mt-12 grid gap-x-8 gap-y-9 sm:grid-cols-2 xl:grid-cols-3">
              {useCases.map(([title, body]) => (
                <div key={title} className="border-t border-slate-300 pt-5">
                  <h3 className="font-black">{title}</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ---------------- final CTA ---------------- */}
        <section className="bg-[#111a2e] px-5 py-20 text-white lg:px-8">
          <div className="mx-auto max-w-4xl text-center">
            <h2 className="text-3xl font-black leading-tight sm:text-4xl">Try it on a parking lot you know</h2>
            <p className="mx-auto mt-5 max-w-2xl text-lg leading-8 text-slate-300">
              Run one of the thirty prepared ground-truth scenarios, or bring your own image or fixed-camera clip and watch the system work out the layout for itself.
            </p>
            <div className="mt-9 flex flex-wrap justify-center gap-3">
              <LinkButton href="/analyse" variant="primary">Analyse an image</LinkButton>
              <LinkButton href="/video" variant="outline">Analyse a video</LinkButton>
              <LinkButton href="/dashboard" variant="outline">Open the dashboard</LinkButton>
            </div>
          </div>
        </section>
      </div>

      <footer className="border-t border-slate-200 bg-white px-5 py-12 lg:px-8">
        <div className="mx-auto flex max-w-7xl flex-col gap-8 md:flex-row md:items-start md:justify-between">
          <div className="max-w-md">
            <p className="flex items-center gap-3 font-extrabold">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-amber-400 text-slate-950">P</span>
              {PRODUCT_NAME}
            </p>
            <p className="mt-4 text-sm leading-6 text-slate-600">
              A computer-vision system for parking-space detection and occupancy analysis. Evaluated on PKLot, CNRPark+EXT and ACPDS under a group-separated protocol.
            </p>
          </div>
          <nav aria-label="Footer" className="grid grid-cols-2 gap-x-12 gap-y-2 text-sm font-bold sm:grid-cols-3">
            {[
              ["/analyse", "Analyse images"],
              ["/video", "Analyse video"],
              ["/dashboard", "Dashboard"],
              ["/analytics", "Analytics"],
              ["/diagnostics", "Model evidence"],
              ["/system", "System status"],
            ].map(([href, label]) => (
              <Link key={href} href={href} className="min-h-11 py-2 text-slate-600 hover:text-slate-950">{label}</Link>
            ))}
          </nav>
        </div>
        <p className="mx-auto mt-10 max-w-7xl border-t border-slate-200 pt-6 text-xs leading-6 text-slate-500">
          Processing happens on the machine running this application. Prerecorded fixed-camera video is supported; live CCTV ingest and moving-camera footage are not.
        </p>
      </footer>
    </main>
  );
}
