const slots = [
  "free", "occupied", "free", "free", "occupied", "occupied",
  "free", "occupied", "free", "occupied", "free", "free",
];

export default function AnalysePage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">Analysis workspace</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">
        Analyse a prepared parking scenario
      </h1>
      <p className="mt-2 text-sm text-slate-500">
        This milestone provides the product shell. Dataset preparation and real model
        inference arrive in the following milestones.
      </p>

      <div className="mt-8 grid gap-6 xl:grid-cols-[360px_1fr]">
        <aside className="card h-fit p-6">
          <h2 className="font-bold text-slate-900">Scenario controls</h2>
          <div className="mt-5 space-y-5">
            {[
              ["Dataset", "PKLot"],
              ["Parking lot", "UFPR04"],
              ["Condition", "Sunny"],
              ["Scenario", "Prepared sample 01"],
            ].map(([label, value]) => (
              <label className="block" key={label}>
                <span className="mb-2 block text-xs font-bold text-slate-500">{label}</span>
                <select className="w-full rounded-xl border border-slate-200 bg-white px-3 py-3 text-sm outline-none focus:border-amber-400">
                  <option>{value}</option>
                </select>
              </label>
            ))}
          </div>
          <button
            disabled
            className="mt-6 w-full cursor-not-allowed rounded-xl bg-slate-200 px-4 py-3 text-sm font-bold text-slate-500"
          >
            Model available in AI milestone
          </button>
        </aside>

        <section className="card overflow-hidden">
          <div className="flex items-center justify-between border-b border-slate-100 px-6 py-5">
            <div>
              <p className="font-bold">Parking-lot preview</p>
              <p className="mt-1 text-xs text-slate-500">Illustrative overlay only</p>
            </div>
            <div className="flex gap-4 text-xs font-bold">
              <span className="text-emerald-600">● Vacant</span>
              <span className="text-red-600">● Occupied</span>
            </div>
          </div>
          <div className="bg-gradient-to-br from-slate-700 to-slate-900 p-8">
            <div className="grid grid-cols-4 gap-3 rounded-2xl border border-dashed border-white/30 bg-slate-950/25 p-5 sm:grid-cols-6">
              {slots.map((status, index) => (
                <div
                  key={index}
                  className={
                    status === "free"
                      ? "grid aspect-[1/1.7] place-items-center rounded border-2 border-emerald-400 bg-emerald-400/15 text-xs font-black text-emerald-300"
                      : "grid aspect-[1/1.7] place-items-center rounded border-2 border-red-400 bg-red-400/20 text-xs font-black text-red-300"
                  }
                >
                  {status === "free" ? "FREE" : "CAR"}
                </div>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-3 divide-x divide-slate-100">
            {[["Total", "12"], ["Vacant", "7"], ["Occupied", "5"]].map(
              ([label, value]) => (
                <div className="p-5 text-center" key={label}>
                  <p className="text-2xl font-black">{value}</p>
                  <p className="mt-1 text-xs font-bold text-slate-400">{label}</p>
                </div>
              ),
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
