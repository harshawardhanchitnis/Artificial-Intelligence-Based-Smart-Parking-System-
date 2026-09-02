"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

const navigation = [
  { href: "/", label: "Dashboard", symbol: "▦" },
  { href: "/presentation", label: "Presentation", symbol: "▶" },
  { href: "/analyse", label: "Analyse", symbol: "◎" },
  { href: "/parking-lots", label: "Parking lots", symbol: "P" },
  { href: "/history", label: "History", symbol: "↺" },
  { href: "/analytics", label: "Analytics", symbol: "⌁" },
  { href: "/reports", label: "Reports", symbol: "▤" },
  { href: "/system", label: "System", symbol: "⚙" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[#f4f7fb]">
      <aside className="app-sidebar fixed inset-y-0 left-0 z-20 flex w-64 flex-col bg-[#111a2e] px-4 py-6 text-white">
        <div className="mb-8 flex items-center gap-3 px-2">
          <div className="grid h-11 w-11 place-items-center rounded-xl bg-amber-500 text-lg font-black text-slate-950">
            P
          </div>
          <div>
            <p className="text-sm font-extrabold tracking-wide">PARKSENSE AI</p>
            <p className="text-xs text-slate-400">Offline parking intelligence</p>
          </div>
        </div>

        <nav className="space-y-1">
          {navigation.map((item) => {
            const active =
              item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={
                  active
                    ? "flex items-center gap-3 rounded-xl bg-white/10 px-3 py-3 text-sm font-semibold text-white"
                    : "flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium text-slate-400 transition hover:bg-white/5 hover:text-white"
                }
              >
                <span className={active ? "text-amber-400" : ""}>{item.symbol}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto rounded-2xl border border-white/10 bg-white/5 p-4">
          <p className="text-xs font-bold text-emerald-400">● OFFLINE MODE</p>
          <p className="mt-2 text-xs leading-5 text-slate-400">
            Dataset-driven analysis. No camera, sensor, or cloud AI connection.
          </p>
        </div>
      </aside>

      <main className="app-main min-h-screen md:ml-64">
        <header className="flex min-h-20 items-center justify-between border-b border-slate-200 bg-white px-6 lg:px-10">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
              Artificial Intelligence Based
            </p>
            <p className="font-bold text-slate-800">Smart Parking System</p>
          </div>
          <div className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-bold text-emerald-700">
            Local system ready
          </div>
        </header>
        <div className="p-6 lg:p-10">{children}</div>
      </main>
    </div>
  );
}
