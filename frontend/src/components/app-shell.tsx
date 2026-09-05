"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { ConnectionStatus } from "@/components/connection-status";
import { navigation, PRODUCT_NAME } from "@/lib/product";

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  if (pathname === "/") return <>{children}</>;

  return (
    <div className="min-h-screen bg-[#f4f7fb]">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <aside className="app-sidebar fixed inset-y-0 left-0 z-20 flex w-64 flex-col bg-[#111a2e] px-4 py-6 text-white">
        <div className="mb-8 flex items-center gap-3 px-2">
          <div className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-amber-400 text-lg font-black text-slate-950">P</div>
          <p className="text-sm font-extrabold leading-5">{PRODUCT_NAME}</p>
        </div>
        <nav className="space-y-1" aria-label="Primary navigation">
          {navigation.map((item) => {
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={active
                  ? "flex min-h-11 items-center gap-3 rounded-xl bg-white/15 px-3 py-3 text-sm font-semibold text-white"
                  : "flex min-h-11 items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium text-slate-300 transition hover:bg-white/10 hover:text-white"}
              >
                <span aria-hidden="true" className={active ? "text-amber-300" : ""}>{item.symbol}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-auto rounded-2xl border border-white/10 bg-white/5 p-4">
          <p className="text-xs font-bold text-emerald-300">● PROCESSING READY</p>
          <p className="mt-2 text-xs leading-5 text-slate-300">Your images, video results, and analysis history stay on this computer.</p>
        </div>
      </aside>

      <main className="app-main min-h-screen md:ml-64">
        <header className="border-b border-slate-200 bg-white px-4 py-4 lg:px-10">
          <div className="flex min-h-12 items-center justify-between gap-4">
            <p className="max-w-[75%] text-sm font-extrabold leading-5 text-slate-900 sm:max-w-none sm:text-base">{PRODUCT_NAME}</p>
            <ConnectionStatus />
          </div>
          <details key={pathname} className="group mt-3 md:hidden">
            <summary className="flex min-h-11 cursor-pointer list-none items-center justify-between rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-extrabold text-slate-900 focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-slate-700">
              Menu
              <span aria-hidden="true" className="transition group-open:rotate-180">⌄</span>
            </summary>
            <nav className="mt-2 grid gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-lg sm:grid-cols-2" aria-label="Mobile navigation">
              {navigation.map((item) => {
                const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
                return (
                  <Link key={item.href} href={item.href} aria-current={active ? "page" : undefined} className={`flex min-h-11 items-center rounded-lg px-3 py-2.5 text-sm font-bold focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-slate-700 ${active ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-800 hover:bg-slate-200"}`}>
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          </details>
        </header>
        <div id="main-content" className="p-4 sm:p-6 lg:p-10">{children}</div>
      </main>
    </div>
  );
}
