import type { ReactNode } from "react";

export function SectionPlaceholder({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="label">{eyebrow}</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">{title}</h1>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">{description}</p>
      <section className="card mt-8 p-8">{children}</section>
    </div>
  );
}
