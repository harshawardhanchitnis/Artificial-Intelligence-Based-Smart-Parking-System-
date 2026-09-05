import Link from "next/link";
import type { ButtonHTMLAttributes, ReactNode } from "react";

const styles = {
  primary:
    "bg-amber-400 text-slate-950 hover:bg-amber-300 active:bg-amber-500 focus-visible:outline-amber-500",
  secondary:
    "bg-slate-900 text-white hover:bg-slate-700 active:bg-slate-950 focus-visible:outline-slate-700",
  outline:
    "border border-slate-300 bg-white text-slate-900 hover:bg-slate-100 active:bg-slate-200 focus-visible:outline-slate-600",
  danger:
    "bg-red-700 text-white hover:bg-red-600 active:bg-red-800 focus-visible:outline-red-700",
};

export function Button({
  variant = "secondary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: keyof typeof styles }) {
  return (
    <button
      {...props}
      className={`inline-flex min-h-11 items-center justify-center rounded-xl px-4 py-2.5 text-sm font-extrabold transition focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-700 disabled:shadow-none ${styles[variant]} ${className}`}
    />
  );
}

export function LinkButton({
  href,
  children,
  variant = "secondary",
  className = "",
}: {
  href: string;
  children: ReactNode;
  variant?: keyof typeof styles;
  className?: string;
}) {
  return (
    <Link
      href={href}
      className={`inline-flex min-h-11 items-center justify-center rounded-xl px-4 py-2.5 text-sm font-extrabold transition focus-visible:outline focus-visible:outline-3 focus-visible:outline-offset-2 ${styles[variant]} ${className}`}
    >
      {children}
    </Link>
  );
}

export function StatusBadge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "success" | "warning" | "danger" | "neutral";
}) {
  const tones = {
    success: "border-emerald-200 bg-emerald-50 text-emerald-800",
    warning: "border-amber-300 bg-amber-50 text-amber-900",
    danger: "border-red-200 bg-red-50 text-red-800",
    neutral: "border-slate-200 bg-slate-100 text-slate-800",
  };
  return (
    <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-extrabold ${tones[tone]}`}>
      {children}
    </span>
  );
}
