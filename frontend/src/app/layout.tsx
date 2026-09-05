import type { Metadata } from "next";
import type { ReactNode } from "react";

import { AppShell } from "@/components/app-shell";
import { PRODUCT_NAME } from "@/lib/product";
import "./globals.css";

export const metadata: Metadata = {
  title: PRODUCT_NAME,
  description: "Automatic parking-space localisation, occupancy analysis, and operational insight.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
