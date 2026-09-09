import { LayoutCorrectionEditor } from "@/components/layout-correction";

export default async function LayoutCorrectionPage({
  params,
}: {
  params: Promise<{ layoutId: string }>;
}) {
  const { layoutId } = await params;
  return <div className="mx-auto max-w-7xl"><p className="label">Diagnostics</p><h1 className="mt-2 text-3xl font-black tracking-tight text-slate-950">Layout inspection</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">A research and diagnostic view, outside the normal product workflow. Parking geometry is established automatically; where it cannot be, the system reports that rather than asking anyone to draw it. Anything adjusted here stays visibly separated from automatic evaluation.</p><LayoutCorrectionEditor layoutId={layoutId} /></div>;
}
