import { LayoutCorrectionEditor } from "@/components/layout-correction";

export default async function LayoutCorrectionPage({
  params,
}: {
  params: Promise<{ layoutId: string }>;
}) {
  const { layoutId } = await params;
  return <div className="mx-auto max-w-7xl"><p className="label">Exceptional layout review</p><h1 className="mt-2 text-3xl font-black tracking-tight text-slate-950">Advanced calibration</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">Review the automatic layout only when localisation confidence is insufficient. Corrected results remain visibly separated from automatic evaluation.</p><LayoutCorrectionEditor layoutId={layoutId} /></div>;
}
