import { AnalysisDetail } from "@/components/analysis-detail";

export default async function AnalysisDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <div className="mx-auto max-w-7xl"><p className="label">History detail</p><h1 className="mt-2 text-3xl font-black tracking-tight text-slate-900">Inspect analysis #{id}</h1><p className="mt-2 text-sm text-slate-500">Compare every saved AI decision with verified dataset ground truth.</p><AnalysisDetail analysisId={id} /></div>;
}
