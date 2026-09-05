import { SectionPlaceholder } from "@/components/section-placeholder";
import { SystemStatus } from "@/components/system-status";

export default function SystemPage() {
  return (
    <SectionPlaceholder
      eyebrow="Environment"
      title="System readiness"
      description="Run deep checks across the database, datasets, prepared catalogue, vision models, media pipeline, and independent benchmark."
    >
      <SystemStatus />
    </SectionPlaceholder>
  );
}
