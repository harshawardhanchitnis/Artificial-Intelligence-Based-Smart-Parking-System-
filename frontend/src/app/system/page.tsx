import { SectionPlaceholder } from "@/components/section-placeholder";
import { SystemStatus } from "@/components/system-status";

export default function SystemPage() {
  return (
    <SectionPlaceholder
      eyebrow="Environment"
      title="System readiness"
      description="Run deep local checks across the database, datasets, demo catalogue, model, and independent benchmark before presenting."
    >
      <SystemStatus />
    </SectionPlaceholder>
  );
}
