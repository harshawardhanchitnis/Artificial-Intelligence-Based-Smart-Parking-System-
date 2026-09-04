import { SectionPlaceholder } from "@/components/section-placeholder";
import { SystemStatus } from "@/components/system-status";

export default function SystemPage() {
  return (
    <SectionPlaceholder
      eyebrow="Environment"
      title="Version 1.0 release status"
      description="Confirm the final offline product contract and run deep checks across the database, datasets, catalogue, model, and independent benchmark."
    >
      <SystemStatus />
    </SectionPlaceholder>
  );
}
