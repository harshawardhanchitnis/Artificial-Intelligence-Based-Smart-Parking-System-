import { SectionPlaceholder } from "@/components/section-placeholder";
import { SystemStatus } from "@/components/system-status";

export default function SystemPage() {
  return (
    <SectionPlaceholder
      eyebrow="Environment"
      title="System readiness"
      description="Confirm the local application, database, and external dataset directory before presenting."
    >
      <SystemStatus />
    </SectionPlaceholder>
  );
}
