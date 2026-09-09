/**
 * Deployment-dependent product copy.
 *
 * The landing page previously asserted that no image ever leaves the computer.
 * That is true of a local install and false of a hosted one, and a marketing
 * claim about privacy is exactly the kind that must not be inherited by a
 * deployment it does not describe. The wording is therefore selected from the
 * deployment mode rather than written into the page.
 *
 * Set NEXT_PUBLIC_DEPLOYMENT_MODE to "cloud" for a hosted instance. The default
 * is "local", which is what a checkout of this repository actually runs.
 */

export type DeploymentMode = "local" | "cloud";

export const deploymentMode: DeploymentMode =
  process.env.NEXT_PUBLIC_DEPLOYMENT_MODE === "cloud" ? "cloud" : "local";

type DeploymentCopy = {
  /** Short eyebrow above the hero headline. */
  eyebrow: string;
  /** How the hero sentence finishes; deployment-dependent, so not hard-coded. */
  heroTail: string;
  /** Headline of the data-handling card. */
  privacyTitle: string;
  /** Body of the data-handling card. */
  privacyBody: string;
  /** Figure and caption for the third hero statistic. */
  privacyStat: readonly [string, string];
  /** How the audit trail is described. */
  auditBody: string;
  /** One-line description used in the hero and the footer. */
  summary: string;
};

const copy: Record<DeploymentMode, DeploymentCopy> = {
  local: {
    eyebrow: "Local computer vision for parking",
    heroTail: "running, in this deployment, entirely on your own machine",
    privacyTitle: "Current deployment: local-first inference",
    privacyBody:
      "In this deployment every model runs on the machine serving the application: uploads, model files, history and generated results are written to local storage, and no image or video leaves it. That is a property of how it is deployed today, not a limit of the architecture -- the same pipeline runs behind a hosted tenant unchanged.",
    privacyStat: ["Local-first", "inference in this deployment"],
    auditBody:
      "Overlays, per-space decisions, occupancy timelines, analytics and exportable reports are written to a local SQLite audit trail.",
    summary:
      "Analyse parking-lot images and fixed-camera video locally. Detect parking-space geometry, verify it, and report vacant, occupied and uncertain spaces with an auditable confidence trail.",
  },
  cloud: {
    eyebrow: "Computer vision for parking",
    heroTail: "running in your own isolated tenant",
    privacyTitle: "Your footage stays yours",
    privacyBody:
      "Uploads, model files, history and generated results are held in your own tenant. Footage is processed for your analyses only, is never used to train shared models, and is deleted on the retention schedule you set.",
    privacyStat: ["Single", "tenant isolation"],
    auditBody:
      "Overlays, per-space decisions, occupancy timelines, analytics and exportable reports are written to a per-tenant audit trail.",
    summary:
      "Analyse parking-lot images and fixed-camera video. Detect parking-space geometry, verify it, and report vacant, occupied and uncertain spaces with an auditable confidence trail.",
  },
};

export const deployment = copy[deploymentMode];
