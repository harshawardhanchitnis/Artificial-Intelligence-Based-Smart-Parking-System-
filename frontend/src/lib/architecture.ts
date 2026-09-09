/**
 * What the system currently is, and what each figure came from.
 *
 * Presentation Mode used to be a benchmark showcase with no architecture
 * narrative, so the only account of the design a viewer got was the landing
 * page's -- which described an oriented-box detector and a human verification
 * step, neither of which the product still has.
 *
 * Every number below carries its own `source`. That field is not decoration:
 * it is the rule that stopped stale figures being copied forward from earlier
 * reports. A value with no verifiable source does not belong in this file.
 * Figures that the running system can report about itself -- readiness, model
 * names, benchmark results -- are fetched live instead of being listed here.
 */

export type Figure = {
  value: string;
  label: string;
  source: string;
};

export const PIPELINE: Array<{ step: string; detail: string }> = [
  {
    step: "Input",
    detail:
      "A still image, or a fixed-camera recording sampled into frames.",
  },
  {
    step: "Scene understanding",
    detail:
      "The frame is read as a whole before any bay map exists, so nothing downstream depends on the geometry succeeding first.",
  },
  {
    step: "Known-layout recall, or generalized localisation",
    detail:
      "A camera the system has seen resolves to its stored polygons by perceptual fingerprint. An unfamiliar one goes to the generalized four-corner detector; a fixed-camera sequence calibrates itself across several frames and keeps only geometry that recurs.",
  },
  {
    step: "Full-scene vehicle detection",
    detail:
      "Every vehicle in the frame is detected independently of the bay map, so a vehicle is found whether or not its bay was.",
  },
  {
    step: "Parking-area reasoning",
    detail:
      "The facility's extent is inferred from bay polygons and parked-vehicle footprints, never from a hardcoded region, so passing road traffic can be excluded from site figures.",
  },
  {
    step: "Vehicle ↔ space association",
    detail:
      "Intersection-over-minimum against a ground-footprint band, so one long vehicle spanning two bays occupies both rather than neither.",
  },
  {
    step: "Occupancy evidence",
    detail:
      "Each space is perspective-rectified to a canonical patch and scored by a temperature-calibrated MobileNetV3 classifier.",
  },
  {
    step: "Evidence fusion",
    detail:
      "Classifier and vehicle evidence are combined in log-odds, so neither can veto the other, and an unsupported vehicle-shaped object still contributes occupancy evidence.",
  },
  {
    step: "VACANT / OCCUPIED / UNCERTAIN",
    detail:
      "A probability inside the calibrated ambiguous band is reported as uncertain rather than forced to a verdict.",
  },
  {
    step: "Temporal confirmation (video)",
    detail:
      "A state change must hold across frames before it is reported, and a detection must recur in two of the last three sampled frames before it counts.",
  },
  {
    step: "History, analytics, reporting",
    detail:
      "Every run is stored with its model name and thresholds, and exported as CSV or JSON.",
  },
];

/** The geometry comparison that selected the shipped representation. */
export const GEOMETRY_CANDIDATES: Array<{
  name: string;
  family: string;
  acpdsRecall50: string;
  acpdsRecall75: string;
  cornerError: string;
  cost: string;
  selected: boolean;
}> = [
  { name: "YOLO11n-pose, four corners", family: "One-stage keypoint", acpdsRecall50: "0.507", acpdsRecall75: "0.144", cornerError: "0.260", cost: "1,089 ms CPU · 10.9 MB", selected: true },
  { name: "YOLO11n-OBB", family: "One-stage oriented box", acpdsRecall50: "0.566", acpdsRecall75: "0.103", cornerError: "0.373", cost: "1,212 ms CPU · 11.0 MB", selected: false },
  { name: "YOLO11n-seg", family: "One-stage instance mask", acpdsRecall50: "0.273", acpdsRecall75: "0.085", cornerError: "0.830", cost: "1,003 ms CPU", selected: false },
  { name: "Keypoint R-CNN (ResNet50-FPN)", family: "Two-stage", acpdsRecall50: "0.542", acpdsRecall75: "0.340", cornerError: "0.203", cost: "5,492 ms CPU · 236 MB", selected: false },
];

export const GEOMETRY_SOURCE =
  "All four scored through the deployed decode path on the same 150-image sample of cameras absent from training. ACPDS is the only source whose annotations follow painted bay boundaries, so it is the one that can answer a geometry question.";

/** Figures verified against their artifact during this pass. */
export const VERIFIED_FIGURES: Figure[] = [
  { value: "97.9%", label: "Occupancy balanced accuracy", source: "parking-occupancy-enhanced-v3.json · protected holdout, 25,113 samples, opened once after the model was frozen" },
  { value: "0.75%", label: "False-vacant rate", source: "same holdout: 94 occupied spaces of 12,555 reported free" },
  { value: "0.015", label: "Expected calibration error", source: "same holdout, after temperature scaling fitted on validation only" },
  { value: "0.340", label: "Best strict-IoU bay recall (Keypoint R-CNN)", source: "tableA/keypoint-rcnn.json · ACPDS recall at IoU 0.75" },
  { value: "0.144", label: "Shipped model, same measure", source: "tableA/pose-geom-onnx.json · ACPDS recall at IoU 0.75" },
  { value: "90.2%", label: "Spaces recovered by disjoint halves of one sequence", source: "calibration-summary.json · CNRPark camera7, 14-frame reconstructed sequence" },
  { value: "0.941", label: "Median polygon IoU between those two halves", source: "calibration-summary.json" },
];

/** Stated plainly, because the interface is where over-claiming does damage. */
export const HONEST_LIMITS: string[] = [
  "Keypoint R-CNN produced the best geometry of the four candidates — 0.340 against 0.144 strict-IoU recall. It was not selected: on the CPU this product deploys to it costs 5,492 ms against 1,089 ms, from a 236 MB checkpoint against 10.9 MB. The four-corner representation therefore has more headroom than the shipped model extracts.",
  "Generalized bay recall on cameras the detector has never seen is low — 0.164 at IoU 0.50 across all sources. Known cameras and calibrated fixed-camera sequences are where the product is strong; a single photograph of an unfamiliar lot is where it abstains.",
  "The vehicle detector is COCO-pretrained and not fine-tuned for parking. It reads oblique views well and near-vertical aerial views badly, where it labels cars as household objects and returns no vehicles at all.",
  "TWO_WHEELER is supported by the architecture and measured only on general-domain imagery. It is not validated in the parking domain, because the parking corpus contains almost no two-wheelers.",
];
