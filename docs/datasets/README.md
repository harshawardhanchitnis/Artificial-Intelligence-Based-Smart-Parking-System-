# Dataset storage plan

External root:

```text
D:\Projects\AI Based Smart Parking System Data
├── archives
│   ├── PKLot
│   ├── CNRPark+EXT
│   └── ACPDS
├── raw
│   ├── PKLot
│   ├── CNRPark+EXT
│   └── ACPDS
├── prepared
└── demo
```

The validator checks seven required files without extracting them. Milestone 2 adapters normalize dataset-specific annotations into a shared scenario manifest containing dataset, lot/camera, condition, image path, normalized slot polygons, and ground-truth occupancy.

Original archives are immutable. Extraction targets `raw`; normalized metadata targets `prepared`; presentation scenarios target `demo`.

## Preparation profiles

- `Plan` reports archive presence, archive bytes, free space, and a conservative full-extraction estimate. It writes nothing.
- `Demo` streams a configurable number of complete scenarios per dataset directly from the archives. It is the recommended profile for development and presentation.
- `Full` safely extracts all archives to `raw` only after `-ConfirmFullExtraction`. It is optional.

```powershell
.\scripts\prepare-datasets.ps1 -Profile Demo -SamplesPerDataset 3
```

The prepared catalogue uses schema version `1.0`. Polygon coordinates are clamped to `[0, 1]`, making one frontend overlay work across different image sizes. The original labels remain identified as `dataset_ground_truth`; no prediction is implied.
