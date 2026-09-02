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

The foundation validator checks seven required files without extracting them. Later adapters will normalize dataset-specific annotations into a shared scenario manifest containing dataset, lot, condition, image path, slot polygons, and ground-truth occupancy.

Original archives are immutable. Extraction targets `raw`; normalized metadata targets `prepared`; presentation scenarios target `demo`.
