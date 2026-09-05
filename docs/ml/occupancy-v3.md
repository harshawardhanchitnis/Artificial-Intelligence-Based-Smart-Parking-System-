# Enhanced occupancy model

`parking-occupancy-logistic-v2` is preserved byte-for-byte as the reproducible benchmark baseline.
The enhanced workflow compares that frozen baseline with HOG+SVM, a compact CNN, MobileNetV3-Small,
and EfficientNet-B0 on validation data. Neural candidates use the same masked perspective-
rectified 128x128 crop used in production. Selection prioritizes worst-group balanced accuracy,
then worst-dataset balanced accuracy, overall balanced accuracy, and CPU latency.

Temperature scaling and the occupied decision threshold are chosen on validation only. Reports
include accuracy, balanced accuracy, occupied precision/recall/F1, vacant specificity, false-
vacant/false-occupied rates, calibration error, per-dataset results, difficult groups, latency,
memory, learning curves, and artifact size. The single protected holdout is read only after the
decision lock is written.

Deployment uses checksum-verified FP32 ONNX with CPUExecutionProvider. FP32 is retained because
dynamic quantization offers limited benefit for convolution-heavy models; any static INT8 model
would require a separate validation-locked calibration study.

## Frozen result

- Selected architecture: MobileNetV3-Small
- Training / validation / protected holdout: 65,805 / 15,989 / 25,113 slots
- Decision threshold / temperature: 0.62 / 1.13543159
- Protected-holdout accuracy and balanced accuracy: 97.9055% / 97.9057%
- Occupied precision / recall / F1: 96.4691% / 99.4504% / 97.9371%
- Vacant specificity: 96.3609%; false-vacant rate: 0.5496%
- Dataset balanced accuracy: ACPDS 94.6303%, CNRPark+EXT 99.7304%, PKLot 96.4417%
- ONNX size: 5.806 MB; measured CPU execution: 1.3457 ms per slot (512-slot batch)

The aggregate result does not erase weak small/date groups. The final report retains, for example,
PUCPR/2012-10-30 at 40.3509% balanced accuracy. Some groups contain only one class, making their
balanced-accuracy interpretation unstable; they remain visible as limitations and future data
targets. Any model change now requires a newly protected evaluation set.
