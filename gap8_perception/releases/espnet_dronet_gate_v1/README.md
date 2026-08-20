# ESPNet DroNet gate v1 release

This release selects the seed-2027 two-frame ESPNet model with a middle-stage
gate branch, four heatmaps, a binary gate mask, structured gate confidence,
and DroNet-compatible yaw/collision outputs.

`architecture_selection.json` is the frozen selection record. It compares the
complete deployable MAC counts and held-out navigation/gate metrics rather than
selecting on gate accuracy alone. The matching generated GAP8 C graphs and
weights live in NanoCockpit under
`app/networks/gap8-espnet-dronet-gate-v1`.

The `onnx/` directory contains a single end-to-end float ONNX for ONNX Runtime
and the five exact integer-domain ONNX graphs used by NeMO/DORY for GAP8.

The source float checkpoint used for export has SHA-256
`cd853cc97988af5308efc8323142ac3b96ed5fbe0393cfea93a2b8e331b0eda2`.
The generated GAP8 package is the portable release artifact; local workspace
checkpoint paths in the generation manifest are provenance only.

All training, quantization, calibration, evaluation, and GVSOC export jobs were
submitted through Slurm on `a2r-main` with their Conda environments stated in
the checked-in `.slurm` files.
