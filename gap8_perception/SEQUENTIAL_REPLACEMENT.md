# Sequential student replacement

The authoritative replacement for the former STDC/FPN model is
`SequentialSTDCNet` in `model_sequential.py`.

| Contract | Value |
|---|---|
| Input | NCHW `[N,1,120,160]` |
| Output | one NCHW tensor `[N,12,15,20]` |
| Parameters | 78,568 |
| MACs | 28,152,000 |
| Topology | stem, 3 stride-2 reductions, six 20×15 DW/PW blocks, linear 1×1 output |
| Graph plumbing | none: no resize, concat, residual, pooling, or dynamic shape |

The checked-in constants are generated for both Python and C by
`output_contract.py` and are mirrored in `include/perception_output_contract.h`.
The exact static report is `configs/sequential_architecture.json`.

## Slurm migration path

Generate geometry-backed labels first, then use `run_sequential_train.slurm`:

```bash
sbatch gap8_perception/run_sequential_targets.slurm
sbatch --export=ALL,RUN_TRAIN=1,DATASET=...,TARGETS=...,TRAIN_OUTPUT=... \
  gap8_perception/run_sequential_train.slurm
```

`generate_sequential_targets.py` records first intersection distances for the four
fixed body-forward normals and a per-direction visibility/confidence mask.
The specified initial ±45° directions lie just outside the calibrated HM01B0
crop, so the implementation uses FOV-matched `[-40, -13.33, 13.33, 40]°`;
this is the design document's required calibration-based angle adjustment.
The fixed-course source lacks vehicle attitude, so its label generator treats
the calibrated camera optical frame as body forward/left/up. New trajectory
data must provide calibrated camera-to-body extrinsics instead of this source
corpus fallback before flight acceptance.

After training, export the one-output structural ONNX graph with
`run_sequential_export.slurm`. The exporter folds BatchNorm and rejects graph
operators beyond Conv/ReLU/Identity. It is intentionally only a structural
gate; the pinned NeMO integer export, DORY lowering, GVSOC, and hardware
numerical-equivalence gates remain required.

The DORY deployment wrapper places the required PACT boundary after the
logical final score head. Firmware reconstructs logical scores as
`score = uint8 * 0.06313495337963104 - 6.0`; the accompanying
`quantization_manifest.json` is the authority for a particular export. This
keeps the deployed output at 3,600 bytes while preserving the shared score
domain. `validate_sequential_onnx_parity.py` checks the HWC NeMO fixture
against ONNX Runtime's NCHW tensor and reports every differing raw integer
element. The checked-in export has zero differing elements and zero maximum
LSB error. This is not a substitute for the GVSOC and physical-GAP8 parity
gates.

Legacy checkpoints and STDC/FPN export scripts are intentionally not loaded by
the replacement model. Their multi-output state dictionaries are incompatible
with this single-output ABI.

Corner supervision uses projected gate-frame centerline points, matching the
real labels. Target archives use schema `sequential_fixed_normal_v1`; archives
containing speed variants or dense danger fields are rejected by the dataset.
