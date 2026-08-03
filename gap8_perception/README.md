# Canonical GAP8 racing perception pipeline

The authority for this repository is
[`docs/Completed_CNN_Design_Document.md`](../docs/Completed_CNN_Design_Document.md).
Checkpoints and former STDC/FPN or dense-danger releases are historical evidence,
not architecture specifications.

## Deployed contract

- Input: one `160x120` monochrome center crop.
- Network: `SequentialSTDCNet`, a sequential depthwise-separable CNN.
- Output: exactly one tensor, `NCHW [1,12,15,20]`.
- Channels `0..3`: ordered TL, TR, BR, BL corner score fields.
- Channels `4..7`: four fixed-normal free-distance fields.
- Channels `8..11`: four corresponding observation-confidence fields.
- Body-frame bearings: `[-40, -13.333333, 13.333333, 40]` degrees.
- No dense danger map, inverse-depth output, speed input, interpolation,
  concatenation, residual connection, or multiple terminal output.

Simulation and real corner labels use the physical gate-frame centerline,
halfway between the inner aperture and outer edge. The inner semantic hole may
be used only to associate a visible gate before projecting centerline labels.

## Canonical commands

All GPU work must be submitted through Slurm.

```bash
sbatch --export=ALL,DATASET=/path/to/corpus,TARGETS=/path/to/targets \
  gap8_perception/run_sequential_targets_75k.slurm

sbatch --export=ALL,RUN_TRAIN=1,DATASET=/path/to/corpus,TARGETS=/path/to/targets,\
TRAIN_OUTPUT=/path/to/output gap8_perception/run_sequential_train.slurm

sbatch --export=ALL,FLOAT_CHECKPOINT=/path/to/best.pt,OUTPUT=/path/to/qat \
  gap8_perception/run_sequential_qat.slurm
```

`generate_sequential_targets.py` produces only centerline corners with
per-corner visibility, four inflated-obstacle first-intersection distances,
and four observation-validity targets. It excludes legacy speed-indexed maps.

`controller_sequential.py` is the Python reference for decoding, confidence
rejection, safety margins, world-frame half-spaces, TinyMPC knot activation,
and reference-direction selection.

Historical architectures remain for reproducibility and are non-canonical.
They must not be imported by sequential run scripts.
