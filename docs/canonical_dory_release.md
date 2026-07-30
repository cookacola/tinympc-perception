# Canonical shared-STDC DORY release

This is the release path for the 160x120 monochrome shared encoder, corner
heatmap head, and danger head. It follows the PULP-Frontnet contract: train in
PyTorch, QAT with the pinned NeMO environment, export a single integer source
of truth, generate with a pinned clean DORY checkout, and prove each generated
layer against the exported integer fixtures before packaging.

## Non-negotiable representation contract

- Input is unsigned 8-bit `120 x 160 x 1`.
- The encoder emits one unsigned 8-bit `30 x 40 x 32` tensor. Both heads
  consume that exact tensor and its exact scale.
- Each convolution must lower to the stock DORY BN form
  `clip((conv * kappa + lambda) >> shift)` with signed-int8 weights.
- Each residual addition must have compatible branch scales and no post-add
  multiplier. The stock GAP8 8-bit `pulp_nn_add` kernel does not implement one.
- Integer ONNX, golden `input.txt`/`out_layer*.txt`, DORY blobs, and GAP8 code
  are one release unit. Never mix them across exports.

The canonical gate deliberately rejects algebraically equivalent graphs that
need a custom lowering. In particular it rejects a `ReluQAddition` whose
generated `out_mult_vector` entry is not one. That forces the issue back into
QAT scale selection instead of silently producing incorrect C.

## Pipeline

1. Fine-tune the shared float checkpoint with NeMO fake quantization. Keep
   BatchNorm statistics fixed, train weights and PACT activation ranges, and
   include synthetic plus held-out real-flight images.
2. Select the QAT checkpoint only after float/fake-quant/int metrics pass.
3. Export encoder, corner head, and danger head from that one checkpoint.
   Clamp convolution weights to signed int8 before both ONNX serialization and
   DORY packing.
4. Run `run_stdc_shared_dory_repair.slurm`. It refuses a dirty DORY checkout,
   exports fixtures, runs DORY tiling, and runs
   `validate_canonical_dory_release.py` on every graph.
5. Run the layer-parity harness on GAP8. It must match ONNX layer CRCs for
   encoder, corner head, danger head, and combined output.
6. Package with `package_stdc_pair_nanocockpit.py --canonical-gate-dir ...`.
   Packaging refuses missing or failed canonical gates.

The current historical `int8fix` and `requantfix` packages are diagnostic
artifacts, not inputs to this release path. Do not flash them as a canonical
release.

## Acceptance record

Save these beside the release tag:

- pinned NeMO, DORY, GAP SDK, and NanoCockpit revisions;
- QAT checkpoint hash and dataset/split manifests;
- all three integer ONNX hashes;
- all three canonical-gate reports;
- packed-weight hashes;
- offline integer ONNX, independent DORY, and GAP8 CRCs for every layer.
