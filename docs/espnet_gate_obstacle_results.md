# ESPNet gate and obstacle result

All model fitting, quantization, and held-out evaluation in this comparison ran
through Slurm on `a2r-main`. Checkpoint selection used validation data only.

## Recommendation

Use epoch 11 of the safety-constrained full ESPNet
(`selected_for_deployment.pt`) as the accuracy teacher. Use the original
103,990-parameter `ESPNetDoryStudent`—not the residual-free control—for the
compact deployment candidate after QAT and safety re-evaluation.

| Model | Unsafe recall | Collision AUROC | Collision AP | Clearance MAE |
|---|---:|---:|---:|---:|
| Original obstacle ESPNet, test | 0.8867 | 0.9774 | 0.8774 | 0.0936 m |
| Joint gate + obstacle ESPNet epoch 11, test | **0.9106** | **0.9812** | **0.8921** | **0.0886 m** |

Epoch 11 also achieved synthetic gate IoU 0.5444, synthetic corner error
8.01 px, real-flight corner error 15.04 px, and no-gate mask pixel rate
0.00198 on validation. The gate confidence recommendation is structured fusion
of presence, mask geometry, and corner agreement: test AUROC 0.9585, AP 0.9638,
balanced accuracy 0.9066, and no-gate FPR 0.0533.

## Compact NEMO/DORY candidates

The float 103,990-parameter DORY student reached collision AUROC 0.9401, AP
0.6809, and recall 0.9721; gate IoU was 0.5913, but real corner error degraded
to 24.69 px.

| Integer candidate | Recall | FPR | AP | Danger IoU | Negative mask pixel rate |
|---|---:|---:|---:|---:|---:|
| PTQ | 1.0000 | 0.6528 | 0.2821 | 0.6971 | 0.00031 |
| Short NEMO QAT | 0.8883 | 0.2269 | 0.6587 | 0.0000 | 0.9999996 |

PTQ is excessively conservative, while short QAT collapses the dense danger and
gate outputs. The PTQ package is retained as a reproducible firmware-integration
artifact only. Promotion to flight requires retraining a quantization-native
student and passing the same validation-selected held-out safety gate.

## GAP8 generated-C gate

Depthwise convolution was not the blocker. GVSOC checksum tests pass for both
stride-1 and padded stride-2 depthwise layers. The failure was DORY's fused
8-bit residual lowering: it discarded the post-add multiplier and represented
two NEMO shifts as one. The compatibility patch in
`gap8_perception/patches/dory-gap8-exact-residual-requant.patch` preserves the
addition shift and applies the post-add multiplier and shift in the GAP8
kernel. With a 260 KB encoder activation arena, Slurm job 5588 passed exact
GVSOC checksums for all 52 generated layers: encoder 11, corner head 3, gate
head 3, and danger head 35. This includes every residual and both depthwise
stride modes. The packaged NanoCockpit GAP8 firmware also compiled. Its PTQ
test AP remains 0.2821 with a 0.6528 false-positive rate, so integer task
quality still requires QAT and held-out safety evaluation before flight
promotion.

Apply the pinned DORY patch before running the deployment Slurm job:

```bash
patch -d /home/cchen/dory -p1 --forward < \
  gap8_perception/patches/dory-gap8-exact-residual-requant.patch
sbatch gap8_perception/run_espnet_dory_deployment.slurm
```
