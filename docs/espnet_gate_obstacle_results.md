# ESPNet gate and obstacle result

All model fitting, quantization, and held-out evaluation in this comparison ran
through Slurm on `a2r-main`. Checkpoint selection used validation data only.

## Recommendation

Use epoch 11 of the safety-constrained full ESPNet
(`selected_for_deployment.pt`). Do not fly either compact integer student.

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
