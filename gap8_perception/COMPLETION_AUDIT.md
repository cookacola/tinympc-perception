# Requirement audit

Last updated: 2026-07-27. “Complete” below means supported by an inspected
artifact or test. It does not imply real-flight safety.

## Dataset and targets

| Requirement | Status | Evidence |
|---|---|---|
| Correct 50,000-frame HM01B0 render | Complete | `workspace/course_hm01b0_nbd_hm01b0_v2_50k`: 50 shards, 50 success markers, 50,000 unique images and camera poses |
| Supplied intrinsics/distortion | Complete in render/labels | `configs/hm01b0_calibration.json`; fx 89.1558, fy 89.4608, cx 81.1038, cy 73.3473 |
| NewBeeDrone textures and lab course | Complete in simulation | v2 texture audit: 49,477 usable gate crops, median within-gate P90–P10 contrast 114 |
| 100-frame annotated inspection | Complete | `workspace/gap8_rollout_audit_hm01b0_v4/inspection_100.jpg` and `inspection_rollout_100.jpg` |
| Missing/duplicate/schema audit | Complete | `full_dataset_audit.json`: no missing, unreadable, schema, exact-duplicate, or pose-duplicate frames |
| Leakage-resistant 40k/5k/5k split | Complete at shard/seed level | `configs/split_shards.json` |
| Exact collision/clearance/TTC labels | Complete | `workspace/gap8_rollout_targets_hm01b0_v4`; swept sphere, 0.1 m drone radius, 0.1 m safety margin, 1 s horizon, 0.05 s timestep, 0.08 s latency |
| Motion dependence | Complete through controller postprocessor | Cached 0.5/1/5 m/s exact targets plus live speed/horizon/latency/range conditioning |

The 98.52% positive fraction at 5 m/s is a per-cell candidate-steering label
distribution, not a 98.52% flight crash rate.

## Model and evaluation

| Requirement | Status | Evidence |
|---|---|---|
| GAP8-friendly shared CNN | Complete | 7,784 parameters, 11.904 M MACs; convolution, depthwise/pointwise convolution, residual addition and ReLU |
| Ordered 40×40 TL/TR/BR/BL corners | Complete | model, target, augmentation and unit tests |
| 20×20 collision/range/uncertainty | Complete | packed output plus conservative reduction |
| Removable gate-opening head | Complete and retained | no-gate ablation improves danger IoU only 0.005 while worsening corners and false gate detections |
| Stage-A learning check | Complete | exact-label 16-frame overfit passes original per-head thresholds; 100 complex frames exceed this network’s memorization capacity |
| Float training/evaluation | Complete | `workspace/gap8_rollout_float_hm01b0_v4`; test corners 2.995 px, PCK@4 0.887, danger recall 0.944, false-safe 0.056, gate IoU 0.882 |
| QAT comparison | Complete | `workspace/gap8_rollout_qat_hm01b0_v4`; worse danger safety than float |
| Gate-head ablation | Complete | `workspace/gap8_rollout_float_no_gate_hm01b0_v4`; 4.66 px corners and 0.665 false gate rate |
| Quantized integer review samples | Complete | `workspace/quantized_review_samples_hm01b0_v4`: 32 PNG/NPZ pairs |
| Real-flight diagnostic | Complete but failing deployment quality | 12,630 gate-only frames; about 41.2 px mean raw corner error and visible sim-to-real mismatch |
| Real danger verification | Missing by dataset construction | Real data contains gates but no obstacles |
| Closed-loop A–F ablations | Not measured | Requires controller-in-loop simulation or physical flight |

## GAP8 and TinyMPC

| Requirement | Status | Evidence |
|---|---|---|
| Reproducible NeMO integer export | Complete | epoch 22, `PYTHONHASHSEED=9`; two independent ONNX files share SHA-256 `b764cc267f46debe198b9a68234f55a284907d8678f96dce7fbed2bb2f8cffc0` |
| Float/integer parity gate | Complete within accepted limits | 32 images: 3.471 heatmap-pixel corner displacement, 0.1601 danger probability MAE, 0.00413 gate probability MAE |
| DORY graph/tiling | Complete | 48 hardware nodes, 11.904 M MACs, maximum estimated L1 tile 36,240 B < 64 KiB |
| GVSOC final checksum | Skipped by user after excessive runtime | Slurm job 2683 was canceled after 38 minutes of active CPU simulation without terminal output; no checksum claim is made |
| NanoCockpit firmware build | Complete | `BUILD/GAP8_V2/GCC_RISCV/pulp_frontnet`; 172,692 B L2 usage (32.94%) |
| Crazyflie firmware build | Complete | official `bitcraze-builder.sif`; `build/cf21bl.bin`, 346 KiB |
| Motion-conditioned danger | Complete in software | speed, horizon, latency, map age, inverse range and uncertainty are combined on STM32 |
| Angular TinyMPC constraints | Complete in software | camera-centered half-spaces, position/velocity rows, selected future steps |
| Soft constraints/diagnostics | Complete | slack projection and maximum/total/cost/failure logs |
| Gate versus solid-obstacle distinction | Complete in software | synchronized gate-opening channel, inset polygon, min-pooling, range and uncertainty guards |
| Packet ABI | Complete | version-2 222-byte CRC packet, four uint4 10×10 maps, seqlock publication |
| Host tests | Complete | 27 perception tests and 14 controller integration tests pass |
| Physical flashing/latency/flight | Not measured | Requires AI-deck, GAP8/JTAG/Crazyflie and safe flight setup |

Obstacle constraints default active after the user explicitly accepted the
quantized accuracy risk. `percept.logOnly=1` remains available for tethered
bring-up.

## Delivery

- Compact checkpoints, metrics, samples, integer graph, reports, firmware,
  hashes and download commands:
  `workspace/gap8_download_bundle_hm01b0_v4`.
- GitHub publication is not yet complete because `gh auth status` reports no
  authenticated GitHub host. Local repositories and intended branches are
  ready for intentional commits after `gh auth login`.
