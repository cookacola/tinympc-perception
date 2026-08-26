# Motion-conditioned inverse-TTC release v1

This release pins the recommended motion-conditioned inverse-TTC network trained
on the 100k Isaac Sim kinematic corpus. It supersedes the image-only TTC model
for continued perception experiments.

## Selected checkpoint

- `model/ttc_motion_v1_epoch20.pt`
- validation-selected epoch: 20
- parameters: 29,191
- SHA-256: `9f25de2c5d302ae8776e36f487ef0bbbd16d8aa8c752521984936afc0c6a82b0`

The input ABI is two consecutive 160 x 160 grayscale images in
`[previous,current]` order plus ten deployable onboard values: body velocity
(3), angular velocity (3), gravity direction in body coordinates (3), and frame
interval (1). The dense output grid is 20 x 20 and contains inverse TTC,
inverse-depth and rigid-flow auxiliary predictions plus three risk logits.

## Held-out result

- inverse-TTC MAE: 0.15000 s^-1
- approaching inverse-TTC MAE: 0.16620 s^-1
- regression-derived risk-class accuracy: 0.90362
- regression-derived critical recall: 0.68818
- risk-head critical recall at the frozen validation threshold 0.552: 0.73973
- risk-head critical precision at that test operating point: 0.68540

The validation split met the declared 0.70 critical-precision target at the
frozen threshold, but the held-out test split reached 0.68540. This is recorded
as a limitation rather than silently recalibrating on test data.

`metrics/test_results.json` is the complete validation/test record. This model
is the parent checkpoint for the shared-encoder gate-head experiment; adding a
gate branch must preserve its existing TTC outputs unless a separately measured
joint fine-tuning stage is explicitly selected.
