# Two-frame ESPNet release v1

This release pins the two ESPNet checkpoints used by the selected accuracy and
GAP8 deployment paths. Both consume calibrated 160 x 160 HM01B0 grayscale
frames in `[previous, current]` order. Temporal training pairs came from the
same trajectory (and therefore the same flight) at adjacent frame indices.

## Which artifact to use

- `model/accuracy_teacher_epoch11.pt` is the selected full multitask ESPNet.
  Use it for accuracy experiments, continued mixed obstacle/gate training, and
  distillation. Selection used validation data only.
- `model/espnet_dory_student_epoch15.pt` is the 103,990-parameter float student
  used to produce the GAP8 candidate. It is useful for export and regression
  testing, but it is not itself the integer firmware network.
- The deployable network is
  `gap8-espnet-dory-student-v2-hybrid-qat` in the NanoCockpit repository. It
  uses the QAT encoder and danger branch with the better-calibrated PTQ corner
  and gate-mask branches.

The selected full model reached 0.9812 collision AUROC, 0.8921 collision AP,
0.9106 unsafe-clearance recall, and 0.0886 m clearance MAE on the held-out
obstacle test set. Its validation gate results were 0.5444 mask IoU, 8.01 px
synthetic corner error, and 15.04 px real-flight corner error.

The GAP8 hybrid passed all 52 NEMO-to-GVSOC layer checksums and compiled and
linked in NanoCockpit. Its held-out integer results were 0.9071 collision
AUROC, 0.4600 AP, 0.9916 recall, 0.6933 danger IoU, 0.5785 gate IoU, 7.03 px
synthetic corner error, and 22.64 px real-flight corner error. The complete
numbers and thresholds are under `metrics/`.

This remains a deployment candidate rather than a flight-approved model. In
particular, its held-out integer collision false-positive rate is 0.4719 at
the validation-selected threshold. Run the live-camera and tethered-flight
checks documented in the NanoCockpit deployment guide before free flight.
