#!/usr/bin/env bash
#SBATCH --job-name=gap8_bundle
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.out
#SBATCH --error=/home/cchen/isaacsim-workspace/workspace/logs/%x-%j.err
set -euo pipefail

REPO=/home/cchen/isaacsim-workspace
BUNDLE=${BUNDLE_ROOT:-$REPO/workspace/gap8_download_bundle_hm01b0_v4}
FLOAT_ROOT=${FLOAT_ROOT:-$REPO/workspace/gap8_rollout_float_hm01b0_v4}
QAT_ROOT=${QAT_ROOT:-$REPO/workspace/gap8_rollout_qat_hm01b0_v4}
NO_GATE_ROOT=${NO_GATE_ROOT:-$REPO/workspace/gap8_rollout_float_no_gate_hm01b0_v4}
NEMO_ROOT=${NEMO_ROOT:-$REPO/workspace/gap8_rollout_nemo_dory_hm01b0_v4_epoch22_seed9}
NEMO_QAT=${NEMO_QAT:-$REPO/workspace/gap8_nemo_distill24_hm01b0_v4/epoch_22_nemo_qat.pt}
AUDIT_ROOT=${AUDIT_ROOT:-$REPO/workspace/gap8_rollout_audit_hm01b0_v4}
REAL_EVAL_ROOT=${REAL_EVAL_ROOT:-$FLOAT_ROOT/real_flight_evaluation}
CONTROLLER=/home/cchen/tinympc-crazyflie/apps/controller_tinympc_eigen
NANO=/home/cchen/tinympc-nanocockpit/src/gap/examples/pulp-frontnet

mkdir -p \
  "$BUNDLE/checkpoints" "$BUNDLE/metrics" "$BUNDLE/samples/float" \
  "$BUNDLE/samples/qat" "$BUNDLE/samples/no_gate" \
  "$BUNDLE/samples/real_flight" "$BUNDLE/samples/integer" \
  "$BUNDLE/audit" "$BUNDLE/export" \
  "$BUNDLE/firmware" "$BUNDLE/controller"

cp "$FLOAT_ROOT/best_total.pt" "$BUNDLE/checkpoints/float_best.pt"
cp "$QAT_ROOT/best_qat.pt" "$BUNDLE/checkpoints/qat_best.pt"
cp "$NEMO_QAT" "$BUNDLE/checkpoints/nemo_integer_epoch22.pt"
cp "$FLOAT_ROOT/log.csv" "$BUNDLE/metrics/float_log.csv"
cp "$QAT_ROOT/log.csv" "$BUNDLE/metrics/qat_log.csv"
cp "$FLOAT_ROOT/evaluation/validation_metrics.json" \
  "$BUNDLE/metrics/float_validation.json"
cp "$FLOAT_ROOT/evaluation/test_metrics.json" "$BUNDLE/metrics/float_test.json"
cp "$QAT_ROOT/evaluation/validation_metrics.json" \
  "$BUNDLE/metrics/qat_validation.json"
cp "$QAT_ROOT/evaluation/test_metrics.json" "$BUNDLE/metrics/qat_test.json"
cp "$FLOAT_ROOT/evaluation/predictions/"*.png "$BUNDLE/samples/float/"
cp "$QAT_ROOT/evaluation/predictions/"*.png "$BUNDLE/samples/qat/"
cp "$NEMO_ROOT/integer_parity_samples/"*.png "$BUNDLE/samples/integer/"
cp "$REPO/workspace/quantized_review_samples_hm01b0_v4/README.txt" \
  "$BUNDLE/samples/integer/"

if [[ -f "$NO_GATE_ROOT/best_total.pt" ]]; then
  cp "$NO_GATE_ROOT/best_total.pt" "$BUNDLE/checkpoints/no_gate_best.pt"
  cp "$NO_GATE_ROOT/log.csv" "$BUNDLE/metrics/no_gate_log.csv"
  cp "$NO_GATE_ROOT/evaluation/validation_metrics.json" \
    "$BUNDLE/metrics/no_gate_validation.json"
  cp "$NO_GATE_ROOT/evaluation/test_metrics.json" \
    "$BUNDLE/metrics/no_gate_test.json"
  cp "$NO_GATE_ROOT/evaluation/predictions/"*.png "$BUNDLE/samples/no_gate/"
fi

cp "$NEMO_ROOT/bridge/packed_float_state.npz" "$BUNDLE/export/"
cp "$NEMO_ROOT/bridge/bridge_report.json" "$BUNDLE/export/"
cp "$NEMO_ROOT/integer/gap8_packed_int.onnx" "$BUNDLE/export/"
cp "$NEMO_ROOT/integer/nemo_export_report.json" "$BUNDLE/export/"
cp "$NEMO_ROOT/dory_frontend_report.json" "$BUNDLE/export/"
cp "$NEMO_ROOT/gvsoc_checksum.log" "$BUNDLE/export/"
cp "$REPO/gap8_perception/GVSOC_STATUS.txt" "$BUNDLE/export/"

cp "$AUDIT_ROOT/target_summary.json" "$BUNDLE/audit/"
cp "$AUDIT_ROOT/full_dataset_audit.json" "$BUNDLE/audit/"
cp "$AUDIT_ROOT/inspection_100.jpg" "$BUNDLE/audit/"
cp "$AUDIT_ROOT/inspection_rollout_100.jpg" "$BUNDLE/audit/"
cp "$REAL_EVAL_ROOT/real_flight_metrics.json" \
  "$BUNDLE/metrics/real_flight_metrics.json"
cp "$REAL_EVAL_ROOT/predictions/real_flight_predictions_100.png" \
  "$BUNDLE/samples/real_flight/"
cp "$REPO/workspace/danger_ground_truth_example.png" "$BUNDLE/samples/"
cp "$REPO/workspace/danger_ground_truth_obstacle_example.png" \
  "$BUNDLE/samples/"

cp "$CONTROLLER/build/cf21bl.bin" "$BUNDLE/firmware/" 2>/dev/null || true
cp "$CONTROLLER/build/firmware.elf" "$BUNDLE/firmware/" 2>/dev/null || true
cp "$NANO/BUILD/GAP8_V2/GCC_RISCV/pulp_frontnet" \
  "$BUNDLE/firmware/nanocockpit_gap8" 2>/dev/null || true
cp "$CONTROLLER/src/perception_model_qparams.h" "$BUNDLE/controller/"
cp "$CONTROLLER/docs/nanocockpit_multitask_perception.md" \
  "$BUNDLE/controller/"
cp "$REPO/gap8_perception/NANOCOCKPIT_DEPLOYMENT.md" "$BUNDLE/"
cp "$REPO/gap8_perception/PIPELINE_STATUS.md" "$BUNDLE/"
cp "$REPO/gap8_perception/COMPLETION_AUDIT.md" "$BUNDLE/"

cp "$REPO/gap8_perception/DOWNLOAD_HM01B0_V4.txt" \
  "$BUNDLE/DOWNLOAD.txt"

(
  cd "$BUNDLE"
  find . -type f ! -name SHA256SUMS -print0 |
    sort -z | xargs -0 sha256sum >SHA256SUMS
)
du -sh "$BUNDLE"
